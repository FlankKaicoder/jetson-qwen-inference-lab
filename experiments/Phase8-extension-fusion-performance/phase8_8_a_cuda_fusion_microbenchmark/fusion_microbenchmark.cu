#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kHidden = 1024;
constexpr int kOutput = 3072;
constexpr int kThreads = 256;
constexpr int kOutputsPerBlock = kThreads / 32;
constexpr float kEpsilon = 1e-6f;
constexpr uint64_t kDefaultSeed = 20260920ULL;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                       \
        if (status != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error at ") + #call +   \
                                     ": " + cudaGetErrorString(status));       \
        }                                                                       \
    } while (false)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                        \
        const cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            throw std::runtime_error(std::string("cuBLAS error at ") + #call + \
                                     ": status=" + std::to_string(status));    \
        }                                                                       \
    } while (false)

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void rmsnorm_fp16_kernel(const __half* __restrict__ input,
                                    const __half* __restrict__ gamma,
                                    __half* __restrict__ normalized) {
    __shared__ float warp_sums[kThreads / 32];
    __shared__ float inv_rms;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sum = 0.0f;
    for (int index = tid; index < kHidden; index += kThreads) {
        const float value = __half2float(input[index]);
        sum += value * value;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < kThreads / 32 ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            inv_rms = rsqrtf(block_sum / static_cast<float>(kHidden) + kEpsilon);
        }
    }
    __syncthreads();
    for (int index = tid; index < kHidden; index += kThreads) {
        normalized[index] = __float2half_rn(
            __half2float(input[index]) * inv_rms * __half2float(gamma[index]));
    }
}

__global__ void fused_rmsnorm_linear_fp16_kernel(
    const __half* __restrict__ input, const __half* __restrict__ gamma,
    const __half* __restrict__ weight, __half* __restrict__ output) {
    __shared__ __half normalized[kHidden];
    __shared__ float warp_sums[kThreads / 32];
    __shared__ float inv_rms;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float sum = 0.0f;
    for (int index = tid; index < kHidden; index += kThreads) {
        const float value = __half2float(input[index]);
        sum += value * value;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < kThreads / 32 ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            inv_rms = rsqrtf(block_sum / static_cast<float>(kHidden) + kEpsilon);
        }
    }
    __syncthreads();
    for (int index = tid; index < kHidden; index += kThreads) {
        normalized[index] = __float2half_rn(
            __half2float(input[index]) * inv_rms * __half2float(gamma[index]));
    }
    __syncthreads();

    // Each block reuses one normalized vector for several output tiles. This
    // is intentionally simple and avoids a global normalized activation.
    for (int out = blockIdx.x * kOutputsPerBlock + warp; out < kOutput;
         out += gridDim.x * kOutputsPerBlock) {
        float dot = 0.0f;
        const size_t row = static_cast<size_t>(out) * kHidden;
        for (int index = lane; index < kHidden; index += 32) {
            dot += __half2float(normalized[index]) *
                   __half2float(weight[row + index]);
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            output[out] = __float2half_rn(dot);
        }
    }
}

struct Metrics {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
    double cosine = 0.0;
    bool finite = true;
};

struct Summary {
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

Summary summarize(std::vector<double> values) {
    Summary result;
    result.mean = mean(values);
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    result.median = values.size() % 2 == 0
                        ? (values[middle - 1] + values[middle]) / 2.0
                        : values[middle];
    result.minimum = values.front();
    result.maximum = values.back();
    double variance = 0.0;
    for (const double value : values) {
        const double delta = value - result.mean;
        variance += delta * delta;
    }
    result.stddev = values.size() > 1
                        ? std::sqrt(variance / static_cast<double>(values.size() - 1))
                        : 0.0;
    return result;
}

Metrics compare(const std::vector<float>& reference,
                const std::vector<__half>& candidate) {
    if (reference.size() != candidate.size()) {
        throw std::runtime_error("comparison size mismatch");
    }
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double squared_candidate = 0.0;
    double dot = 0.0;
    Metrics result;
    for (size_t index = 0; index < reference.size(); ++index) {
        const double lhs = reference[index];
        const double rhs = __half2float(candidate[index]);
        result.finite = result.finite && std::isfinite(lhs) && std::isfinite(rhs);
        const double difference = rhs - lhs;
        result.max_abs = std::max(result.max_abs, std::abs(difference));
        squared_error += difference * difference;
        squared_reference += lhs * lhs;
        squared_candidate += rhs * rhs;
        dot += lhs * rhs;
    }
    result.relative_l2 = squared_reference > 0.0
                             ? std::sqrt(squared_error / squared_reference)
                             : std::sqrt(squared_error);
    result.cosine = squared_reference > 0.0 && squared_candidate > 0.0
                        ? dot / std::sqrt(squared_reference * squared_candidate)
                        : 0.0;
    return result;
}

void write_text(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    output << content;
}

class Benchmark {
public:
    Benchmark(uint64_t seed) {
        CUDA_CHECK(cudaGetDevice(&device_));
        CUDA_CHECK(cudaGetDeviceProperties(&properties_, device_));
        fused_blocks_ = std::min((kOutput + kOutputsPerBlock - 1) /
                                     kOutputsPerBlock,
                                 properties_.multiProcessorCount * 4);

        std::mt19937_64 generator(seed);
        std::normal_distribution<float> input_distribution(0.0f, 1.0f);
        std::normal_distribution<float> gamma_distribution(1.0f, 0.05f);
        std::normal_distribution<float> weight_distribution(0.0f, 0.02f);
        host_input_.resize(kHidden);
        host_gamma_.resize(kHidden);
        host_weight_.resize(static_cast<size_t>(kOutput) * kHidden);
        for (auto& value : host_input_) {
            value = __float2half_rn(input_distribution(generator));
        }
        for (auto& value : host_gamma_) {
            value = __float2half_rn(gamma_distribution(generator));
        }
        for (auto& value : host_weight_) {
            value = __float2half_rn(weight_distribution(generator));
        }

        CUDA_CHECK(cudaMalloc(&input_, kHidden * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&gamma_, kHidden * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&normalized_, kHidden * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&weight_, host_weight_.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&output_, kOutput * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(input_, host_input_.data(), kHidden * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gamma_, host_gamma_.data(), kHidden * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(weight_, host_weight_.data(),
                              host_weight_.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetStream(handle_, stream_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    }

    ~Benchmark() {
        if (handle_) cublasDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
        if (input_) cudaFree(input_);
        if (gamma_) cudaFree(gamma_);
        if (normalized_) cudaFree(normalized_);
        if (weight_) cudaFree(weight_);
        if (output_) cudaFree(output_);
    }

    void launch_baseline() {
        rmsnorm_fp16_kernel<<<1, kThreads, 0, stream_>>>(input_, gamma_, normalized_);
        CUDA_CHECK(cudaGetLastError());
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(
            handle_, CUBLAS_OP_T, CUBLAS_OP_N, kOutput, 1, kHidden, &alpha,
            weight_, CUDA_R_16F, kHidden, normalized_, CUDA_R_16F, kHidden,
            &beta, output_, CUDA_R_16F, kOutput, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    void launch_fusion() {
        fused_rmsnorm_linear_fp16_kernel<<<fused_blocks_, kThreads, 0, stream_>>>(
            input_, gamma_, weight_, output_);
        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

    std::vector<__half> copy_output() {
        synchronize();
        std::vector<__half> output(kOutput);
        CUDA_CHECK(cudaMemcpy(output.data(), output_, kOutput * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        return output;
    }

    std::vector<float> reference() const {
        double sum_squares = 0.0;
        for (const __half value : host_input_) {
            const double converted = __half2float(value);
            sum_squares += converted * converted;
        }
        const float inv_rms = 1.0f /
            std::sqrt(static_cast<float>(sum_squares / kHidden) + kEpsilon);
        std::vector<__half> normalized(kHidden);
        for (int index = 0; index < kHidden; ++index) {
            normalized[index] = __float2half_rn(
                __half2float(host_input_[index]) * inv_rms *
                __half2float(host_gamma_[index]));
        }
        std::vector<float> output(kOutput, 0.0f);
        for (int out = 0; out < kOutput; ++out) {
            float sum = 0.0f;
            const size_t row = static_cast<size_t>(out) * kHidden;
            for (int index = 0; index < kHidden; ++index) {
                sum += __half2float(normalized[index]) *
                       __half2float(host_weight_[row + index]);
            }
            output[out] = sum;
        }
        return output;
    }

    double time(bool fusion, int repetitions) {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream_));
        for (int index = 0; index < repetitions; ++index) {
            fusion ? launch_fusion() : launch_baseline();
        }
        CUDA_CHECK(cudaEventRecord(stop, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return static_cast<double>(milliseconds) / repetitions;
    }

    const cudaDeviceProp& properties() const { return properties_; }
    int fused_blocks() const { return fused_blocks_; }

private:
    int device_ = 0;
    int fused_blocks_ = 0;
    cudaDeviceProp properties_{};
    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    __half* input_ = nullptr;
    __half* gamma_ = nullptr;
    __half* normalized_ = nullptr;
    __half* weight_ = nullptr;
    __half* output_ = nullptr;
    std::vector<__half> host_input_;
    std::vector<__half> host_gamma_;
    std::vector<__half> host_weight_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        int warmup = 50;
        int repetitions = 500;
        int trials = 10;
        int profile_warmup = 20;
        uint64_t seed = kDefaultSeed;
        fs::path output_dir = "phase8_8a_results";
        std::string profile_mode = "none";
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto value = [&]() {
                if (index + 1 >= argc) {
                    throw std::runtime_error("missing value for " + argument);
                }
                return std::string(argv[++index]);
            };
            if (argument == "--warmup") warmup = std::stoi(value());
            else if (argument == "--repetitions") repetitions = std::stoi(value());
            else if (argument == "--trials") trials = std::stoi(value());
            else if (argument == "--profile-warmup") profile_warmup = std::stoi(value());
            else if (argument == "--seed") seed = std::stoull(value());
            else if (argument == "--output-dir") output_dir = value();
            else if (argument == "--profile-mode") profile_mode = value();
            else throw std::runtime_error("unknown argument: " + argument);
        }
        if (warmup < 0 || repetitions <= 0 || trials <= 0 || profile_warmup < 0) {
            throw std::runtime_error("invalid benchmark counts");
        }
        if (profile_mode != "none" && profile_mode != "baseline" &&
            profile_mode != "fusion") {
            throw std::runtime_error("profile mode must be none, baseline, or fusion");
        }

        Benchmark benchmark(seed);
        if (profile_mode != "none") {
            const bool fusion = profile_mode == "fusion";
            for (int index = 0; index < profile_warmup; ++index) {
                fusion ? benchmark.launch_fusion() : benchmark.launch_baseline();
            }
            benchmark.synchronize();
            CUDA_CHECK(cudaProfilerStart());
            fusion ? benchmark.launch_fusion() : benchmark.launch_baseline();
            benchmark.synchronize();
            CUDA_CHECK(cudaProfilerStop());
            std::cout << "profile_mode=" << profile_mode << " launches=1\n";
            return 0;
        }

        const std::vector<float> reference = benchmark.reference();
        benchmark.launch_baseline();
        const std::vector<__half> baseline_output = benchmark.copy_output();
        benchmark.launch_fusion();
        const std::vector<__half> fusion_output = benchmark.copy_output();
        const Metrics baseline_metrics = compare(reference, baseline_output);
        const Metrics fusion_metrics = compare(reference, fusion_output);
        std::vector<float> baseline_as_float(baseline_output.size());
        for (size_t index = 0; index < baseline_output.size(); ++index) {
            baseline_as_float[index] = __half2float(baseline_output[index]);
        }
        const Metrics cross_metrics = compare(baseline_as_float, fusion_output);
        const bool baseline_gate = baseline_metrics.finite &&
                                   baseline_metrics.relative_l2 <= 0.01;
        const bool fusion_gate = fusion_metrics.finite &&
                                 fusion_metrics.relative_l2 <= 0.01;
        const bool cross_gate = cross_metrics.finite &&
                                cross_metrics.relative_l2 <= 0.01;

        std::ostringstream correctness;
        correctness << "path,max_abs,relative_l2,cosine,finite,gate\n"
                    << std::setprecision(12)
                    << "baseline," << baseline_metrics.max_abs << ','
                    << baseline_metrics.relative_l2 << ','
                    << baseline_metrics.cosine << ','
                    << (baseline_metrics.finite ? "true" : "false") << ','
                    << (baseline_gate ? "PASS" : "FAIL") << '\n'
                    << "fusion," << fusion_metrics.max_abs << ','
                    << fusion_metrics.relative_l2 << ','
                    << fusion_metrics.cosine << ','
                    << (fusion_metrics.finite ? "true" : "false") << ','
                    << (fusion_gate ? "PASS" : "FAIL") << '\n'
                    << "fusion_vs_baseline," << cross_metrics.max_abs << ','
                    << cross_metrics.relative_l2 << ',' << cross_metrics.cosine
                    << ',' << (cross_metrics.finite ? "true" : "false") << ','
                    << (cross_gate ? "PASS" : "FAIL") << '\n';
        write_text(output_dir / "correctness.csv", correctness.str());
        if (!(baseline_gate && fusion_gate && cross_gate)) {
            throw std::runtime_error("correctness gate failed; benchmark not run");
        }

        for (int index = 0; index < warmup; ++index) {
            benchmark.launch_baseline();
            benchmark.launch_fusion();
        }
        benchmark.synchronize();

        std::vector<double> baseline_times;
        std::vector<double> fusion_times;
        std::ostringstream trial_csv;
        trial_csv << "trial,order,path,repetitions,per_call_ms\n";
        for (int trial = 0; trial < trials; ++trial) {
            const bool fusion_first = trial % 2 == 1;
            const std::string order = fusion_first ? "fusion-baseline" :
                                                     "baseline-fusion";
            double baseline_ms = 0.0;
            double fusion_ms = 0.0;
            if (fusion_first) {
                fusion_ms = benchmark.time(true, repetitions);
                baseline_ms = benchmark.time(false, repetitions);
            } else {
                baseline_ms = benchmark.time(false, repetitions);
                fusion_ms = benchmark.time(true, repetitions);
            }
            baseline_times.push_back(baseline_ms);
            fusion_times.push_back(fusion_ms);
            trial_csv << std::setprecision(12) << trial + 1 << ',' << order
                      << ",baseline," << repetitions << ',' << baseline_ms << '\n'
                      << trial + 1 << ',' << order << ",fusion," << repetitions
                      << ',' << fusion_ms << '\n';
        }
        write_text(output_dir / "benchmark_trials.csv", trial_csv.str());

        const Summary baseline = summarize(baseline_times);
        const Summary fusion = summarize(fusion_times);
        std::ostringstream summary;
        summary << "path,trials,repetitions_per_trial,mean_ms,median_ms,std_ms,min_ms,max_ms\n"
                << std::setprecision(12)
                << "baseline," << trials << ',' << repetitions << ','
                << baseline.mean << ',' << baseline.median << ',' << baseline.stddev
                << ',' << baseline.minimum << ',' << baseline.maximum << '\n'
                << "fusion," << trials << ',' << repetitions << ',' << fusion.mean
                << ',' << fusion.median << ',' << fusion.stddev << ','
                << fusion.minimum << ',' << fusion.maximum << '\n';
        write_text(output_dir / "benchmark_summary.csv", summary.str());

        int driver = 0;
        int runtime = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        std::ostringstream metadata;
        metadata << "device=" << benchmark.properties().name << '\n'
                 << "compute_capability=" << benchmark.properties().major << '.'
                 << benchmark.properties().minor << '\n'
                 << "multiprocessors=" << benchmark.properties().multiProcessorCount
                 << '\n'
                 << "cuda_driver_version=" << driver << '\n'
                 << "cuda_runtime_version=" << runtime << '\n'
                 << "dtype=FP16\nshape=M1_K1024_N3072\n"
                 << "epsilon=" << kEpsilon << '\n'
                 << "seed=" << seed << '\n'
                 << "warmup_pairs=" << warmup << '\n'
                 << "trials=" << trials << '\n'
                 << "repetitions_per_trial=" << repetitions << '\n'
                 << "timing=cuda_event_batched_per_call\n"
                 << "trial_order=alternating_AB_BA\n"
                 << "baseline=rmsnorm_fp16_kernel+cublasGemmEx\n"
                 << "fusion=fused_rmsnorm_linear_fp16_kernel\n"
                 << "fused_blocks=" << benchmark.fused_blocks() << '\n'
                 << "threads_per_block=" << kThreads << '\n';
        write_text(output_dir / "run_metadata.txt", metadata.str());
        std::cout << summary.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
