#include "rmsnorm_kernel.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

constexpr float kEpsilon = 1e-6f;
constexpr int kHidden = 1024;
constexpr uint64_t kSeed = 20260907ULL;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        const cudaError_t cuda_status = (call);                             \
        if (cuda_status != cudaSuccess) {                                   \
            throw std::runtime_error(std::string("CUDA error at ") + #call + \
                                     ": " + cudaGetErrorString(cuda_status)); \
        }                                                                   \
    } while (false)

struct Version {
    const char* name;
    void (*launch_fp16)(const void*, void*, const void*, int, int, float,
                        cudaStream_t);
    void (*launch_bf16)(const void*, void*, const void*, int, int, float,
                        cudaStream_t);
};

struct Metrics {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
    bool finite = false;
};

std::string g_current_dtype;
bool g_current_is_float16 = true;
std::string current_dtype_name() {
    return g_current_dtype;
}
bool dtype_is_float16() {
    return g_current_is_float16;
}

struct Trial {
    int trial_id = 0;
    std::string version;
    std::string dtype;
    std::string shape;
    double percall_mean_ms = 0.0;
    double percall_median_ms = 0.0;
    double percall_std_ms = 0.0;
    double percall_cv = 0.0;
    double amortized_mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

template <typename T>
std::vector<float> make_reference(const std::vector<T>& input,
                                  const std::vector<T>& weight, int tokens,
                                  int hidden) {
    std::vector<float> reference(static_cast<size_t>(tokens) * hidden);
    for (int token = 0; token < tokens; ++token) {
        double sum_squares = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const float value = static_cast<float>(input[token * hidden + i]);
            sum_squares += static_cast<double>(value) * value;
        }
        const float mean = static_cast<float>(sum_squares / hidden);
        const float inv_rms = 1.0f / std::sqrt(mean + kEpsilon);
        for (int i = 0; i < hidden; ++i) {
            const size_t index = static_cast<size_t>(token) * hidden + i;
            reference[index] = static_cast<float>(input[index]) * inv_rms *
                               static_cast<float>(weight[i]);
        }
    }
    return reference;
}

Metrics compare(const std::vector<float>& reference,
                const std::vector<float>& candidate) {
    if (reference.size() != candidate.size()) {
        throw std::runtime_error("reference/candidate size mismatch");
    }
    double max_abs = 0.0;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    bool finite = std::isfinite(squared_error) && std::isfinite(squared_reference);
    for (size_t i = 0; i < reference.size(); ++i) {
        const double difference = static_cast<double>(candidate[i]) -
                                  static_cast<double>(reference[i]);
        max_abs = std::max(max_abs, std::abs(difference));
        squared_error += difference * difference;
        squared_reference += static_cast<double>(reference[i]) *
                             static_cast<double>(reference[i]);
        finite = finite && std::isfinite(reference[i]) && std::isfinite(candidate[i]);
    }
    Metrics metrics;
    metrics.max_abs = max_abs;
    metrics.relative_l2 = squared_reference > 0.0
                              ? std::sqrt(squared_error / squared_reference)
                              : std::sqrt(squared_error);
    metrics.finite = finite;
    return metrics;
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    }
    return values[mid];
}

double standard_deviation(const std::vector<double>& values, double sample_mean) {
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (const double value : values) {
        const double delta = value - sample_mean;
        sum += delta * delta;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

void write_csv(const fs::path& path, const std::string& content,
               bool append = false) {
    fs::create_directories(path.parent_path());
    const std::ios::openmode mode = append ? std::ios::app : std::ios::out;
    std::ofstream output(path, mode);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path.string());
    }
    output << content;
}

template <typename T>
void run_case(const std::string& dtype_name, int tokens,
              const std::vector<Version>& versions, int warmup, int repetitions,
              int trials, const fs::path& output_dir) {
    std::vector<T> input(static_cast<size_t>(tokens) * kHidden);
    std::vector<T> weight(kHidden);
    std::mt19937_64 generator(kSeed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (T& value : input) {
        value = static_cast<T>(distribution(generator));
    }
    for (T& value : weight) {
        value = static_cast<T>(distribution(generator));
    }

    T* device_input = nullptr;
    T* device_output = nullptr;
    T* device_weight = nullptr;
    const size_t byte_count = input.size() * sizeof(T);
    const size_t weight_bytes = weight.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(&device_input, byte_count));
    CUDA_CHECK(cudaMalloc(&device_output, byte_count));
    CUDA_CHECK(cudaMalloc(&device_weight, weight_bytes));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), byte_count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_weight, weight.data(), weight_bytes,
                          cudaMemcpyHostToDevice));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const std::string shape = "[1," + std::to_string(tokens) + "," +
                              std::to_string(kHidden) + "]";

    for (const Version& version : versions) {
        auto launch = [&]() {
            if (dtype_name == "float16") {
                version.launch_fp16(device_input, device_output, device_weight,
                                    tokens, kHidden, kEpsilon, stream);
            } else {
                version.launch_bf16(device_input, device_output, device_weight,
                                    tokens, kHidden, kEpsilon, stream);
            }
        };

        for (int i = 0; i < warmup; ++i) {
            launch();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<double> percall_values;
        percall_values.reserve(static_cast<size_t>(trials) * repetitions);
        std::vector<double> amortized_values;
        amortized_values.reserve(trials);
        for (int trial = 0; trial < trials; ++trial) {
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                cudaEvent_t start;
                cudaEvent_t stop;
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&stop));
                CUDA_CHECK(cudaEventRecord(start, stream));
                launch();
                CUDA_CHECK(cudaEventRecord(stop, stream));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float milliseconds = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                percall_values.push_back(static_cast<double>(milliseconds));
                CUDA_CHECK(cudaEventDestroy(start));
                CUDA_CHECK(cudaEventDestroy(stop));
            }

            cudaEvent_t amortized_start;
            cudaEvent_t amortized_stop;
            CUDA_CHECK(cudaEventCreate(&amortized_start));
            CUDA_CHECK(cudaEventCreate(&amortized_stop));
            CUDA_CHECK(cudaEventRecord(amortized_start, stream));
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                launch();
            }
            CUDA_CHECK(cudaEventRecord(amortized_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(amortized_stop));
            float amortized_milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&amortized_milliseconds,
                                            amortized_start, amortized_stop));
            amortized_values.push_back(static_cast<double>(amortized_milliseconds) /
                                       repetitions);
            CUDA_CHECK(cudaEventDestroy(amortized_start));
            CUDA_CHECK(cudaEventDestroy(amortized_stop));
            const std::vector<double> trial_values(percall_values.end() - repetitions,
                                                   percall_values.end());
            const double trial_mean = mean(trial_values);
            const double trial_median = median(trial_values);
            const double trial_std = standard_deviation(trial_values, trial_mean);
            const double trial_cv = trial_mean > 0.0 ? trial_std / trial_mean : 0.0;
            const double trial_amortized = amortized_values.back();
            const double trial_min = *std::min_element(trial_values.begin(),
                                                       trial_values.end());
            const double trial_max = *std::max_element(trial_values.begin(),
                                                       trial_values.end());
            std::ostringstream trial_csv;
            trial_csv << std::setprecision(10) << trial + 1 << ','
                      << version.name << ',' << dtype_name << ',' << shape << ','
                      << trial_mean << ',' << trial_median << ',' << trial_std
                      << ',' << trial_cv << ',' << trial_amortized << ','
                      << trial_min << ',' << trial_max << '\n';
            write_csv(output_dir / "benchmark_trials.csv", trial_csv.str(), true);
        }

        const double percall_mean = mean(percall_values);
        const double percall_median = median(percall_values);
        const double percall_std = standard_deviation(percall_values, percall_mean);
        const double percall_cv = percall_mean > 0.0 ? percall_std / percall_mean : 0.0;
        const double amortized_mean = mean(amortized_values);
        const double minimum = *std::min_element(percall_values.begin(),
                                                 percall_values.end());
        const double maximum = *std::max_element(percall_values.begin(),
                                                 percall_values.end());
        std::ostringstream summary_csv;
        summary_csv << std::setprecision(10) << version.name << ',' << dtype_name
                    << ',' << shape << ',' << percall_values.size() << ','
                    << percall_mean << ',' << percall_median << ',' << percall_std
                    << ',' << percall_cv << ',' << amortized_mean << ','
                    << minimum << ',' << maximum << '\n';
        write_csv(output_dir / "benchmark_summary.csv", summary_csv.str(), true);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_weight));
}

void run_case_dispatch(const std::string& dtype_name, int tokens,
                       const std::vector<Version>& versions, int warmup,
                       int repetitions, int trials, const fs::path& output_dir) {
    if (dtype_name == "float16") {
        run_case<__half>(dtype_name, tokens, versions, warmup, repetitions,
                         trials, output_dir);
    } else {
        run_case<__nv_bfloat16>(dtype_name, tokens, versions, warmup,
                                repetitions, trials, output_dir);
    }
}

template <typename T>
void run_correctness(int tokens, const std::vector<Version>& versions,
                     const fs::path& output_dir) {
    std::vector<T> input(static_cast<size_t>(tokens) * kHidden);
    std::vector<T> weight(kHidden);
    std::mt19937_64 generator(kSeed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (T& value : input) {
        value = static_cast<T>(distribution(generator));
    }
    for (T& value : weight) {
        value = static_cast<T>(distribution(generator));
    }
    const std::vector<float> reference = make_reference(input, weight, tokens, kHidden);

    T* device_input = nullptr;
    T* device_output = nullptr;
    T* device_weight = nullptr;
    const size_t byte_count = input.size() * sizeof(T);
    const size_t weight_bytes = weight.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(&device_input, byte_count));
    CUDA_CHECK(cudaMalloc(&device_output, byte_count));
    CUDA_CHECK(cudaMalloc(&device_weight, weight_bytes));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), byte_count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_weight, weight.data(), weight_bytes,
                          cudaMemcpyHostToDevice));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const std::string shape = "[1," + std::to_string(tokens) + "," +
                              std::to_string(kHidden) + "]";
    for (const Version& version : versions) {
        CUDA_CHECK(cudaMemset(device_output, 0, byte_count));
        if (input.size() * sizeof(T) != byte_count) {
            throw std::runtime_error("size invariant failed");
        }
        if (dtype_is_float16()) {
            version.launch_fp16(device_input, device_output, device_weight,
                                tokens, kHidden, kEpsilon, stream);
        } else {
            version.launch_bf16(device_input, device_output, device_weight,
                                tokens, kHidden, kEpsilon, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<T> output(input.size());
        CUDA_CHECK(cudaMemcpy(output.data(), device_output, byte_count,
                              cudaMemcpyDeviceToHost));
        std::vector<float> candidate(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            candidate[i] = static_cast<float>(output[i]);
        }
        const Metrics metrics = compare(reference, candidate);
        const bool gate = metrics.finite && metrics.relative_l2 <= 0.005;
        std::ostringstream csv;
        csv << std::setprecision(12) << version.name << ',' << current_dtype_name()
            << ',' << shape << ',' << metrics.max_abs << ',' << metrics.relative_l2
            << ',' << (metrics.finite ? "true" : "false") << ','
            << (gate ? "PASS" : "FAIL") << '\n';
        write_csv(output_dir / "correctness_raw.csv", csv.str(), true);
        std::ostringstream summary;
        summary << current_dtype_name() << ',' << shape << ',' << version.name
                << ',' << metrics.max_abs << ',' << metrics.relative_l2 << ','
                << (metrics.finite ? "true" : "false") << ','
                << (gate ? "PASS" : "FAIL") << '\n';
        write_csv(output_dir / "correctness_summary.csv", summary.str(), true);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_weight));
}

void run_correctness_dispatch(const std::string& dtype_name, int tokens,
                              const std::vector<Version>& versions,
                              const fs::path& output_dir) {
    g_current_dtype = dtype_name;
    g_current_is_float16 = dtype_name == "float16";
    if (g_current_is_float16) {
        run_correctness<__half>(tokens, versions, output_dir);
    } else {
        run_correctness<__nv_bfloat16>(tokens, versions, output_dir);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int warmup = 50;
    int repetitions = 200;
    int trials = 5;
    std::string dtype = "all";
    fs::path output_dir = "phase8_1_results";
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto require_value = [&](const char* name) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        if (argument == "--warmup") {
            warmup = std::stoi(require_value("--warmup"));
        } else if (argument == "--repetitions") {
            repetitions = std::stoi(require_value("--repetitions"));
        } else if (argument == "--trials") {
            trials = std::stoi(require_value("--trials"));
        } else if (argument == "--dtype") {
            dtype = require_value("--dtype");
        } else if (argument == "--output-dir") {
            output_dir = require_value("--output-dir");
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (warmup < 0 || repetitions <= 0 || trials <= 0) {
        throw std::runtime_error("invalid benchmark protocol");
    }
    if (dtype != "all" && dtype != "float16" && dtype != "bfloat16") {
        throw std::runtime_error("dtype must be all, float16, or bfloat16");
    }

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int driver_version = 0;
    int runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    std::ostringstream header;
    header << "device=" << properties.name << "\n"
           << "compute_capability=" << properties.major << "." << properties.minor
           << "\n"
           << "multiprocessors=" << properties.multiProcessorCount << "\n"
           << "cuda_driver_version=" << driver_version << "\n"
           << "cuda_runtime_version=" << runtime_version << "\n"
           << "hidden=" << kHidden << "\n"
           << "epsilon=" << kEpsilon << "\n"
           << "seed=" << kSeed << "\n"
           << "weight_source=deterministic_normal\n"
           << "warmup=" << warmup << "\n"
           << "repetitions=" << repetitions << "\n"
           << "trials=" << trials << "\n"
           << "timing_method=cuda_event\n"
           << "percall_timing=event pair around each kernel\n"
           << "amortized_timing=one event pair around " << repetitions
           << " kernels\n";
    write_csv(output_dir / "run_metadata.txt", header.str());

    const std::vector<Version> versions = {
        {"V0", reinterpret_cast<void (*)(const void*, void*, const void*, int,
                                         int, float, cudaStream_t)>(
                   &launch_rmsnorm_v0<__half>),
               reinterpret_cast<void (*)(const void*, void*, const void*, int,
                                         int, float, cudaStream_t)>(
                   &launch_rmsnorm_v0<__nv_bfloat16>)},
        {"V1", reinterpret_cast<void (*)(const void*, void*, const void*, int,
                                         int, float, cudaStream_t)>(
                   &launch_rmsnorm_v1<__half>),
               reinterpret_cast<void (*)(const void*, void*, const void*, int,
                                         int, float, cudaStream_t)>(
                   &launch_rmsnorm_v1<__nv_bfloat16>)},
        {"V2", reinterpret_cast<void (*)(const void*, void*, const void*, int,
                                         int, float, cudaStream_t)>(
                   &launch_rmsnorm_v2<__half>),
               reinterpret_cast<void (*)(const void*, void*, const void*, int,
                                         int, float, cudaStream_t)>(
                   &launch_rmsnorm_v2<__nv_bfloat16>)},
    };

    write_csv(output_dir / "correctness_raw.csv",
              "version,dtype,shape,max_abs,relative_l2,finite,gate\n");
    write_csv(output_dir / "correctness_summary.csv",
              "dtype,shape,version,max_abs,relative_l2,finite,gate\n");
    write_csv(output_dir / "benchmark_trials.csv",
              "trial,version,dtype,shape,percall_mean_ms,percall_median_ms,"
              "percall_std_ms,percall_cv,amortized_mean_ms,min_ms,max_ms\n");
    write_csv(output_dir / "benchmark_summary.csv",
              "version,dtype,shape,repetitions,mean_ms,median_ms,std_ms,cv,"
              "amortized_mean_ms,min_ms,max_ms\n");

    const int shape_tokens[2] = {8, 1};
    const std::string shape_names[2] = {"prefill_s8", "decode_s1"};
    for (const std::string& current_dtype :
         (dtype == "all" ? std::vector<std::string>{"float16", "bfloat16"}
                         : std::vector<std::string>{dtype})) {
        for (int case_index = 0; case_index < 2; ++case_index) {
            run_correctness_dispatch(current_dtype, shape_tokens[case_index],
                                     versions, output_dir);
            run_case_dispatch(current_dtype, shape_tokens[case_index], versions,
                              warmup, repetitions, trials, output_dir);
        }
    }
    return 0;
}
