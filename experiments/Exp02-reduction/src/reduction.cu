#include "reduction_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status_ = (call);                                     \
        if (status_ != cudaSuccess) {                                           \
            std::ostringstream message_;                                        \
            message_ << "CUDA error at " << __FILE__ << ':' << __LINE__       \
                     << ": " << cudaGetErrorString(status_) << " ("          \
                     << static_cast<int>(status_) << ')';                      \
            throw std::runtime_error(message_.str());                           \
        }                                                                       \
    } while (0)

namespace {

constexpr int kSupportedBlocks[] = {32, 64, 128, 256, 512, 1024};
constexpr std::uint32_t kPatternSeed = 0x02C0FFEEu;
constexpr double kEpsilon = 1.0 / 8388608.0;

enum class Pattern { Ones, Signed, Cancellation };

const char* versionName(ReductionVersion version) {
    switch (version) {
        case ReductionVersion::V1GlobalMultiPass: return "V1";
        case ReductionVersion::V2SharedModulo: return "V2";
        case ReductionVersion::V3SharedIndexed: return "V3";
        case ReductionVersion::V4SharedSequential: return "V4";
        case ReductionVersion::V5FirstAddLoad: return "V5";
        case ReductionVersion::V6SharedWarpTail: return "V6";
        case ReductionVersion::V7Shuffle: return "V7";
    }
    return "unknown";
}

const char* patternName(Pattern pattern) {
    switch (pattern) {
        case Pattern::Ones: return "ones";
        case Pattern::Signed: return "signed_seeded";
        case Pattern::Cancellation: return "cancellation";
    }
    return "unknown";
}

bool supportedBlock(int block) {
    for (int candidate : kSupportedBlocks) {
        if (candidate == block) return true;
    }
    return false;
}

ReductionVersion parseVersion(int value) {
    if (value < 1 || value > 7) {
        throw std::invalid_argument("version must be in [1, 7]");
    }
    return static_cast<ReductionVersion>(value);
}

Pattern parsePattern(const std::string& value) {
    if (value == "ones") return Pattern::Ones;
    if (value == "signed") return Pattern::Signed;
    if (value == "cancellation") return Pattern::Cancellation;
    throw std::invalid_argument("pattern must be ones, signed, or cancellation");
}

std::string utcTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') escaped += "\"\"";
        else escaped += ch;
    }
    escaped += '"';
    return escaped;
}

std::vector<float> makeInput(std::size_t n, Pattern pattern) {
    if (n == 0) throw std::invalid_argument("N must be positive");
    std::vector<float> input(n);
    std::mt19937 rng(kPatternSeed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
        if (pattern == Pattern::Ones) {
            input[i] = 1.0f;
        } else if (pattern == Pattern::Signed) {
            input[i] = distribution(rng);
        } else {
            const float magnitude =
                0.25f + static_cast<float>(i % 17) * 0.03125f;
            input[i] = (i & 1U) == 0U ? magnitude : -magnitude;
        }
    }
    return input;
}

double referenceSum(const std::vector<float>& input) {
    double sum = 0.0;
    for (float value : input) sum += static_cast<double>(value);
    return sum;
}

double absoluteSum(const std::vector<float>& input) {
    double sum = 0.0;
    for (float value : input) sum += std::abs(static_cast<double>(value));
    return sum;
}

double toleranceFor(std::size_t n, double sumAbs) {
    const double k =
        std::ceil(std::log2(static_cast<double>(std::max<std::size_t>(n, 2)))) +
        2.0;
    const double gamma = (k * kEpsilon) / (1.0 - k * kEpsilon);
    return 8.0 * gamma * sumAbs + 1.0e-6;
}

struct DeviceBuffers {
    float* input = nullptr;
    float* ping = nullptr;
    float* pong = nullptr;

    void allocate(std::size_t count) {
        const std::size_t bytes = count * sizeof(float);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&input), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ping), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pong), bytes));
    }

    void release() {
        if (pong != nullptr) {
            CUDA_CHECK(cudaFree(pong));
            pong = nullptr;
        }
        if (ping != nullptr) {
            CUDA_CHECK(cudaFree(ping));
            ping = nullptr;
        }
        if (input != nullptr) {
            CUDA_CHECK(cudaFree(input));
            input = nullptr;
        }
    }

    ~DeviceBuffers() {
        if (pong != nullptr) cudaFree(pong);
        if (ping != nullptr) cudaFree(ping);
        if (input != nullptr) cudaFree(input);
    }
};

float executeVersion(ReductionVersion version, const std::vector<float>& host,
                     int blockSize) {
    if (!supportedBlock(blockSize)) {
        throw std::invalid_argument(
            "unsupported block size; supported values: 32,64,128,256,512,1024");
    }
    DeviceBuffers device;
    device.allocate(host.size());
    CUDA_CHECK(cudaMemcpy(device.input, host.data(),
                          host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::size_t count = host.size();
    const float* current = device.input;
    float* output = device.ping;
    float* spare = device.pong;
    while (count > 1) {
        const std::size_t next = reductionOutputCount(version, count, blockSize);
        CUDA_CHECK(launchReduction(version, current, output, count, blockSize));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        count = next;
        current = output;
        std::swap(output, spare);
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, current, sizeof(float),
                          cudaMemcpyDeviceToHost));
    device.release();
    return result;
}

struct Result {
    double reference = 0.0;
    double gpu = 0.0;
    double sumAbs = 0.0;
    double absError = 0.0;
    double normalizedError = 0.0;
    double tolerance = 0.0;
    bool pass = false;
    std::string error;
};

Result runOne(ReductionVersion version, Pattern pattern, std::size_t n,
              int blockSize) {
    const std::vector<float> input = makeInput(n, pattern);
    Result result;
    result.reference = referenceSum(input);
    result.sumAbs = absoluteSum(input);
    result.tolerance = toleranceFor(n, result.sumAbs);
    result.gpu = executeVersion(version, input, blockSize);
    result.absError = std::abs(result.gpu - result.reference);
    result.normalizedError = result.absError / std::max(result.sumAbs, 1.0);
    result.pass = result.absError <= result.tolerance;
    if (pattern == Pattern::Ones && result.gpu != static_cast<float>(n)) {
        result.pass = false;
        result.error = "all-ones exact result mismatch";
    }
    return result;
}

struct Row {
    std::string timestamp;
    ReductionVersion version{};
    Pattern pattern{};
    std::size_t n = 0;
    int block = 0;
    int repeat = 0;
    Result result;
};

void writeRawHeader(std::ostream& out) {
    out << "timestamp,version,pattern,seed,N,block_size,repeat,reference_sum,"
           "gpu_sum,absolute_error,sum_abs,normalized_error,tolerance,"
           "runtime_success,correctness,error_message\n";
}

void writeRawRow(std::ostream& out, const Row& row) {
    out << row.timestamp << ',' << versionName(row.version) << ','
        << patternName(row.pattern) << ',' << kPatternSeed << ',' << row.n
        << ',' << row.block << ',' << row.repeat << ',' << std::setprecision(17)
        << row.result.reference << ',' << row.result.gpu << ','
        << row.result.absError << ',' << row.result.sumAbs << ','
        << row.result.normalizedError << ',' << row.result.tolerance << ','
        << (row.result.error.empty() ? "true" : "false") << ','
        << (row.result.pass ? "PASS" : "FAIL") << ','
        << csvEscape(row.result.error) << '\n';
}

std::vector<std::size_t> sizesForBlock(int block) {
    std::vector<std::size_t> sizes = {
        1U,
        static_cast<std::size_t>(block - 1),
        static_cast<std::size_t>(block),
        static_cast<std::size_t>(block + 1),
        static_cast<std::size_t>(2 * block - 1),
        static_cast<std::size_t>(2 * block),
        static_cast<std::size_t>(2 * block + 1),
        static_cast<std::size_t>(17 * block + 13),
    };
    if (block == 256) sizes.push_back(1048576U + 13U);
    return sizes;
}

void writeSummary(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open summary CSV: " + path);
    out << "version,executions,passes,fails,max_absolute_error,"
           "max_absolute_error_case,max_normalized_error,"
           "max_normalized_error_case\n";
    for (int versionValue = 1; versionValue <= 7; ++versionValue) {
        const ReductionVersion version = parseVersion(versionValue);
        std::size_t executions = 0;
        std::size_t passes = 0;
        const Row* maxAbsRow = nullptr;
        const Row* maxNormRow = nullptr;
        for (const Row& row : rows) {
            if (row.version != version) continue;
            ++executions;
            if (row.result.pass) ++passes;
            if (maxAbsRow == nullptr ||
                row.result.absError > maxAbsRow->result.absError) {
                maxAbsRow = &row;
            }
            if (maxNormRow == nullptr ||
                row.result.normalizedError >
                    maxNormRow->result.normalizedError) {
                maxNormRow = &row;
            }
        }
        auto caseName = [](const Row* row) {
            if (row == nullptr) return std::string();
            std::ostringstream name;
            name << patternName(row->pattern) << ":N=" << row->n
                 << ":B=" << row->block << ":repeat=" << row->repeat;
            return name.str();
        };
        out << versionName(version) << ',' << executions << ',' << passes << ','
            << executions - passes << ',' << std::setprecision(17)
            << (maxAbsRow == nullptr ? 0.0 : maxAbsRow->result.absError) << ','
            << csvEscape(caseName(maxAbsRow)) << ','
            << (maxNormRow == nullptr ? 0.0
                                      : maxNormRow->result.normalizedError)
            << ',' << csvEscape(caseName(maxNormRow)) << '\n';
    }
}

void runCorrectness(const std::string& rawPath,
                    const std::string& summaryPath) {
    std::ofstream raw(rawPath);
    if (!raw) throw std::runtime_error("cannot open raw CSV: " + rawPath);
    writeRawHeader(raw);
    std::vector<Row> rows;
    bool allPassed = true;
    const Pattern patterns[] = {Pattern::Ones, Pattern::Signed,
                                Pattern::Cancellation};
    for (int block : kSupportedBlocks) {
        for (std::size_t n : sizesForBlock(block)) {
            for (Pattern pattern : patterns) {
                for (int versionValue = 1; versionValue <= 7; ++versionValue) {
                    const ReductionVersion version = parseVersion(versionValue);
                    for (int repeat = 1; repeat <= 3; ++repeat) {
                        Row row;
                        row.timestamp = utcTimestamp();
                        row.version = version;
                        row.pattern = pattern;
                        row.n = n;
                        row.block = block;
                        row.repeat = repeat;
                        try {
                            row.result = runOne(version, pattern, n, block);
                        } catch (const std::exception& error) {
                            row.result.error = error.what();
                            row.result.pass = false;
                        }
                        writeRawRow(raw, row);
                        rows.push_back(row);
                        allPassed = allPassed && row.result.pass;
                    }
                }
            }
        }
        std::cout << "completed_block=" << block << '\n';
    }
    raw.close();
    writeSummary(summaryPath, rows);
    std::cout << "correctness_executions=" << rows.size() << '\n'
              << "correctness=" << (allPassed ? "PASS" : "FAIL") << '\n';
    if (!allPassed) throw std::runtime_error("correctness Gate A failed");
}

struct BenchmarkResult {
    double meanMs = 0.0;
    double medianMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    double stdMs = 0.0;
    double cvPct = 0.0;
    float gpu = 0.0f;
    double reference = 0.0;
    double absError = 0.0;
    double normalizedError = 0.0;
    double tolerance = 0.0;
    std::size_t firstGrid = 0;
    std::size_t passCount = 0;
    std::size_t launchCount = 0;
    bool correct = false;
};

BenchmarkResult benchmarkSingle(ReductionVersion version, int blockSize,
                                std::size_t n, int warmup, int repetitions) {
    if (!supportedBlock(blockSize) || n == 0 || warmup < 0 || repetitions <= 0)
        throw std::invalid_argument("invalid benchmark configuration");
    const std::vector<float> host = makeInput(n, Pattern::Signed);
    DeviceBuffers device;
    device.allocate(n);
    CUDA_CHECK(cudaMemcpy(device.input, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const std::size_t firstOutput = reductionOutputCount(version, n, blockSize);
    const std::size_t firstGrid = version == ReductionVersion::V1GlobalMultiPass
                                      ? (firstOutput + blockSize - 1) / blockSize
                                      : firstOutput;
    const std::size_t passCount = [&]() {
        std::size_t count = n, passes = 0;
        while (count > 1) { count = reductionOutputCount(version, count, blockSize); ++passes; }
        return passes;
    }();
    const auto launchPipeline = [&]() {
        std::size_t count = n;
        const float* current = device.input;
        float* output = device.ping;
        float* spare = device.pong;
        while (count > 1) {
            CUDA_CHECK(launchReduction(version, current, output, count, blockSize));
            count = reductionOutputCount(version, count, blockSize);
            current = output;
            std::swap(output, spare);
        }
        return current;
    };
    for (int i = 0; i < warmup; ++i) (void)launchPipeline();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<cudaEvent_t> starts(static_cast<std::size_t>(repetitions));
    std::vector<cudaEvent_t> stops(static_cast<std::size_t>(repetitions));
    const float* finalOutput = nullptr;
    for (int i = 0; i < repetitions; ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));
        CUDA_CHECK(cudaEventRecord(starts[i]));
        finalOutput = launchPipeline();
        CUDA_CHECK(cudaEventRecord(stops[i]));
    }
    CUDA_CHECK(cudaEventSynchronize(stops.back()));
    std::vector<double> latencies;
    latencies.reserve(static_cast<std::size_t>(repetitions));
    for (int i = 0; i < repetitions; ++i) {
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, starts[i], stops[i]));
        latencies.push_back(static_cast<double>(elapsed));
        CUDA_CHECK(cudaEventDestroy(stops[i]));
        CUDA_CHECK(cudaEventDestroy(starts[i]));
    }
    float gpu = 0.0f;
    CUDA_CHECK(cudaMemcpy(&gpu, finalOutput, sizeof(float), cudaMemcpyDeviceToHost));
    const double reference = referenceSum(host);
    const double sumAbs = absoluteSum(host);
    const double absError = std::abs(static_cast<double>(gpu) - reference);
    std::sort(latencies.begin(), latencies.end());
    const double sumLatency = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double meanLatency = sumLatency / latencies.size();
    double squared = 0.0;
    for (double latency : latencies) squared += (latency - meanLatency) * (latency - meanLatency);
    const double stdLatency = latencies.size() > 1 ? std::sqrt(squared / (latencies.size() - 1)) : 0.0;
    BenchmarkResult result;
    result.meanMs = meanLatency;
    result.medianMs = latencies.size() % 2 ? latencies[latencies.size() / 2] : (latencies[latencies.size() / 2 - 1] + latencies[latencies.size() / 2]) / 2.0;
    result.minMs = latencies.front();
    result.maxMs = latencies.back();
    result.stdMs = stdLatency;
    result.cvPct = meanLatency > 0.0 ? 100.0 * stdLatency / meanLatency : 0.0;
    result.gpu = gpu;
    result.reference = reference;
    result.absError = absError;
    result.normalizedError = absError / std::max(sumAbs, 1.0);
    result.tolerance = toleranceFor(n, sumAbs);
    result.firstGrid = firstGrid;
    result.passCount = passCount;
    result.launchCount = passCount * static_cast<std::size_t>(repetitions);
    result.correct = absError <= result.tolerance;
    device.release();
    return result;
}

void appendBenchmarkRow(const std::string& path, ReductionVersion version,
                        Pattern pattern, std::size_t n, int block, int round,
                        int warmup, int repetitions,
                        const BenchmarkResult& result) {
    const bool exists = static_cast<bool>(std::ifstream(path));
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot open benchmark CSV: " + path);
    if (!exists) out << "timestamp,version,pattern,N,block_size,round,warmup,repetitions,mean_latency_ms,median_latency_ms,min_latency_ms,max_latency_ms,sample_std_latency_ms,cv_pct,reference_sum,gpu_sum,absolute_error,normalized_error,tolerance,correctness,first_stage_grid,total_pass_count,total_kernel_launch_count\n";
    out << utcTimestamp() << ',' << versionName(version) << ',' << patternName(pattern) << ',' << n << ',' << block << ',' << round << ',' << warmup << ',' << repetitions << ',' << std::setprecision(12) << result.meanMs << ',' << result.medianMs << ',' << result.minMs << ',' << result.maxMs << ',' << result.stdMs << ',' << result.cvPct << ',' << std::setprecision(17) << result.reference << ',' << result.gpu << ',' << result.absError << ',' << result.normalizedError << ',' << result.tolerance << ',' << (result.correct ? "PASS" : "FAIL") << ',' << result.firstGrid << ',' << result.passCount << ',' << result.launchCount << '\n';
}
void printDeviceInfo() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int runtimeVersion = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    std::cout << "gpu_name=" << prop.name << '\n'
              << "compute_capability=" << prop.major << '.' << prop.minor
              << '\n'
              << "cuda_runtime_version=" << runtimeVersion << '\n'
              << "sm_count=" << prop.multiProcessorCount << '\n'
              << "warp_size=" << prop.warpSize << '\n'
              << "max_threads_per_block=" << prop.maxThreadsPerBlock << '\n'
              << "supported_blocks=32,64,128,256,512,1024\n";
}

}  // namespace

__global__ void reductionGlobalPass(const float* input, float* output,
                                    std::size_t count) {
    const std::size_t outputIndex =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t first = outputIndex * 2;
    if (first < count) {
        float value = input[first];
        if (first + 1 < count) value += input[first + 1];
        output[outputIndex] = value;
    }
}

__global__ void reductionSharedModulo(const float* input, float* output,
                                      std::size_t count) {
    extern __shared__ float values[];
    const unsigned tid = threadIdx.x;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    values[tid] = index < count ? input[index] : 0.0f;
    __syncthreads();
    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = values[0];
}

__global__ void reductionSharedIndexed(const float* input, float* output,
                                       std::size_t count) {
    extern __shared__ float values[];
    const unsigned tid = threadIdx.x;
    const std::size_t inputIndex =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    values[tid] = inputIndex < count ? input[inputIndex] : 0.0f;
    __syncthreads();
    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        const unsigned index = 2 * stride * tid;
        if (index < blockDim.x) values[index] += values[index + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = values[0];
}

__global__ void reductionSharedSequential(const float* input, float* output,
                                          std::size_t count) {
    extern __shared__ float values[];
    const unsigned tid = threadIdx.x;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    values[tid] = index < count ? input[index] : 0.0f;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = values[0];
}

__global__ void reductionFirstAddLoad(const float* input, float* output,
                                      std::size_t count) {
    extern __shared__ float values[];
    const unsigned tid = threadIdx.x;
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + tid;
    float value = first < count ? input[first] : 0.0f;
    if (first + blockDim.x < count) value += input[first + blockDim.x];
    values[tid] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = values[0];
}

__global__ void reductionSharedWarpTail(const float* input, float* output,
                                        std::size_t count) {
    extern __shared__ float values[];
    const unsigned tid = threadIdx.x;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    values[tid] = index < count ? input[index] : 0.0f;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {
        if (blockDim.x >= 64) values[tid] += values[tid + 32];
        __syncwarp();
        for (unsigned stride = 16; stride > 0; stride /= 2) {
            if (tid < stride) values[tid] += values[tid + stride];
            __syncwarp();
        }
    }
    if (tid == 0) output[blockIdx.x] = values[0];
}

__device__ float warpReduceMasked(float value, unsigned mask) {
    const unsigned lane = threadIdx.x & 31U;
    const unsigned activeLanes = __popc(mask);
    for (int offset = 16; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(mask, value, offset);
        if (lane + static_cast<unsigned>(offset) < activeLanes) value += other;
    }
    return value;
}

__global__ void reductionShuffle(const float* input, float* output,
                                 std::size_t count) {
    extern __shared__ float warpSums[];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31U;
    const unsigned warp = tid / 32U;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const bool valid = index < count;
    const unsigned warpMask = __ballot_sync(__activemask(), valid);
    if (valid) {
        float value = warpReduceMasked(input[index], warpMask);
        if (lane == 0) warpSums[warp] = value;
    } else if (warpMask == 0 && lane == 0) {
        warpSums[warp] = 0.0f;
    }
    __syncthreads();

    const unsigned warpCount = (blockDim.x + 31U) / 32U;
    if (warp == 0) {
        const bool hasWarpSum = lane < warpCount;
        const unsigned aggregateMask =
            __ballot_sync(__activemask(), hasWarpSum);
        if (hasWarpSum) {
            const float blockValue =
                warpReduceMasked(warpSums[lane], aggregateMask);
            if (lane == 0) output[blockIdx.x] = blockValue;
        }
    }
}

std::size_t reductionOutputCount(ReductionVersion version, std::size_t count,
                                 int blockSize) {
    if (version == ReductionVersion::V1GlobalMultiPass) return (count + 1) / 2;
    if (version == ReductionVersion::V5FirstAddLoad) {
        return (count + 2 * static_cast<std::size_t>(blockSize) - 1) /
               (2 * static_cast<std::size_t>(blockSize));
    }
    return (count + static_cast<std::size_t>(blockSize) - 1) /
           static_cast<std::size_t>(blockSize);
}

cudaError_t launchReduction(ReductionVersion version, const float* input,
                            float* output, std::size_t count, int blockSize,
                            cudaStream_t stream) {
    if (count == 0 || !supportedBlock(blockSize)) return cudaErrorInvalidValue;
    const std::size_t outputs = reductionOutputCount(version, count, blockSize);
    const unsigned grid = static_cast<unsigned>(
        (version == ReductionVersion::V1GlobalMultiPass)
            ? (outputs + static_cast<std::size_t>(blockSize) - 1) / blockSize
            : outputs);
    const dim3 block(static_cast<unsigned>(blockSize));
    const std::size_t sharedBytes =
        version == ReductionVersion::V7Shuffle
            ? ((blockSize + 31) / 32) * sizeof(float)
            : blockSize * sizeof(float);
    switch (version) {
        case ReductionVersion::V1GlobalMultiPass:
            reductionGlobalPass<<<grid, block, 0, stream>>>(input, output, count);
            break;
        case ReductionVersion::V2SharedModulo:
            reductionSharedModulo<<<grid, block, sharedBytes, stream>>>(
                input, output, count);
            break;
        case ReductionVersion::V3SharedIndexed:
            reductionSharedIndexed<<<grid, block, sharedBytes, stream>>>(
                input, output, count);
            break;
        case ReductionVersion::V4SharedSequential:
            reductionSharedSequential<<<grid, block, sharedBytes, stream>>>(
                input, output, count);
            break;
        case ReductionVersion::V5FirstAddLoad:
            reductionFirstAddLoad<<<grid, block, sharedBytes, stream>>>(
                input, output, count);
            break;
        case ReductionVersion::V6SharedWarpTail:
            reductionSharedWarpTail<<<grid, block, sharedBytes, stream>>>(
                input, output, count);
            break;
        case ReductionVersion::V7Shuffle:
            reductionShuffle<<<grid, block, sharedBytes, stream>>>(input, output,
                                                                   count);
            break;
    }
    return cudaPeekAtLastError();
}

int main(int argc, char** argv) {
    try {
        std::string mode;
        std::string rawPath;
        std::string summaryPath;
        std::size_t n = 0;
        int block = 0;
        int versionValue = 0;
        int warmup = 20;
        int repetitions = 100;
        int round = 1;
        Pattern pattern = Pattern::Ones;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto requireValue = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("missing option value");
                return argv[i];
            };
            if (arg == "--device-info") mode = "device-info";
            else if (arg == "--correctness") mode = "correctness";
            else if (arg == "--single") mode = "single";
            else if (arg == "--benchmark-single") mode = "benchmark-single";
            else if (arg == "--raw") rawPath = requireValue();
            else if (arg == "--summary") summaryPath = requireValue();
            else if (arg == "--n") n = std::stoull(requireValue());
            else if (arg == "--block") block = std::stoi(requireValue());
            else if (arg == "--version") versionValue = std::stoi(requireValue());
            else if (arg == "--pattern") pattern = parsePattern(requireValue());
            else if (arg == "--warmup") warmup = std::stoi(requireValue());
            else if (arg == "--repetitions") repetitions = std::stoi(requireValue());
            else if (arg == "--round") round = std::stoi(requireValue());
            else throw std::invalid_argument("unknown option: " + arg);
        }
        if (mode == "device-info") {
            printDeviceInfo();
            return 0;
        }
        if (mode == "benchmark-single") {
            if (rawPath.empty() || n == 0 || block == 0 || versionValue == 0) throw std::invalid_argument("--benchmark-single requires --raw, --version, --n and --block");
            const ReductionVersion version = parseVersion(versionValue);
            const BenchmarkResult result = benchmarkSingle(version, block, n, warmup, repetitions);
            appendBenchmarkRow(rawPath, version, pattern, n, block, round, warmup, repetitions, result);
            return result.correct ? 0 : 2;
        }
        if (mode == "correctness") {
            if (rawPath.empty() || summaryPath.empty()) {
                throw std::invalid_argument(
                    "--correctness requires --raw PATH --summary PATH");
            }
            runCorrectness(rawPath, summaryPath);
            return 0;
        }
        if (mode == "single") {
            if (n == 0 || block == 0 || versionValue == 0) {
                throw std::invalid_argument(
                    "--single requires --version, --pattern, --n and --block");
            }
            const ReductionVersion version = parseVersion(versionValue);
            const Result result = runOne(version, pattern, n, block);
            std::cout << "version=" << versionName(version) << '\n'
                      << "pattern=" << patternName(pattern) << '\n'
                      << "N=" << n << '\n'
                      << "block_size=" << block << '\n'
                      << std::setprecision(17)
                      << "reference_sum=" << result.reference << '\n'
                      << "gpu_sum=" << result.gpu << '\n'
                      << "absolute_error=" << result.absError << '\n'
                      << "sum_abs=" << result.sumAbs << '\n'
                      << "normalized_error=" << result.normalizedError << '\n'
                      << "tolerance=" << result.tolerance << '\n'
                      << "correctness=" << (result.pass ? "PASS" : "FAIL")
                      << '\n';
            return result.pass ? 0 : 2;
        }
        throw std::invalid_argument(
            "select --device-info, --correctness, or --single");
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}

