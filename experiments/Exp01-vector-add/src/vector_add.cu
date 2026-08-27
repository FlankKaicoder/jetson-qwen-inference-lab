#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        const cudaError_t status_ = (call);                                       \
        if (status_ != cudaSuccess) {                                             \
            std::ostringstream message_;                                         \
            message_ << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": " \
                     << cudaGetErrorString(status_) << " ("                      \
                     << static_cast<int>(status_) << ')';                        \
            throw std::runtime_error(message_.str());                            \
        }                                                                        \
    } while (0)

__global__ void vectorAddKernel(const float* a, const float* b, float* c,
                                std::size_t n) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

namespace {

constexpr int kBlockCandidates[] = {16, 32, 64, 128, 256, 512, 1024};
constexpr std::size_t kCorrectnessSizes[] = {
    1, 31, 32, 33, 255, 256, 257, 1000, 1024, 1025, 1048576};
constexpr std::size_t kBenchmarkSizes[] = {1048576, 4194304, 16777216};
constexpr std::size_t kStabilityN = 16777216;
constexpr int kStabilityBlocks[] = {32, 64, 128, 256, 512, 1024};
constexpr int kStabilityOrders[][6] = {
    {32, 64, 128, 256, 512, 1024},
    {128, 256, 512, 1024, 32, 64},
    {512, 1024, 32, 64, 128, 256},
    {256, 128, 64, 32, 1024, 512},
    {1024, 512, 256, 128, 64, 32},
};

struct DeviceBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;

    void allocate(std::size_t bytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c), bytes));
    }

    void release() {
        if (c != nullptr) {
            CUDA_CHECK(cudaFree(c));
            c = nullptr;
        }
        if (b != nullptr) {
            CUDA_CHECK(cudaFree(b));
            b = nullptr;
        }
        if (a != nullptr) {
            CUDA_CHECK(cudaFree(a));
            a = nullptr;
        }
    }

    ~DeviceBuffers() {
        if (c != nullptr) cudaFree(c);
        if (b != nullptr) cudaFree(b);
        if (a != nullptr) cudaFree(a);
    }
};

struct TimingEvents {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    void create() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    void release() {
        if (stop != nullptr) {
            CUDA_CHECK(cudaEventDestroy(stop));
            stop = nullptr;
        }
        if (start != nullptr) {
            CUDA_CHECK(cudaEventDestroy(start));
            start = nullptr;
        }
    }

    ~TimingEvents() {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
    }
};

struct CaseResult {
    std::size_t n = 0;
    int blockSize = 0;
    std::uint64_t gridSize = 0;
    std::uint64_t totalThreads = 0;
    int warpsPerBlock = 0;
    int activeBlocksPerSm = 0;
    int activeWarpsPerSm = 0;
    double theoreticalOccupancyPct = 0.0;
    int warmup = 0;
    int repetitions = 0;
    float avgKernelLatencyMs = 0.0f;
    double effectiveBandwidthGBps = 0.0;
    double maxAbsError = 0.0;
    bool correct = false;
};

std::string utcTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string versionString(int version) {
    std::ostringstream out;
    out << version / 1000 << '.' << (version % 1000) / 10;
    return out.str();
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

std::vector<int> supportedBlocks(const cudaDeviceProp& prop) {
    std::vector<int> blocks;
    for (int block : kBlockCandidates) {
        if (block <= prop.maxThreadsPerBlock && block <= prop.maxThreadsDim[0]) {
            blocks.push_back(block);
        }
    }
    return blocks;
}

CaseResult runCase(std::size_t n, int blockSize, int warmup, int repetitions,
                   const cudaDeviceProp& prop) {
    if (n == 0) throw std::invalid_argument("N must be positive");
    if (blockSize <= 0) throw std::invalid_argument("block size must be positive");
    if (warmup < 0) throw std::invalid_argument("warmup must be non-negative");
    if (repetitions <= 0) {
        throw std::invalid_argument("repetitions must be positive");
    }
    if (blockSize > prop.maxThreadsPerBlock ||
        blockSize > prop.maxThreadsDim[0]) {
        throw std::invalid_argument("block size is unsupported by this device");
    }

    const std::uint64_t gridSize =
        (static_cast<std::uint64_t>(n) + blockSize - 1) / blockSize;
    if (gridSize > static_cast<std::uint64_t>(prop.maxGridSize[0])) {
        throw std::invalid_argument("grid size exceeds maxGridSize.x");
    }

    std::vector<float> hostA(n);
    std::vector<float> hostB(n);
    std::vector<float> hostC(n);
    std::vector<float> hostReference(n);
    for (std::size_t i = 0; i < n; ++i) {
        hostA[i] = static_cast<float>(i % 1024) * 0.001f;
        hostB[i] = static_cast<float>((i * 7) % 2048) * 0.0005f;
        hostReference[i] = hostA[i] + hostB[i];
    }

    const std::size_t bytes = n * sizeof(float);
    DeviceBuffers device;
    device.allocate(bytes);
    CUDA_CHECK(cudaMemcpy(device.a, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.b, hostB.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        vectorAddKernel<<<static_cast<unsigned int>(gridSize), blockSize>>>(
            device.a, device.b, device.c, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    TimingEvents events;
    events.create();
    CUDA_CHECK(cudaEventRecord(events.start));
    for (int i = 0; i < repetitions; ++i) {
        vectorAddKernel<<<static_cast<unsigned int>(gridSize), blockSize>>>(
            device.a, device.b, device.c, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(events.stop));
    CUDA_CHECK(cudaEventSynchronize(events.stop));

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, events.start, events.stop));
    const float averageMs = totalMs / static_cast<float>(repetitions);

    CUDA_CHECK(cudaMemcpy(hostC.data(), device.c, bytes, cudaMemcpyDeviceToHost));
    double maxAbsError = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        maxAbsError =
            std::max(maxAbsError, std::abs(static_cast<double>(hostC[i]) -
                                           static_cast<double>(hostReference[i])));
    }

    int activeBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocks, vectorAddKernel, blockSize, 0));
    const int warpsPerBlock = (blockSize + prop.warpSize - 1) / prop.warpSize;
    const int activeWarps = activeBlocks * warpsPerBlock;
    const int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    CaseResult result;
    result.n = n;
    result.blockSize = blockSize;
    result.gridSize = gridSize;
    result.totalThreads = gridSize * static_cast<std::uint64_t>(blockSize);
    result.warpsPerBlock = warpsPerBlock;
    result.activeBlocksPerSm = activeBlocks;
    result.activeWarpsPerSm = activeWarps;
    result.theoreticalOccupancyPct =
        maxWarps > 0 ? 100.0 * activeWarps / maxWarps : 0.0;
    result.warmup = warmup;
    result.repetitions = repetitions;
    result.avgKernelLatencyMs = averageMs;
    result.effectiveBandwidthGBps =
        (3.0 * static_cast<double>(n) * sizeof(float)) /
        (static_cast<double>(averageMs) * 1.0e6);
    result.maxAbsError = maxAbsError;
    result.correct = maxAbsError <= 1.0e-6;

    events.release();
    device.release();
    return result;
}

void printDeviceInfo(const cudaDeviceProp& prop) {
    int runtimeVersion = 0;
    int driverVersion = 0;
    int maxBlocksPerSm = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxBlocksPerSm,
                                      cudaDevAttrMaxBlocksPerMultiprocessor, 0));
    cudaFuncAttributes kernelAttributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&kernelAttributes, vectorAddKernel));

    std::cout << "gpu_name=" << prop.name << '\n'
              << "cuda_runtime_version=" << versionString(runtimeVersion) << '\n'
              << "cuda_driver_api_version=" << versionString(driverVersion) << '\n'
              << "compute_capability=" << prop.major << '.' << prop.minor << '\n'
              << "sm_count=" << prop.multiProcessorCount << '\n'
              << "warp_size=" << prop.warpSize << '\n'
              << "max_threads_per_block=" << prop.maxThreadsPerBlock << '\n'
              << "max_threads_dim=" << prop.maxThreadsDim[0] << ','
              << prop.maxThreadsDim[1] << ',' << prop.maxThreadsDim[2] << '\n'
              << "max_grid_size=" << prop.maxGridSize[0] << ','
              << prop.maxGridSize[1] << ',' << prop.maxGridSize[2] << '\n'
              << "max_threads_per_multiprocessor="
              << prop.maxThreadsPerMultiProcessor << '\n'
              << "max_warps_per_multiprocessor="
              << prop.maxThreadsPerMultiProcessor / prop.warpSize << '\n'
              << "max_blocks_per_multiprocessor=" << maxBlocksPerSm << '\n'
              << "shared_mem_per_block_bytes=" << prop.sharedMemPerBlock << '\n'
              << "shared_mem_per_multiprocessor_bytes="
              << prop.sharedMemPerMultiprocessor << '\n'
              << "regs_per_block=" << prop.regsPerBlock << '\n'
              << "regs_per_multiprocessor=" << prop.regsPerMultiprocessor << '\n'
              << "total_global_mem_bytes=" << prop.totalGlobalMem << '\n'
              << "l2_cache_size_bytes=" << prop.l2CacheSize << '\n'
              << "memory_bus_width_bits=" << prop.memoryBusWidth << '\n'
              << "memory_clock_rate_khz=" << prop.memoryClockRate << '\n'
              << "concurrent_kernels=" << prop.concurrentKernels << '\n'
              << "unified_addressing=" << prop.unifiedAddressing << '\n'
              << "kernel_registers_per_thread=" << kernelAttributes.numRegs << '\n'
              << "kernel_static_shared_mem_bytes="
              << kernelAttributes.sharedSizeBytes << '\n'
              << "kernel_max_threads_per_block="
              << kernelAttributes.maxThreadsPerBlock << '\n';
}

void writeCorrectnessHeader(std::ostream& out) {
    out << "timestamp,gpu_name,cuda_version,N,block_size,grid_size,"
           "total_threads,warps_per_block,runtime_success,max_abs_error,"
           "correctness,error_message\n";
}

void writeCorrectnessRow(std::ostream& out, const std::string& timestamp,
                         const std::string& gpuName,
                         const std::string& cudaVersion,
                         const CaseResult& result) {
    out << timestamp << ',' << csvEscape(gpuName) << ',' << cudaVersion << ','
        << result.n << ',' << result.blockSize << ',' << result.gridSize << ','
        << result.totalThreads << ',' << result.warpsPerBlock << ",true,"
        << std::scientific << std::setprecision(9) << result.maxAbsError << ','
        << (result.correct ? "PASS" : "FAIL") << ",\n";
}

void writeCorrectnessError(std::ostream& out, const std::string& timestamp,
                           const std::string& gpuName,
                           const std::string& cudaVersion, std::size_t n,
                           int blockSize, const std::string& error) {
    out << timestamp << ',' << csvEscape(gpuName) << ',' << cudaVersion << ','
        << n << ',' << blockSize
        << ",,,,,false,,FAIL," << csvEscape(error) << '\n';
}

void writeBenchmarkHeader(std::ostream& out) {
    out << "timestamp,gpu_name,cuda_version,N,block_size,grid_size,"
           "warps_per_block,warmup,repetitions,avg_kernel_latency_ms,"
           "effective_bandwidth_GBps,correctness,max_abs_error,total_threads,"
           "active_blocks_per_sm,active_warps_per_sm,"
           "theoretical_occupancy_pct,error_message\n";
}

void writeBenchmarkRow(std::ostream& out, const std::string& timestamp,
                       const std::string& gpuName,
                       const std::string& cudaVersion,
                       const CaseResult& result) {
    out << timestamp << ',' << csvEscape(gpuName) << ',' << cudaVersion << ','
        << result.n << ',' << result.blockSize << ',' << result.gridSize << ','
        << result.warpsPerBlock << ',' << result.warmup << ','
        << result.repetitions << ',' << std::fixed << std::setprecision(6)
        << result.avgKernelLatencyMs << ',' << std::setprecision(3)
        << result.effectiveBandwidthGBps << ','
        << (result.correct ? "PASS" : "FAIL") << ',' << std::scientific
        << std::setprecision(9) << result.maxAbsError << ',' << std::fixed
        << result.totalThreads << ',' << result.activeBlocksPerSm << ','
        << result.activeWarpsPerSm << ',' << std::setprecision(2)
        << result.theoreticalOccupancyPct << ",\n";
}

void writeBenchmarkError(std::ostream& out, const std::string& timestamp,
                         const std::string& gpuName,
                         const std::string& cudaVersion, std::size_t n,
                         int blockSize, int warmup, int repetitions,
                         const std::string& error) {
    out << timestamp << ',' << csvEscape(gpuName) << ',' << cudaVersion << ','
        << n << ',' << blockSize << ",,,"
        << warmup << ',' << repetitions
        << ",,,FAIL,,,,,," << csvEscape(error) << '\n';
}

struct StabilityObservation {
    int round = 0;
    int orderIndex = 0;
    CaseResult result;
};

void writeStabilityHeader(std::ostream& out) {
    out << "timestamp,round,order_index,N,block_size,grid_size,"
           "warps_per_block,active_blocks_per_sm,theoretical_occupancy_pct,"
           "warmup,repetitions,avg_kernel_latency_ms,"
           "effective_bandwidth_GBps,correctness,max_abs_error\n";
}

void writeStabilityRow(std::ostream& out, const std::string& timestamp,
                       const StabilityObservation& observation) {
    const CaseResult& result = observation.result;
    out << timestamp << ',' << observation.round << ','
        << observation.orderIndex << ',' << result.n << ','
        << result.blockSize << ',' << result.gridSize << ','
        << result.warpsPerBlock << ',' << result.activeBlocksPerSm << ','
        << std::fixed << std::setprecision(2)
        << result.theoreticalOccupancyPct << ',' << result.warmup << ','
        << result.repetitions << ',' << std::setprecision(6)
        << result.avgKernelLatencyMs << ',' << std::setprecision(3)
        << result.effectiveBandwidthGBps << ','
        << (result.correct ? "PASS" : "FAIL") << ',' << std::scientific
        << std::setprecision(9) << result.maxAbsError << '\n';
}

double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (double value : values) sum += value;
    return sum / static_cast<double>(values.size());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2.0;
}

double sampleStdDev(const std::vector<double>& values, double valuesMean) {
    if (values.size() < 2) return 0.0;
    double squaredDifferenceSum = 0.0;
    for (double value : values) {
        const double difference = value - valuesMean;
        squaredDifferenceSum += difference * difference;
    }
    return std::sqrt(squaredDifferenceSum /
                     static_cast<double>(values.size() - 1));
}

void writeStabilitySummary(
    std::ostream& out,
    const std::vector<StabilityObservation>& observations) {
    out << "block_size,count,mean_latency_ms,median_latency_ms,"
           "sample_std_latency_ms,min_latency_ms,max_latency_ms,cv_pct,"
           "mean_effective_bandwidth_GBps,"
           "median_effective_bandwidth_GBps\n";
    for (int block : kStabilityBlocks) {
        std::vector<double> latencies;
        std::vector<double> bandwidths;
        for (const StabilityObservation& observation : observations) {
            if (observation.result.blockSize == block) {
                latencies.push_back(observation.result.avgKernelLatencyMs);
                bandwidths.push_back(
                    observation.result.effectiveBandwidthGBps);
            }
        }
        if (latencies.empty()) {
            throw std::runtime_error("missing stability observations");
        }
        const double latencyMean = mean(latencies);
        const double latencyMedian = median(latencies);
        const double latencyStdDev = sampleStdDev(latencies, latencyMean);
        const auto latencyRange =
            std::minmax_element(latencies.begin(), latencies.end());
        const double cvPct =
            latencyMean > 0.0 ? 100.0 * latencyStdDev / latencyMean : 0.0;
        out << block << ',' << latencies.size() << ',' << std::fixed
            << std::setprecision(6) << latencyMean << ',' << latencyMedian
            << ',' << latencyStdDev << ',' << *latencyRange.first << ','
            << *latencyRange.second << ',' << std::setprecision(3) << cvPct
            << ',' << mean(bandwidths) << ',' << median(bandwidths) << '\n';
    }
}

struct Options {
    std::string mode;
    std::string csvPath;
    std::string summaryPath;
    std::size_t n = 0;
    int blockSize = 0;
    int warmup = 20;
    int repetitions = 200;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const char* name) -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(std::string("missing value for ") +
                                            name);
            }
            return argv[i];
        };
        if (arg == "--device-info") {
            options.mode = "device-info";
        } else if (arg == "--correctness") {
            options.mode = "correctness";
        } else if (arg == "--benchmark") {
            options.mode = "benchmark";
        } else if (arg == "--single") {
            options.mode = "single";
        } else if (arg == "--stability") {
            options.mode = "stability";
        } else if (arg == "--test-csv-escape") {
            options.mode = "test-csv-escape";
        } else if (arg == "--csv") {
            options.csvPath = requireValue("--csv");
        } else if (arg == "--summary") {
            options.summaryPath = requireValue("--summary");
        } else if (arg == "--n") {
            options.n = std::stoull(requireValue("--n"));
        } else if (arg == "--block") {
            options.blockSize = std::stoi(requireValue("--block"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(requireValue("--warmup"));
        } else if (arg == "--repetitions") {
            options.repetitions = std::stoi(requireValue("--repetitions"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.mode.empty()) {
        throw std::invalid_argument(
            "select --device-info, --correctness, --benchmark, --single, "
            "--stability, or --test-csv-escape");
    }
    if ((options.mode == "correctness" || options.mode == "benchmark") &&
        options.csvPath.empty()) {
        throw std::invalid_argument("--csv is required for sweep modes");
    }
    if (options.mode == "single" &&
        (options.n == 0 || options.blockSize == 0)) {
        throw std::invalid_argument("--single requires --n and --block");
    }
    if (options.mode == "stability" &&
        (options.csvPath.empty() || options.summaryPath.empty())) {
        throw std::invalid_argument(
            "--stability requires --csv and --summary");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.mode == "test-csv-escape") {
            const std::string plainInput = "abc";
            const std::string quotedInput = "a\"b";
            const std::string quotedExpected = "\"a\"\"b\"";
            const std::string plainOutput = csvEscape(plainInput);
            const std::string quotedOutput = csvEscape(quotedInput);
            const bool passed = plainOutput == plainInput &&
                                quotedOutput == quotedExpected;
            std::cout << "plain_input=" << plainInput << '\n'
                      << "plain_output=" << plainOutput << '\n'
                      << "quoted_input=" << quotedInput << '\n'
                      << "quoted_output=" << quotedOutput << '\n'
                      << "csv_escape_test=" << (passed ? "PASS" : "FAIL")
                      << '\n';
            return passed ? 0 : 2;
        }
        int deviceId = 0;
        CUDA_CHECK(cudaGetDevice(&deviceId));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
        int runtimeVersion = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
        const std::string cudaVersion = versionString(runtimeVersion);
        const std::string timestamp = utcTimestamp();
        const std::vector<int> blocks = supportedBlocks(prop);

        if (options.mode == "device-info") {
            printDeviceInfo(prop);
            std::cout << "supported_sweep_blocks=";
            for (std::size_t i = 0; i < blocks.size(); ++i) {
                if (i != 0) std::cout << ',';
                std::cout << blocks[i];
            }
            std::cout << '\n';
            return 0;
        }

        if (options.mode == "single") {
            const CaseResult result =
                runCase(options.n, options.blockSize, options.warmup,
                        options.repetitions, prop);
            std::cout << "N=" << result.n << '\n'
                      << "block_size=" << result.blockSize << '\n'
                      << "grid_size=" << result.gridSize << '\n'
                      << "total_threads=" << result.totalThreads << '\n'
                      << "warps_per_block=" << result.warpsPerBlock << '\n'
                      << "extra_logical_threads="
                      << result.totalThreads - result.n << '\n'
                      << "active_blocks_per_sm=" << result.activeBlocksPerSm
                      << '\n'
                      << "active_warps_per_sm=" << result.activeWarpsPerSm
                      << '\n'
                      << "theoretical_occupancy_pct="
                      << result.theoreticalOccupancyPct << '\n'
                      << "avg_kernel_latency_ms=" << result.avgKernelLatencyMs
                      << '\n'
                      << "effective_bandwidth_GBps="
                      << result.effectiveBandwidthGBps << '\n'
                      << "max_abs_error=" << result.maxAbsError << '\n'
                      << "correctness=" << (result.correct ? "PASS" : "FAIL")
                      << '\n';
            return result.correct ? 0 : 2;
        }

        if (options.mode == "stability") {
            std::ofstream rawCsv(options.csvPath);
            if (!rawCsv) {
                throw std::runtime_error("cannot open CSV path: " +
                                         options.csvPath);
            }
            std::ofstream summaryCsv(options.summaryPath);
            if (!summaryCsv) {
                throw std::runtime_error("cannot open summary path: " +
                                         options.summaryPath);
            }
            writeStabilityHeader(rawCsv);
            std::vector<StabilityObservation> observations;
            bool allPassed = true;
            std::cout << "round,order_index,N,block_size,grid_size,"
                         "warps_per_block,active_blocks_per_sm,"
                         "theoretical_occupancy_pct,warmup,repetitions,"
                         "avg_kernel_latency_ms,effective_bandwidth_GBps,"
                         "correctness,max_abs_error\n";
            for (int round = 0; round < 5; ++round) {
                for (int orderIndex = 0; orderIndex < 6; ++orderIndex) {
                    const int block = kStabilityOrders[round][orderIndex];
                    StabilityObservation observation;
                    observation.round = round + 1;
                    observation.orderIndex = orderIndex + 1;
                    observation.result =
                        runCase(kStabilityN, block, options.warmup,
                                options.repetitions, prop);
                    writeStabilityRow(rawCsv, utcTimestamp(), observation);
                    const CaseResult& result = observation.result;
                    std::cout << observation.round << ','
                              << observation.orderIndex << ',' << result.n << ','
                              << result.blockSize << ',' << result.gridSize << ','
                              << result.warpsPerBlock << ','
                              << result.activeBlocksPerSm << ',' << std::fixed
                              << std::setprecision(2)
                              << result.theoreticalOccupancyPct << ','
                              << result.warmup << ',' << result.repetitions << ','
                              << std::setprecision(6)
                              << result.avgKernelLatencyMs << ','
                              << std::setprecision(3)
                              << result.effectiveBandwidthGBps << ','
                              << (result.correct ? "PASS" : "FAIL") << ','
                              << std::scientific << std::setprecision(9)
                              << result.maxAbsError << '\n';
                    allPassed = allPassed && result.correct;
                    observations.push_back(observation);
                }
            }
            writeStabilitySummary(summaryCsv, observations);
            rawCsv.close();
            summaryCsv.close();
            return allPassed ? 0 : 2;
        }

        std::ofstream csv(options.csvPath);
        if (!csv) {
            throw std::runtime_error("cannot open CSV path: " + options.csvPath);
        }
        bool allPassed = true;

        if (options.mode == "correctness") {
            writeCorrectnessHeader(csv);
            std::cout << "N,block_size,grid_size,total_threads,warps_per_block,"
                         "max_abs_error,correctness\n";
            for (std::size_t n : kCorrectnessSizes) {
                for (int block : blocks) {
                    try {
                        const CaseResult result = runCase(n, block, 0, 1, prop);
                        writeCorrectnessRow(csv, timestamp, prop.name, cudaVersion,
                                            result);
                        std::cout << result.n << ',' << result.blockSize << ','
                                  << result.gridSize << ',' << result.totalThreads
                                  << ',' << result.warpsPerBlock << ','
                                  << std::scientific << result.maxAbsError << ','
                                  << (result.correct ? "PASS" : "FAIL") << '\n';
                        allPassed = allPassed && result.correct;
                    } catch (const std::exception& error) {
                        writeCorrectnessError(csv, timestamp, prop.name,
                                              cudaVersion, n, block, error.what());
                        std::cerr << "N=" << n << " block=" << block
                                  << " runtime_error=" << error.what() << '\n';
                        allPassed = false;
                    }
                }
            }
        } else {
            writeBenchmarkHeader(csv);
            std::cout << "N,block_size,grid_size,warps_per_block,warmup,"
                         "repetitions,avg_kernel_latency_ms,"
                         "effective_bandwidth_GBps,correctness,max_abs_error,"
                         "active_blocks_per_sm,active_warps_per_sm,"
                         "theoretical_occupancy_pct\n";
            for (std::size_t n : kBenchmarkSizes) {
                for (int block : blocks) {
                    try {
                        const CaseResult result =
                            runCase(n, block, options.warmup,
                                    options.repetitions, prop);
                        writeBenchmarkRow(csv, timestamp, prop.name, cudaVersion,
                                          result);
                        std::cout << result.n << ',' << result.blockSize << ','
                                  << result.gridSize << ',' << result.warpsPerBlock
                                  << ',' << result.warmup << ','
                                  << result.repetitions << ',' << std::fixed
                                  << std::setprecision(6)
                                  << result.avgKernelLatencyMs << ','
                                  << std::setprecision(3)
                                  << result.effectiveBandwidthGBps << ','
                                  << (result.correct ? "PASS" : "FAIL") << ','
                                  << std::scientific << result.maxAbsError << ','
                                  << std::fixed << result.activeBlocksPerSm << ','
                                  << result.activeWarpsPerSm << ','
                                  << std::setprecision(2)
                                  << result.theoreticalOccupancyPct << '\n';
                        allPassed = allPassed && result.correct;
                    } catch (const std::exception& error) {
                        writeBenchmarkError(csv, timestamp, prop.name,
                                            cudaVersion, n, block,
                                            options.warmup, options.repetitions,
                                            error.what());
                        std::cerr << "N=" << n << " block=" << block
                                  << " runtime_error=" << error.what() << '\n';
                        allPassed = false;
                    }
                }
            }
        }
        csv.close();
        return allPassed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
