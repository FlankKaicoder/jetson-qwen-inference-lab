#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kM = 1;
constexpr int kK = 1024;
constexpr int kN = 3072;
constexpr int kWarmupCalls = 200;
constexpr int kMinimumMeasurements = 500;
constexpr double kWarmupWindowMs = 1000.0;
constexpr int kTrials = 7;
constexpr double kTargetWindowMs = 500.0;
constexpr size_t kWorkspaceBytes = 32ull << 20;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = (call);                                              \
    if (status_ != cudaSuccess) {                                              \
      std::cerr << "CUDA_ERROR " << cudaGetErrorString(status_) << " at "      \
                << __LINE__ << std::endl;                                      \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status_ = (call);                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::cerr << "CUBLAS_ERROR " << static_cast<int>(status_) << " at "      \
                << __LINE__ << std::endl;                                      \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)

struct TimingStats {
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double sample_std_ms = 0.0;
  double cv = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double p95_ms = 0.0;
  double tflops = 0.0;
  double iterations_per_trial = 0.0;
};

struct Correctness {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double max_abs_reference = 0.0;
  double gate_limit = 0.0;
  bool pass = false;
};

struct Checksums {
  double sum = 0.0;
  uint64_t fnv1a = 1469598103934665603ull;
};

struct AlgoRecord {
  std::string compute_name;
  int heuristic_index = -1;
  int64_t algorithm_id = -1;
  size_t workspace_size = 0;
  float waves_count = -1.0f;
  int state = -1;
  size_t window_warmup_calls = 0;
  TimingStats timing;
  Correctness correctness;
};

std::vector<__half> MakeMatrix(int rows, int cols, uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<float> distribution(-0.125f, 0.125f);
  std::vector<__half> matrix(static_cast<size_t>(rows) * cols);
  for (auto &value : matrix) {
    value = __float2half(distribution(generator));
  }
  return matrix;
}

Checksums HostChecksum(const std::vector<__half> &values) {
  Checksums result;
  for (const __half &value : values) {
    const float host_value = __half2float(value);
    result.sum += host_value;
    result.fnv1a ^= static_cast<uint64_t>(
        *reinterpret_cast<const uint16_t *>(&value));
    result.fnv1a *= 1099511628211ull;
  }
  return result;
}

Correctness CompareAgainstReference(const std::vector<__half> &a,
                                    const std::vector<__half> &b,
                                    const std::vector<__half> &c) {
  Correctness result;
  for (int n = 0; n < kN; ++n) {
    double reference = 0.0;
    for (int k = 0; k < kK; ++k) {
      reference += static_cast<double>(__half2float(a[k])) *
                   static_cast<double>(__half2float(b[k * kN + n]));
    }
    const double actual = static_cast<double>(__half2float(c[n]));
    const double error = std::abs(actual - reference);
    const double rel_error = error / std::max(std::abs(reference), 1e-12);
    result.max_abs_error = std::max(result.max_abs_error, error);
    result.max_rel_error = std::max(result.max_rel_error, rel_error);
    result.max_abs_reference = std::max(result.max_abs_reference,
                                        std::abs(reference));
  }
  result.gate_limit = 1e-3 + 1e-4 * result.max_abs_reference;
  result.pass = result.max_abs_error <= result.gate_limit;
  return result;
}

TimingStats Summarize(const std::vector<double> &latencies,
                      const std::vector<double> &iterations) {
  TimingStats stats;
  std::vector<double> sorted = latencies;
  std::sort(sorted.begin(), sorted.end());
  const size_t count = sorted.size();
  double sum = 0.0;
  for (double value : sorted) sum += value;
  stats.mean_ms = sum / count;
  stats.median_ms = count % 2 == 1
      ? sorted[count / 2]
      : 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);
  double variance = 0.0;
  for (double value : sorted) variance += (value - stats.mean_ms) *
                                          (value - stats.mean_ms);
  stats.sample_std_ms = count > 1 ? std::sqrt(variance / (count - 1)) : 0.0;
  stats.cv = stats.mean_ms > 0.0 ? stats.sample_std_ms / stats.mean_ms : 0.0;
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  const size_t p95_index = static_cast<size_t>(
      std::ceil(0.95 * count)) - 1;
  stats.p95_ms = sorted[p95_index];
  stats.iterations_per_trial = iterations.front();
  const double latency_seconds = stats.mean_ms / 1000.0;
  const double flops = 2.0 * kM * kK * kN;
  stats.tflops = latency_seconds > 0.0 ? flops / latency_seconds / 1e12 : 0.0;
  return stats;
}

void PrintStats(const TimingStats &stats) {
  std::cout << ",\"mean_ms\":" << std::fixed << std::setprecision(9)
            << stats.mean_ms;
  std::cout << ",\"median_ms\":" << stats.median_ms;
  std::cout << ",\"sample_std_ms\":" << stats.sample_std_ms;
  std::cout << ",\"cv\":" << stats.cv;
  std::cout << ",\"min_ms\":" << stats.min_ms;
  std::cout << ",\"max_ms\":" << stats.max_ms;
  std::cout << ",\"p95_ms\":" << stats.p95_ms;
  std::cout << ",\"tflops\":" << stats.tflops;
  std::cout << ",\"iterations_per_trial\":" << stats.iterations_per_trial;
}

void PrintCorrectness(const Correctness &correctness) {
  std::cout << ",\"max_abs_error\":" << std::fixed << std::setprecision(9)
            << correctness.max_abs_error;
  std::cout << ",\"max_rel_error\":" << correctness.max_rel_error;
  std::cout << ",\"max_abs_reference\":" << correctness.max_abs_reference;
  std::cout << ",\"gate_limit\":" << correctness.gate_limit;
  std::cout << ",\"correctness\":\"" << (correctness.pass ? "PASS" : "FAIL")
            << "\"";
}

void BenchmarkOne(cublasLtHandle_t handle,
                  cublasLtMatmulDesc_t desc,
                  cublasLtMatrixLayout_t layout_a,
                  cublasLtMatrixLayout_t layout_b,
                  cublasLtMatrixLayout_t layout_c,
                  const void *a_device,
                  const void *b_device,
                  void *c_device,
                  void *workspace,
                  const void *alpha,
                  const void *beta,
                  const cublasLtMatmulHeuristicResult_t &heuristic,
                  const std::vector<__half> &a_host,
                  const std::vector<__half> &b_host,
                  AlgoRecord &record) {
  record.workspace_size = heuristic.workspaceSize;
  record.waves_count = heuristic.wavesCount;
  record.state = static_cast<int>(heuristic.state);
  int32_t algorithm_id = -1;
  size_t algorithm_id_bytes = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &heuristic.algo, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id,
      sizeof(algorithm_id), &algorithm_id_bytes));
  record.algorithm_id = algorithm_id;

  for (int i = 0; i < kWarmupCalls; ++i) {
    CUBLAS_CHECK(cublasLtMatmul(handle, desc, alpha, a_device, layout_a,
                                b_device, layout_b, beta, c_device, layout_c,
                                c_device, layout_c, &heuristic.algo, workspace,
                                heuristic.workspaceSize, 0));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t warmup_start;
  cudaEvent_t warmup_stop;
  CUDA_CHECK(cudaEventCreate(&warmup_start));
  CUDA_CHECK(cudaEventCreate(&warmup_stop));
  size_t window_warmup_calls = 0;
  CUDA_CHECK(cudaEventRecord(warmup_start, 0));
  while (true) {
    for (int i = 0; i < 100; ++i) {
      CUBLAS_CHECK(cublasLtMatmul(handle, desc, alpha, a_device, layout_a,
                                  b_device, layout_b, beta, c_device,
                                  layout_c, c_device, layout_c,
                                  &heuristic.algo, workspace,
                                  heuristic.workspaceSize, 0));
    }
    window_warmup_calls += 100;
    CUDA_CHECK(cudaEventRecord(warmup_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(warmup_stop));
    float warmup_elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&warmup_elapsed_ms, warmup_start,
                                    warmup_stop));
    if (warmup_elapsed_ms >= kWarmupWindowMs) break;
  }
  CUDA_CHECK(cudaEventDestroy(warmup_start));
  CUDA_CHECK(cudaEventDestroy(warmup_stop));
  record.window_warmup_calls = window_warmup_calls;

  std::vector<double> trial_means;
  std::vector<double> iterations_per_trial;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int trial = 0; trial < kTrials; ++trial) {
    int iterations = kMinimumMeasurements;
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; ++i) {
      CUBLAS_CHECK(cublasLtMatmul(handle, desc, alpha, a_device, layout_a,
                                  b_device, layout_b, beta, c_device, layout_c,
                                  c_device, layout_c, &heuristic.algo,
                                  workspace, heuristic.workspaceSize, 0));
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double per_call_ms = elapsed_ms / iterations;
    if (elapsed_ms < kTargetWindowMs) {
      iterations = static_cast<int>(std::ceil(
          kTargetWindowMs / std::max(per_call_ms, 1e-9)));
      iterations = std::max(iterations, kMinimumMeasurements);
    }

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; ++i) {
      CUBLAS_CHECK(cublasLtMatmul(handle, desc, alpha, a_device, layout_a,
                                  b_device, layout_b, beta, c_device, layout_c,
                                  c_device, layout_c, &heuristic.algo,
                                  workspace, heuristic.workspaceSize, 0));
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    trial_means.push_back(elapsed_ms / iterations);
    iterations_per_trial.push_back(iterations);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  record.timing = Summarize(trial_means, iterations_per_trial);

  std::vector<__half> c_host(kN);
  CUDA_CHECK(cudaMemcpy(c_host.data(), c_device, kN * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  record.correctness = CompareAgainstReference(a_host, b_host, c_host);
}

}  // namespace

int main() {
  cublasLtHandle_t handle;
  CUBLAS_CHECK(cublasLtCreate(&handle));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver_version = 0;
  int runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  size_t cublaslt_version = 0;
  cublaslt_version = cublasLtGetVersion();

  auto a_host = MakeMatrix(kM, kK, 0x5a17a0001ull);
  auto b_host = MakeMatrix(kK, kN, 0x5a17a0002ull);
  auto c_host = std::vector<__half>(kN, __float2half(0.0f));
  const Checksums a_checksum = HostChecksum(a_host);
  const Checksums b_checksum = HostChecksum(b_host);

  __half *a_device = nullptr;
  __half *b_device = nullptr;
  __half *c_device = nullptr;
  CUDA_CHECK(cudaMalloc(&a_device, a_host.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&b_device, b_host.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&c_device, c_host.size() * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(a_device, a_host.data(), a_host.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_device, b_host.data(), b_host.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c_device, c_host.data(), c_host.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));

  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16F, kM, kK, kK));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16F, kK, kN, kN));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16F, kM, kN, kN));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

  std::vector<AlgoRecord> records;
  struct Variant {
    const char *name;
    cublasComputeType_t compute_type;
    cudaDataType_t scale_type;
    float fp32_alpha;
    float fp32_beta;
    __half half_alpha;
    __half half_beta;
  };
  const std::vector<Variant> variants = {
      {"FP32_ACCUMULATE", CUBLAS_COMPUTE_32F, CUDA_R_32F, 1.0f, 0.0f,
       __float2half(0.0f), __float2half(0.0f)},
      {"FP16_ACCUMULATE", CUBLAS_COMPUTE_16F, CUDA_R_16F, 0.0f, 0.0f,
       __float2half(1.0f), __float2half(0.0f)}};

  for (const auto &variant : variants) {
    cublasLtMatmulDesc_t desc = nullptr;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, variant.compute_type,
                                          variant.scale_type));
    cublasOperation_t trans = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    cublasLtMatmulPreference_t preference = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspace_bytes = kWorkspaceBytes;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));

    constexpr int kRequested = 8;
    cublasLtMatmulHeuristicResult_t heuristics[kRequested] = {};
    int returned = 0;
    cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
        handle, desc, layout_a, layout_b, layout_c, layout_c, preference,
        kRequested, heuristics, &returned);
    if (heuristic_status != CUBLAS_STATUS_SUCCESS || returned == 0) {
      std::cout << "{\"status\":\"NO_HEURISTIC\",\"compute\":\""
                << variant.name << "\",\"returned\":" << returned
                << ",\"cublas_status\":" << static_cast<int>(heuristic_status)
                << "}" << std::endl;
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatmulDescDestroy(desc);
      continue;
    }

    void *workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, kWorkspaceBytes));
    const void *alpha = variant.compute_type == CUBLAS_COMPUTE_32F
        ? static_cast<const void *>(&variant.fp32_alpha)
        : static_cast<const void *>(&variant.half_alpha);
    const void *beta = variant.compute_type == CUBLAS_COMPUTE_32F
        ? static_cast<const void *>(&variant.fp32_beta)
        : static_cast<const void *>(&variant.half_beta);

    for (int i = 0; i < returned; ++i) {
      if (heuristics[i].state != CUBLAS_STATUS_SUCCESS ||
          heuristics[i].workspaceSize > kWorkspaceBytes) {
        continue;
      }
      AlgoRecord record;
      record.compute_name = variant.name;
      record.heuristic_index = i;
      BenchmarkOne(handle, desc, layout_a, layout_b, layout_c, a_device,
                   b_device, c_device, workspace, alpha, beta, heuristics[i],
                   a_host, b_host, record);
      records.push_back(record);
    }

    CUDA_CHECK(cudaFree(workspace));
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(desc);
  }

  const AlgoRecord *best = nullptr;
  for (const auto &record : records) {
    if (record.compute_name != "FP32_ACCUMULATE") continue;
    if (best == nullptr || record.timing.mean_ms < best->timing.mean_ms) {
      best = &record;
    }
  }

  std::cout << "{";
  std::cout << "\"phase\":\"Phase5A\",";
  std::cout << "\"backend\":\"cuBLASLt\",";
  std::cout << "\"api\":\"cublasLtMatmul\",";
  std::cout << "\"device_name\":\"" << prop.name << "\",";
  std::cout << "\"compute_capability\":" << prop.major << prop.minor << ",";
  std::cout << "\"cuda_driver_version\":" << driver_version << ",";
  std::cout << "\"cuda_runtime_version\":" << runtime_version << ",";
  std::cout << "\"cublaslt_version\":" << cublaslt_version << ",";
  std::cout << "\"shape\":{\"M\":" << kM << ",\"K\":" << kK << ",\"N\":"
            << kN << "},";
  std::cout << "\"layout\":\"row-major\",";
  std::cout << "\"warmup_calls\":" << kWarmupCalls << ",";
  std::cout << "\"warmup_window_ms\":" << kWarmupWindowMs << ",";
  std::cout << "\"input_distribution\":\"uniform_[-0.125,0.125]\",";
  std::cout << "\"a_checksum_sum\":" << std::setprecision(12)
            << a_checksum.sum << ",";
  std::cout << "\"a_checksum_fnv1a\":" << a_checksum.fnv1a << ",";
  std::cout << "\"b_checksum_sum\":" << b_checksum.sum << ",";
  std::cout << "\"b_checksum_fnv1a\":" << b_checksum.fnv1a << ",";
  std::cout << "\"trials\":" << kTrials << ",";
  std::cout << "\"target_window_ms\":" << kTargetWindowMs << ",";
  std::cout << "\"records\":[";
  for (size_t i = 0; i < records.size(); ++i) {
    const auto &record = records[i];
    if (i != 0) std::cout << ",";
    std::cout << "{\"compute\":\"" << record.compute_name << "\",";
    std::cout << "\"heuristic_index\":" << record.heuristic_index << ",";
    std::cout << "\"algorithm_id\":" << record.algorithm_id << ",";
    std::cout << "\"workspace_size\":" << record.workspace_size << ",";
    std::cout << "\"waves_count\":" << record.waves_count << ",";
    std::cout << "\"state\":" << record.state;
    std::cout << ",\"window_warmup_calls\":" << record.window_warmup_calls;
    PrintStats(record.timing);
    PrintCorrectness(record.correctness);
    std::cout << "}";
  }
  std::cout << "],";
  if (best != nullptr) {
    std::cout << "\"primary_selected_compute\":\"FP32_ACCUMULATE\",";
    std::cout << "\"primary_selected_algorithm_id\":" << best->algorithm_id
              << ",";
    std::cout << "\"primary_selected_heuristic_index\":"
              << best->heuristic_index << ",";
    std::cout << "\"primary_selected_mean_ms\":" << best->timing.mean_ms << ",";
    std::cout << "\"primary_selected_tflops\":" << best->timing.tflops;
  } else {
    std::cout << "\"primary_selected_compute\":\"UNKNOWN\"";
  }
  std::cout << "}" << std::endl;

  cublasLtMatrixLayoutDestroy(layout_a);
  cublasLtMatrixLayoutDestroy(layout_b);
  cublasLtMatrixLayoutDestroy(layout_c);
  CUDA_CHECK(cudaFree(a_device));
  CUDA_CHECK(cudaFree(b_device));
  CUDA_CHECK(cudaFree(c_device));
  CUBLAS_CHECK(cublasLtDestroy(handle));
  return 0;
}
