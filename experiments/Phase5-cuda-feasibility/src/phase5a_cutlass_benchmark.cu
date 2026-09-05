#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kM = 1;
constexpr int kK = 1024;
constexpr int kN = 3072;
constexpr int kWarmupCalls = 200;
constexpr double kWarmupWindowMs = 1000.0;
constexpr int kMinimumMeasurements = 500;
constexpr int kTrials = 7;
constexpr double kTargetWindowMs = 500.0;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = (call);                                              \
    if (status_ != cudaSuccess) {                                              \
      std::cerr << "CUDA_ERROR " << cudaGetErrorString(status_) << " at "      \
                << __LINE__ << std::endl;                                      \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

#define CUTLASS_CHECK(call)                                                    \
  do {                                                                         \
    cutlass::Status status_ = (call);                                          \
    if (status_ != cutlass::Status::kSuccess) {                                \
      std::cerr << "CUTLASS_ERROR code=" << cutlassGetStatusString(status_)    \
                << " at " << __LINE__ << std::endl;                            \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)

struct Checksums {
  double sum = 0.0;
  uint64_t fnv1a = 1469598103934665603ull;
};

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

struct Candidate {
  int id;
  std::string template_name;
  int threadblock_m;
  int threadblock_n;
  int threadblock_k;
  int warp_m;
  int warp_n;
  int warp_k;
  int stages;
  int split_k;
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
    const double relative = error / std::max(std::abs(reference), 1e-12);
    result.max_abs_error = std::max(result.max_abs_error, error);
    result.max_rel_error = std::max(result.max_rel_error, relative);
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
  for (double value : sorted) {
    variance += (value - stats.mean_ms) * (value - stats.mean_ms);
  }
  stats.sample_std_ms = count > 1 ? std::sqrt(variance / (count - 1)) : 0.0;
  stats.cv = stats.mean_ms > 0.0 ? stats.sample_std_ms / stats.mean_ms : 0.0;
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  const size_t p95_index = static_cast<size_t>(std::ceil(0.95 * count)) - 1;
  stats.p95_ms = sorted[p95_index];
  stats.iterations_per_trial = iterations.front();
  const double flops = 2.0 * kM * kK * kN;
  stats.tflops = flops / (stats.mean_ms / 1000.0) / 1e12;
  return stats;
}

void PrintStats(const TimingStats &stats) {
  std::cout << std::fixed << std::setprecision(9);
  std::cout << ",\"mean_ms\":" << stats.mean_ms;
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
  std::cout << std::fixed << std::setprecision(9);
  std::cout << ",\"max_abs_error\":" << correctness.max_abs_error;
  std::cout << ",\"max_rel_error\":" << correctness.max_rel_error;
  std::cout << ",\"max_abs_reference\":" << correctness.max_abs_reference;
  std::cout << ",\"gate_limit\":" << correctness.gate_limit;
  std::cout << ",\"correctness\":\"" << (correctness.pass ? "PASS" : "FAIL")
            << "\"";
}

template <typename GemmType>
void RunCandidate(const Candidate &candidate,
                  const void *a_device,
                  const void *b_device,
                  void *c_device,
                  const std::vector<__half> &a_host,
                  const std::vector<__half> &b_host,
                  TimingStats &timing,
                  Correctness &correctness,
                  size_t &window_warmup_calls) {
  GemmType gemm_op;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  typename GemmType::Arguments arguments(
      {kM, kN, kK},
      {reinterpret_cast<const cutlass::half_t *>(a_device), kK},
      {reinterpret_cast<const cutlass::half_t *>(b_device), kN},
      {reinterpret_cast<const cutlass::half_t *>(c_device), kN},
      {reinterpret_cast<cutlass::half_t *>(c_device), kN},
      {alpha, beta},
      candidate.split_k);
  const size_t workspace_size = GemmType::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace));
  CUTLASS_CHECK(gemm_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < kWarmupCalls; ++i) {
    CUTLASS_CHECK(gemm_op.run());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t warmup_start;
  cudaEvent_t warmup_stop;
  CUDA_CHECK(cudaEventCreate(&warmup_start));
  CUDA_CHECK(cudaEventCreate(&warmup_stop));
  window_warmup_calls = 0;
  CUDA_CHECK(cudaEventRecord(warmup_start, 0));
  while (true) {
    for (int i = 0; i < 100; ++i) {
      CUTLASS_CHECK(gemm_op.run());
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
      CUTLASS_CHECK(gemm_op.run());
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
      CUTLASS_CHECK(gemm_op.run());
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    trial_means.push_back(elapsed_ms / iterations);
    iterations_per_trial.push_back(iterations);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  if (workspace != nullptr) {
    CUDA_CHECK(cudaFree(workspace));
  }
  timing = Summarize(trial_means, iterations_per_trial);

  std::vector<__half> c_host(kN);
  CUDA_CHECK(cudaMemcpy(c_host.data(), c_device, kN * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  correctness = CompareAgainstReference(a_host, b_host, c_host);
}

}  // namespace

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver_version = 0;
  int runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

  auto a_host = MakeMatrix(kM, kK, 0x5a17a0001ull);
  auto b_host = MakeMatrix(kK, kN, 0x5a17a0002ull);
  const Checksums a_checksum = HostChecksum(a_host);
  const Checksums b_checksum = HostChecksum(b_host);
  void *a_device = nullptr;
  void *b_device = nullptr;
  void *c_device = nullptr;
  CUDA_CHECK(cudaMalloc(&a_device, a_host.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&b_device, b_host.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&c_device, kN * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(a_device, a_host.data(), a_host.size() *
                        sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_device, b_host.data(), b_host.size() *
                        sizeof(__half), cudaMemcpyHostToDevice));

  const std::vector<Candidate> candidates = {
      {1, "TensorOp_16x8x16_tb32x32x64_warp32x32x64_s4", 32, 32, 64, 32, 32,
       64, 4, 1},
      {2, "TensorOp_16x8x16_tb32x32x64_warp32x32x64_s4", 32, 32, 64, 32, 32,
       64, 4, 2},
      {3, "TensorOp_16x8x16_tb32x32x64_warp32x32x64_s4", 32, 32, 64, 32, 32,
       64, 4, 4},
      {4, "TensorOp_16x8x16_tb32x32x64_warp32x32x64_s4", 32, 32, 64, 32, 32,
       64, 4, 8},
      {5, "TensorOp_16x8x16_tb64x64x64_warp32x32x64_s4", 64, 64, 64, 32, 32,
       64, 4, 1},
      {6, "TensorOp_16x8x16_tb64x64x64_warp32x32x64_s4", 64, 64, 64, 32, 32,
       64, 4, 2},
      {7, "TensorOp_16x8x16_tb64x64x64_warp32x32x64_s4", 64, 64, 64, 32, 32,
       64, 4, 4},
      {8, "TensorOp_16x8x16_tb64x64x64_warp32x32x64_s4", 64, 64, 64, 32, 32,
       64, 4, 8},
      {9, "TensorOp_16x8x16_tb128x64x64_warp64x32x64_s5", 128, 64, 64, 64, 32,
       64, 5, 1},
      {10, "TensorOp_16x8x16_tb128x64x64_warp64x32x64_s5", 128, 64, 64, 64,
       32, 64, 5, 4}};

  std::vector<TimingStats> timings(candidates.size());
  std::vector<Correctness> correctnesses(candidates.size());
  std::vector<size_t> warmups(candidates.size());
  std::vector<bool> completed(candidates.size(), false);

  for (size_t index = 0; index < candidates.size(); ++index) {
    const Candidate &candidate = candidates[index];
    try {
      if (candidate.threadblock_m == 32) {
        using GemmType = cutlass::gemm::device::Gemm<
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<32, 32, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<
                cutlass::half_t, 1, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            4, 8, 8, true>;
        RunCandidate<GemmType>(candidate, a_device, b_device, c_device,
                               a_host, b_host, timings[index],
                               correctnesses[index], warmups[index]);
      } else if (candidate.threadblock_m == 64) {
        using GemmType = cutlass::gemm::device::Gemm<
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<
                cutlass::half_t, 1, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            4, 8, 8, true>;
        RunCandidate<GemmType>(candidate, a_device, b_device, c_device,
                               a_host, b_host, timings[index],
                               correctnesses[index], warmups[index]);
      } else {
        using GemmType = cutlass::gemm::device::Gemm<
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 64, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<
                cutlass::half_t, 1, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            5, 8, 8, true>;
        RunCandidate<GemmType>(candidate, a_device, b_device, c_device,
                               a_host, b_host, timings[index],
                               correctnesses[index], warmups[index]);
      }
      completed[index] = true;
    } catch (const std::exception &error) {
      std::cerr << "CANDIDATE_RUNTIME_ERROR id=" << candidate.id << " "
                << error.what() << std::endl;
    }
  }

  std::cout << "{";
  std::cout << "\"phase\":\"Phase5A\",";
  std::cout << "\"backend\":\"CUTLASS\",";
  std::cout << "\"api\":\"cutlass::gemm::device::Gemm\",";
  std::cout << "\"cutlass_version\":\"v3.5.1\",";
  std::cout << "\"cutlass_commit\":\"f7b19de32c5d1f3cedfc735c2849f12b537522ee\",";
  std::cout << "\"architecture\":\"Sm80_kernel_on_SM87\",";
  std::cout << "\"device_name\":\"" << prop.name << "\",";
  std::cout << "\"compute_capability\":" << prop.major << prop.minor << ",";
  std::cout << "\"cuda_driver_version\":" << driver_version << ",";
  std::cout << "\"cuda_runtime_version\":" << runtime_version << ",";
  std::cout << "\"shape\":{\"M\":" << kM << ",\"K\":" << kK << ",\"N\":"
            << kN << "},";
  std::cout << "\"layout\":\"row-major\",";
  std::cout << "\"dtype\":\"FP16_input_FP16_output\",";
  std::cout << "\"accumulate\":\"FP32\",";
  std::cout << "\"alpha\":1.0,\"beta\":0.0,";
  std::cout << "\"warmup_calls\":" << kWarmupCalls << ",";
  std::cout << "\"warmup_window_ms\":" << kWarmupWindowMs << ",";
  std::cout << "\"trials\":" << kTrials << ",";
  std::cout << "\"target_window_ms\":" << kTargetWindowMs << ",";
  std::cout << "\"input_distribution\":\"uniform_[-0.125,0.125]\",";
  std::cout << "\"a_checksum_sum\":" << a_checksum.sum << ",";
  std::cout << "\"a_checksum_fnv1a\":" << a_checksum.fnv1a << ",";
  std::cout << "\"b_checksum_sum\":" << b_checksum.sum << ",";
  std::cout << "\"b_checksum_fnv1a\":" << b_checksum.fnv1a << ",";
  std::cout << "\"records\":[";
  bool first = true;
  const TimingStats *best = nullptr;
  int best_id = -1;
  for (size_t index = 0; index < candidates.size(); ++index) {
    if (!completed[index]) continue;
    if (!first) std::cout << ",";
    first = false;
    const Candidate &candidate = candidates[index];
    std::cout << "{\"candidate_id\":" << candidate.id;
    std::cout << ",\"configuration\":\"" << candidate.template_name << "\"";
    std::cout << ",\"threadblock\":\"" << candidate.threadblock_m << "x"
              << candidate.threadblock_n << "x" << candidate.threadblock_k
              << "\"";
    std::cout << ",\"warp\":\"" << candidate.warp_m << "x" << candidate.warp_n
              << "x" << candidate.warp_k << "\"";
    std::cout << ",\"stages\":" << candidate.stages;
    std::cout << ",\"split_k_slices\":" << candidate.split_k;
    std::cout << ",\"window_warmup_calls\":" << warmups[index];
    PrintStats(timings[index]);
    PrintCorrectness(correctnesses[index]);
    std::cout << "}";
    if (correctnesses[index].pass &&
        (best == nullptr || timings[index].mean_ms < best->mean_ms)) {
      best = &timings[index];
      best_id = candidate.id;
    }
  }
  std::cout << "],";
  if (best != nullptr) {
    std::cout << "\"primary_selected_candidate_id\":" << best_id << ",";
    std::cout << "\"primary_selected_mean_ms\":" << best->mean_ms << ",";
    std::cout << "\"primary_selected_median_ms\":" << best->median_ms << ",";
    std::cout << "\"primary_selected_tflops\":" << best->tflops;
  } else {
    std::cout << "\"primary_selected_candidate_id\":\"UNKNOWN\"";
  }
  std::cout << "}" << std::endl;

  CUDA_CHECK(cudaFree(a_device));
  CUDA_CHECK(cudaFree(b_device));
  CUDA_CHECK(cudaFree(c_device));
  return 0;
}
