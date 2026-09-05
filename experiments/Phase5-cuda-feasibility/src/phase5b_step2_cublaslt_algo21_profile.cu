#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kM = 1;
constexpr int kK = 1024;
constexpr int kN = 3072;
constexpr int kWarmupCalls = 100;
constexpr int kHeuristicIndex = 4;
constexpr int kExpectedAlgorithmId = 21;
constexpr size_t kMaxWorkspaceBytes = 32ull << 20;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t status_ = (call);                                        \
    if (status_ != cudaSuccess) {                                              \
      std::cerr << "CUDA_ERROR " << cudaGetErrorString(status_) << " at "      \
                << __LINE__ << std::endl;                                      \
      return 2;                                                                \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    const cublasStatus_t status_ = (call);                                     \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::cerr << "CUBLAS_ERROR " << static_cast<int>(status_) << " at "      \
                << __LINE__ << std::endl;                                      \
      return 3;                                                                \
    }                                                                          \
  } while (0)

std::vector<__half> MakeMatrix(int rows, int cols, uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<float> distribution(-0.125f, 0.125f);
  std::vector<__half> matrix(static_cast<size_t>(rows) * cols);
  for (auto &value : matrix) {
    value = __float2half(distribution(generator));
  }
  return matrix;
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
  const size_t cublaslt_version = cublasLtGetVersion();

  const auto a_host = MakeMatrix(kM, kK, 0x5a17a0001ull);
  const auto b_host = MakeMatrix(kK, kN, 0x5a17a0002ull);
  const auto c_host = std::vector<__half>(kN, __float2half(0.0f));

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

  cublasLtMatmulDesc_t desc = nullptr;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F,
                                        CUDA_R_32F));
  const cublasOperation_t transpose = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      desc, CUBLASLT_MATMUL_DESC_TRANSB, &transpose, sizeof(transpose)));

  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16F, kM, kK, kK));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16F, kK, kN, kN));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16F, kM, kN, kN));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout_c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  size_t workspace_bytes = kMaxWorkspaceBytes;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
      sizeof(workspace_bytes)));

  constexpr int kRequested = 8;
  cublasLtMatmulHeuristicResult_t heuristics[kRequested] = {};
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      handle, desc, layout_a, layout_b, layout_c, layout_c, preference,
      kRequested, heuristics, &returned));
  if (returned <= kHeuristicIndex ||
      heuristics[kHeuristicIndex].state != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "HEURISTIC_INDEX_UNAVAILABLE" << std::endl;
    return 4;
  }

  const auto &heuristic = heuristics[kHeuristicIndex];
  int32_t algorithm_id = -1;
  size_t algorithm_id_bytes = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &heuristic.algo, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id,
      sizeof(algorithm_id), &algorithm_id_bytes));
  if (algorithm_id != kExpectedAlgorithmId) {
    std::cerr << "UNEXPECTED_ALGORITHM_ID " << algorithm_id << std::endl;
    return 5;
  }

  void *workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&workspace, kMaxWorkspaceBytes));
  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i = 0; i < kWarmupCalls; ++i) {
    CUBLAS_CHECK(cublasLtMatmul(handle, desc, &alpha, a_device, layout_a,
                                b_device, layout_b, &beta, c_device, layout_c,
                                c_device, layout_c, &heuristic.algo, workspace,
                                heuristic.workspaceSize, 0));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUBLAS_CHECK(cublasLtMatmul(handle, desc, &alpha, a_device, layout_a,
                              b_device, layout_b, &beta, c_device, layout_c,
                              c_device, layout_c, &heuristic.algo, workspace,
                              heuristic.workspaceSize, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<__half> result(kN);
  CUDA_CHECK(cudaMemcpy(result.data(), c_device, result.size() * sizeof(__half),
                        cudaMemcpyDeviceToHost));
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double max_abs_reference = 0.0;
  for (int n = 0; n < kN; ++n) {
    double reference = 0.0;
    for (int k = 0; k < kK; ++k) {
      reference += static_cast<double>(__half2float(a_host[k])) *
                   static_cast<double>(__half2float(b_host[k * kN + n]));
    }
    const double actual = static_cast<double>(__half2float(result[n]));
    const double error = std::abs(actual - reference);
    max_abs_error = std::max(max_abs_error, error);
    max_rel_error = std::max(max_rel_error,
                             error / std::max(std::abs(reference), 1e-12));
    max_abs_reference = std::max(max_abs_reference, std::abs(reference));
  }
  const double gate_limit = 1e-3 + 1e-4 * max_abs_reference;
  const bool pass = max_abs_error <= gate_limit;

  std::cout << std::fixed << std::setprecision(9) << "{"
            << "\"backend\":\"cuBLASLt\",\"api\":\"cublasLtMatmul\","
            << "\"device_name\":\"" << prop.name << "\","
            << "\"compute_capability\":" << prop.major << prop.minor << ","
            << "\"cuda_driver_version\":" << driver_version << ","
            << "\"cuda_runtime_version\":" << runtime_version << ","
            << "\"cublaslt_version\":" << cublaslt_version << ","
            << "\"shape\":{\"M\":" << kM << ",\"K\":" << kK << ",\"N\":"
            << kN << "},\"layout\":\"row-major\","
            << "\"compute\":\"FP32_ACCUMULATE\",\"scale_type\":\"FP32\","
            << "\"input_dtype\":\"FP16\",\"output_dtype\":\"FP16\","
            << "\"heuristic_index\":" << kHeuristicIndex << ","
            << "\"algorithm_id\":" << algorithm_id << ","
            << "\"workspace_size\":" << heuristic.workspaceSize << ","
            << "\"waves_count\":" << heuristic.wavesCount << ","
            << "\"warmup_calls\":" << kWarmupCalls << ","
            << "\"profiled_launches\":1,"
            << "\"max_abs_error\":" << max_abs_error << ","
            << "\"max_rel_error\":" << max_rel_error << ","
            << "\"max_abs_reference\":" << max_abs_reference << ","
            << "\"gate_limit\":" << gate_limit << ","
            << "\"correctness\":\"" << (pass ? "PASS" : "FAIL") << "\"}"
            << std::endl;

  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(a_device));
  CUDA_CHECK(cudaFree(b_device));
  CUDA_CHECK(cudaFree(c_device));
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(layout_a);
  cublasLtMatrixLayoutDestroy(layout_b);
  cublasLtMatrixLayoutDestroy(layout_c);
  cublasLtMatmulDescDestroy(desc);
  CUBLAS_CHECK(cublasLtDestroy(handle));
  return pass ? 0 : 6;
}
