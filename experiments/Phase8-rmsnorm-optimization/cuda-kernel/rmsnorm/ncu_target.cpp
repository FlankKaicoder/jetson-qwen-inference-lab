#include "rmsnorm_kernel.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

constexpr int kHidden = 1024;
constexpr float kEpsilon = 1e-6f;
constexpr unsigned long long kSeed = 20260907ULL;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        const cudaError_t cuda_status = (call);                             \
        if (cuda_status != cudaSuccess) {                                   \
            throw std::runtime_error(std::string("CUDA error at ") + #call + \
                                     ": " + cudaGetErrorString(cuda_status)); \
        }                                                                   \
    } while (false)

}  // namespace

int main(int argc, char** argv) {
    std::string version;
    std::string dtype;
    int tokens = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&]() {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing argument value");
            }
            return std::string(argv[++i]);
        };
        if (argument == "--version") {
            version = value();
        } else if (argument == "--dtype") {
            dtype = value();
        } else if (argument == "--tokens") {
            tokens = std::stoi(value());
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if ((version != "V0" && version != "V1" && version != "V2") ||
        (dtype != "float16" && dtype != "bfloat16") ||
        (tokens != 1 && tokens != 8)) {
        throw std::runtime_error(
            "expected --version V0|V1|V2 --dtype float16|bfloat16 "
            "--tokens 1|8");
    }

    const size_t element_count = static_cast<size_t>(tokens) * kHidden;
    std::vector<__half> half_input(element_count);
    std::vector<__half> half_weight(kHidden);
    std::vector<__nv_bfloat16> bf16_input(element_count);
    std::vector<__nv_bfloat16> bf16_weight(kHidden);
    std::mt19937_64 generator(kSeed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < element_count; ++i) {
        const float value = distribution(generator);
        half_input[i] = static_cast<__half>(value);
        bf16_input[i] = static_cast<__nv_bfloat16>(value);
    }
    for (int i = 0; i < kHidden; ++i) {
        const float value = distribution(generator);
        half_weight[i] = static_cast<__half>(value);
        bf16_weight[i] = static_cast<__nv_bfloat16>(value);
    }

    void* device_input = nullptr;
    void* device_output = nullptr;
    void* device_weight = nullptr;
    const size_t input_bytes = element_count *
        (dtype == "float16" ? sizeof(__half) : sizeof(__nv_bfloat16));
    const size_t weight_bytes = static_cast<size_t>(kHidden) *
        (dtype == "float16" ? sizeof(__half) : sizeof(__nv_bfloat16));
    CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&device_output, input_bytes));
    CUDA_CHECK(cudaMalloc(&device_weight, weight_bytes));
    if (dtype == "float16") {
        CUDA_CHECK(cudaMemcpy(device_input, half_input.data(), input_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_weight, half_weight.data(), weight_bytes,
                              cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(device_input, bf16_input.data(), input_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_weight, bf16_weight.data(), weight_bytes,
                              cudaMemcpyHostToDevice));
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (dtype == "float16") {
        const __half* input = static_cast<const __half*>(device_input);
        __half* output = static_cast<__half*>(device_output);
        const __half* weight = static_cast<const __half*>(device_weight);
        if (version == "V0") {
            launch_rmsnorm_v0<__half>(input, output, weight, tokens, kHidden,
                                      kEpsilon, stream);
        } else if (version == "V1") {
            launch_rmsnorm_v1<__half>(input, output, weight, tokens, kHidden,
                                      kEpsilon, stream);
        } else {
            launch_rmsnorm_v2<__half>(input, output, weight, tokens, kHidden,
                                      kEpsilon, stream);
        }
    } else {
        const __nv_bfloat16* input =
            static_cast<const __nv_bfloat16*>(device_input);
        __nv_bfloat16* output = static_cast<__nv_bfloat16*>(device_output);
        const __nv_bfloat16* weight =
            static_cast<const __nv_bfloat16*>(device_weight);
        if (version == "V0") {
            launch_rmsnorm_v0<__nv_bfloat16>(input, output, weight, tokens,
                                             kHidden, kEpsilon, stream);
        } else if (version == "V1") {
            launch_rmsnorm_v1<__nv_bfloat16>(input, output, weight, tokens,
                                             kHidden, kEpsilon, stream);
        } else {
            launch_rmsnorm_v2<__nv_bfloat16>(input, output, weight, tokens,
                                             kHidden, kEpsilon, stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_weight));

    std::cout << "version=" << version << " dtype=" << dtype
              << " tokens=" << tokens << " hidden=" << kHidden
              << " epsilon=" << kEpsilon << "\n";
    return 0;
}
