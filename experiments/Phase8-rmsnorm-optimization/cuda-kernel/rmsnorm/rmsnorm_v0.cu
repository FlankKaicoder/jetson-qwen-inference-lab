#include "rmsnorm_kernel.cuh"

#include <cstddef>

namespace {

__device__ __forceinline__ float square_sum(float value) {
    return value * value;
}

template <typename T>
__global__ void rmsnorm_v0_kernel(const T* __restrict__ x,
                                  T* __restrict__ y,
                                  const T* __restrict__ weight,
                                  int hidden, float eps) {
    extern __shared__ float shared[];
    const int token = blockIdx.x;
    const int tid = threadIdx.x;

    float sum_squares = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float value = static_cast<float>(x[token * hidden + i]);
        sum_squares = square_sum(value);
    }
    shared[tid] = sum_squares;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float mean = shared[0] / static_cast<float>(hidden);
        shared[0] = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();

    const float inv_rms = shared[0];
    for (int i = tid; i < hidden; i += blockDim.x) {
        const int index = token * hidden + i;
        y[index] = static_cast<T>(static_cast<float>(x[index]) * inv_rms *
                                  static_cast<float>(weight[i]));
    }
}

}  // namespace

template <typename T>
void launch_rmsnorm_v0(const T* x, T* y, const T* weight, int tokens,
                       int hidden, float eps, cudaStream_t stream) {
    const int threads = hidden;
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
    rmsnorm_v0_kernel<T><<<tokens, threads, shared_bytes, stream>>>(
        x, y, weight, hidden, eps);
}

template void launch_rmsnorm_v0<__half>(const __half*, __half*, const __half*,
                                        int, int, float, cudaStream_t);
template void launch_rmsnorm_v0<__nv_bfloat16>(const __nv_bfloat16*,
                                                __nv_bfloat16*,
                                                const __nv_bfloat16*, int, int,
                                                float, cudaStream_t);
