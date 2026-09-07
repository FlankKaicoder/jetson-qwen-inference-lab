#include "rmsnorm_kernel.cuh"

#include <cstddef>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (unsigned int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T>
__global__ void rmsnorm_v1_kernel(const T* __restrict__ x,
                                  T* __restrict__ y,
                                  const T* __restrict__ weight,
                                  int hidden, float eps) {
    extern __shared__ float shared[];
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;

    float sum_squares = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float value = static_cast<float>(x[token * hidden + i]);
        sum_squares += value * value;
    }
    sum_squares = warp_reduce_sum(sum_squares);
    if (lane == 0) {
        shared[warp_id] = sum_squares;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            const float mean = block_sum / static_cast<float>(hidden);
            shared[0] = 1.0f / sqrtf(mean + eps);
        }
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
void launch_rmsnorm_v1(const T* x, T* y, const T* weight, int tokens,
                       int hidden, float eps, cudaStream_t stream) {
    const int threads = hidden;
    const int warps = threads / 32;
    const size_t shared_bytes = static_cast<size_t>(warps) * sizeof(float);
    rmsnorm_v1_kernel<T><<<tokens, threads, shared_bytes, stream>>>(
        x, y, weight, hidden, eps);
}

template void launch_rmsnorm_v1<__half>(const __half*, __half*, const __half*,
                                        int, int, float, cudaStream_t);
template void launch_rmsnorm_v1<__nv_bfloat16>(const __nv_bfloat16*,
                                                __nv_bfloat16*,
                                                const __nv_bfloat16*, int, int,
                                                float, cudaStream_t);
