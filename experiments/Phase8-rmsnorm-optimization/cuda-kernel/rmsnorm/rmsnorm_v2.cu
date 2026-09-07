#include "rmsnorm_kernel.cuh"

#include <cstddef>
#include <cuda_bf16.h>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (unsigned int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T>
__global__ void rmsnorm_v2_kernel(const T* __restrict__ x,
                                  T* __restrict__ y,
                                  const T* __restrict__ weight,
                                  int hidden, float eps);

template <>
__global__ void rmsnorm_v2_kernel<__half>(const __half* __restrict__ x,
                                          __half* __restrict__ y,
                                          const __half* __restrict__ weight,
                                          int hidden, float eps) {
    extern __shared__ float shared[];
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int vector_index = token * (hidden / 2) + tid;

    const __half2 x2 = *reinterpret_cast<const __half2*>(x + token * hidden + 2 * tid);
    const float2 value = __half22float2(x2);
    float sum_squares = value.x * value.x + value.y * value.y;

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
    const __half2 weight2 = *reinterpret_cast<const __half2*>(
        weight + 2 * tid);
    const float2 weight_value = __half22float2(weight2);
    const __half2 result = __floats2half2_rn(
        value.x * inv_rms * weight_value.x,
        value.y * inv_rms * weight_value.y);
    *reinterpret_cast<__half2*>(y + vector_index * 2) = result;
}

template <>
__global__ void rmsnorm_v2_kernel<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ weight, int hidden, float eps) {
    extern __shared__ float shared[];
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int vector_index = token * (hidden / 2) + tid;

    const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162*>(
        x + token * hidden + 2 * tid);
    const float2 value = __bfloat1622float2(x2);
    float sum_squares = value.x * value.x + value.y * value.y;

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
    const __nv_bfloat162 weight2 = *reinterpret_cast<const __nv_bfloat162*>(
        weight + 2 * tid);
    const float2 weight_value = __bfloat1622float2(weight2);
    const __nv_bfloat162 result = __floats2bfloat162_rn(
        value.x * inv_rms * weight_value.x,
        value.y * inv_rms * weight_value.y);
    *reinterpret_cast<__nv_bfloat162*>(y + vector_index * 2) = result;
}

}  // namespace

template <typename T>
void launch_rmsnorm_v2(const T* x, T* y, const T* weight, int tokens,
                       int hidden, float eps, cudaStream_t stream) {
    const int threads = hidden / 2;
    const int warps = threads / 32;
    const size_t shared_bytes = static_cast<size_t>(warps) * sizeof(float);
    rmsnorm_v2_kernel<T><<<tokens, threads, shared_bytes, stream>>>(
        x, y, weight, hidden, eps);
}

template void launch_rmsnorm_v2<__half>(const __half*, __half*, const __half*,
                                        int, int, float, cudaStream_t);
template void launch_rmsnorm_v2<__nv_bfloat16>(const __nv_bfloat16*,
                                                __nv_bfloat16*,
                                                const __nv_bfloat16*, int, int,
                                                float, cudaStream_t);
