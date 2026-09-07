#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
void launch_rmsnorm_v0(const T* x, T* y, const T* weight, int tokens,
                       int hidden, float eps, cudaStream_t stream);

template <typename T>
void launch_rmsnorm_v1(const T* x, T* y, const T* weight, int tokens,
                       int hidden, float eps, cudaStream_t stream);

template <typename T>
void launch_rmsnorm_v2(const T* x, T* y, const T* weight, int tokens,
                       int hidden, float eps, cudaStream_t stream);
