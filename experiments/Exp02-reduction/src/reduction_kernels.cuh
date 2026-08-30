#pragma once

#include <cuda_runtime.h>

#include <cstddef>

enum class ReductionVersion : int {
    V1GlobalMultiPass = 1,
    V2SharedModulo = 2,
    V3SharedIndexed = 3,
    V4SharedSequential = 4,
    V5FirstAddLoad = 5,
    V6SharedWarpTail = 6,
    V7Shuffle = 7,
};

std::size_t reductionOutputCount(ReductionVersion version, std::size_t count,
                                 int blockSize);
cudaError_t launchReduction(ReductionVersion version, const float* input,
                            float* output, std::size_t count, int blockSize,
                            cudaStream_t stream = nullptr);

