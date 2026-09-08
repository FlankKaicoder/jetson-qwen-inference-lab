#include "rmsnorm_plugin.h"

#include "../cuda-kernel/rmsnorm/rmsnorm_kernel.cuh"

#include <cuda_fp16.h>

namespace phase8 {

cudaError_t launchRMSNormPluginKernel(const void* input, void* output,
                                      const void* gamma, int tokens,
                                      float epsilon, cudaStream_t stream) {
    launch_rmsnorm_v2<__half>(static_cast<const __half*>(input),
                              static_cast<__half*>(output),
                              static_cast<const __half*>(gamma), tokens,
                              kHiddenSize, epsilon, stream);
    return cudaPeekAtLastError();
}

}  // namespace phase8
