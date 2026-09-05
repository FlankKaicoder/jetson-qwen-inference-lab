# Phase 4-D Standalone GEMM Benchmark Protocol

## Scope

This protocol defines a standalone reference measurement for the exact logical
GEMM shape only. It is not a TensorRT runtime baseline and does not authorize a
custom kernel.

## Reference implementation

- Primary: `torch.matmul(A, B)` with `A=[1,1024]`, `B=[1024,3072]`, both Half and contiguous.
- Secondary: `torch.nn.functional.linear(A, W)` with `W=[3072,1024]` contiguous. This is logically `A * W.T`, but its physical weight layout differs from `B[K,N]`.

PyTorch dispatches to a trusted GPU library path. The exact backend kernel
identity is `UNKNOWN` and must not be guessed.

## Measurement

- Use deterministic synthetic Half operands unless real weights are separately authorized.
- Allocate inputs and outputs before timing.
- Use CUDA Events around repeated library calls only.
- Exclude allocation, H2D/D2H, input generation, finite checks, reference computation, and synchronization from the timed region.
- Calibrate, warm up toward approximately 1000 ms, then measure an adaptive window targeting approximately 500 ms per trial.
- Retain at least seven trials.
- Report mean, median, sample standard deviation, CV, min, and max.
- Record shape, dtype, contiguity, implementation, PyTorch/CUDA versions, device, capability, and memory counters.

## Interpretation limits

The standalone baseline is not the TensorRT `up_proj` baseline. It does not
include TensorRT launch context, memory formatting, fusion, or runtime overhead.
It also does not prove that a custom CUDA kernel will be faster.
