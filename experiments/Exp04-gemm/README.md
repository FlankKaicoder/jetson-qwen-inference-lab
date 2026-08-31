# Exp04 - CUDA GEMM: Naive FP32 Foundation

## Scope

This first stage establishes a CPU FP32 reference (V0) and the simplest one-thread-per-output CUDA FP32 kernel (V1). It intentionally stops before shared-memory tiling, WMMA/Tensor Cores, vectorization, register tiling, double buffering, or `cp.async`.

Status: `RUNNING` (initial Gate A1 only)

## Reproduction

On Jetson, use the scripts in `scripts/`:

```bash
bash experiments/Exp04-gemm/scripts/build.sh
bash experiments/Exp04-gemm/scripts/run_correctness.sh
bash experiments/Exp04-gemm/scripts/run_benchmark.sh
```

The binary is kept outside Git under `/tmp/jetson-qwen-exp04-build`. Timestamped correctness, environment, calibration, raw trial, and summary files are written under `benchmark/` and are never overwritten.

## Versions

- V0: CPU row-major FP32 reference, `C[m,n] = sum_k A[m,k] * B[k,n]`. Correctness-only.
- V1: naive CUDA FP32, one thread computes one `C[row,col]`, with bounds checks for arbitrary dimensions.

V2 Shared Memory Tiling, V3 WMMA, double buffering and `cp.async` are explicitly not started in this stage.

## Current gates

- Gate A1: CPU/V1 correctness across tiny, boundary-like, rectangular, non-power-of-two and large cases, with FP32 error tolerances and output canaries.
- Gate B/C and the final Exp04 Gate A/B/C remain `NOT_STARTED` until the corresponding evidence exists.
