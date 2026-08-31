# Exp03 - CUDA Matrix Transpose

## Status

Exp03.0 initialization is complete and the Exp03.1 correctness harness is ready to run. Gate A is `PENDING`; Gate B (Benchmark/Stability) and Gate C (Nsight/Microarchitecture) are `NOT_STARTED`; Overall is `IN_PROGRESS`. Formal benchmark, Nsight Compute, performance ranking and Exp04 were not started.

## Learning objective

This experiment separates coalesced global memory access from shared-memory bank conflicts. The fixed layout is row-major FP32 input `height x width`, where `input[y * width + x]` becomes row-major output `width x height` at `output[x * height + y]`.

## Fixed geometry

`TILE_DIM=32`, `BLOCK_ROWS=8`, and `dim3 block(32, 8)` (256 threads, 8 warps). Tiled kernels process 32 x 32 tiles and each thread handles four y positions.

## Versions

- V0: CPU reference using the exact transpose mapping.
- V1: GPU copy control, `output[y * width + x] = input[y * width + x]`; it is not a transpose competitor.
- V2: naive transpose with coalesced input reads and strided row-major output stores.
- V3: shared-memory tiled transpose with `tile[32][32]`; global reads and writes are coalesced, while the transposed shared read is intentionally unpadded.
- V4: identical to V3 except `tile[32][33]`; the extra column changes shared-memory bank mapping.

V3 and V4 use the same mapping, guards, synchronization and block geometry. Their mechanism difference is padding only.

## Correctness Gate A

The harness covers 23 dimensions, three deterministic FP32 patterns (coordinate-coded, sequential, fixed-seed signed), four GPU versions and three independent executions per case: 828 executions. V1 is compared bitwise with input; V2-V4 are compared bitwise with the CPU V0 reference. It covers tiny, single-row/column, square, rectangular, exact-tile and partial-tile matrices, including 4093/4096 boundaries.

Each case records dimensions and 2D launch metadata, checks all CUDA allocation/copy/launch/last-error/synchronization calls, and validates prefix/suffix output canaries. The canary is an auxiliary OOB check, not an equivalent to Compute Sanitizer; Compute Sanitizer is recorded as `N/A` and was not installed or run.

## Gates

- Gate A: `PASS` after all 828 executions passed bitwise correctness, guards and CUDA checks; source mechanism audit passed.
- Gate B: `NOT_STARTED`; no formal benchmark or stability run was performed.
- Gate C: `NOT_STARTED`; no Nsight Compute run was performed.
- Overall: `IN_PROGRESS`; next authorized state is `READY_FOR_EXP03_BENCHMARK`.

## Reproduction

On Jetson:

```bash
experiments/Exp03-matrix-transpose/scripts/build.sh
experiments/Exp03-matrix-transpose/scripts/run_correctness.sh
```

The binary stays under `/tmp/jetson-qwen-exp03-build`; timestamped CSV and environment artifacts are kept under `benchmark/raw/`.
