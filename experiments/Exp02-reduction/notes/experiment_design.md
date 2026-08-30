# Exp02 Reduction Experiment Design

## Scope

Exp02 studies `global memory -> thread-local load -> shared memory -> block reduction -> __syncthreads() -> warp reduction -> warp shuffle -> final scalar`. Exp02.0 initializes code and evidence layout. Exp02.1 closes only Correctness Gate A. Benchmark and Nsight work are later phases.

## Versions

- V1: global-memory pairwise multi-pass baseline. Every pass launches a kernel and writes global intermediate values.
- V2: shared-memory interleaved modulo addressing. It retains modulo selection and divergence while every thread reaches every barrier.
- V3: indexed interleaved shared reduction. It removes modulo selection and exposes an indexed active-thread pattern.
- V4: sequential-addressing shared reduction. Active threads are contiguous as stride halves.
- V5: first-add-during-load. Each thread combines two inputs with tail guards before shared reduction.
- V6: shared-memory reduction with an explicitly synchronized warp tail using `__syncwarp()`.
- V7: warp reduction with `__shfl_down_sync` and an `__activemask()`-derived mask. Shared memory only aggregates warp sums.

Supported block sizes are exactly 32, 64, 128, 256, 512 and 1024. Other values are explicitly rejected.

## Correctness protocol

Inputs are deterministic all-ones, fixed-seed signed random (`seed=0x02C0FFEE`) and alternating-sign cancellation. For each block B, N is `1, B-1, B, B+1, 2B-1, 2B, 2B+1, 17B+13, 2^20+13`. Each case/version is executed three independent times. CPU reference uses double accumulation.

Before data collection, the floating-point gate is frozen as:

```text
epsilon = 2^-23
k = ceil(log2(max(N, 2))) + 2
gamma_k = (k * epsilon) / (1 - k * epsilon)
absolute_error <= 8 * gamma_k * sum_abs + 1e-6
normalized_error = absolute_error / max(sum_abs, 1)
```

All-ones cases additionally require the exact integer result after conversion to FP32. CUDA allocation, copy, launch, last-error and synchronization calls are checked.

## Gates

- Gate A: V1-V7, all patterns, boundaries and repeats pass; no runtime, OOB, race or barrier error. Compute Sanitizer support and representative results are recorded without fabricating unsupported PASS values.
- Gate B: `NOT_STARTED`; no benchmark, timing or stability run is permitted in Exp02.1.
- Gate C: `NOT_STARTED`; no Nsight Compute profiling is permitted in Exp02.1.
- Overall: `IN_PROGRESS` even when Gate A passes.

No performance ranking is made by Exp02.1.

