# Exp04 Reduction? No: GEMM Experiment Design

## Question

How does a row-major FP32 GEMM map threads, warps, blocks and grids on Jetson Orin Nano Super, and how do later shared-memory and Tensor Core versions change that mapping? This stage answers only the baseline questions: can the mathematical contract be implemented correctly, what addresses does V1 generate, and what stable kernel-only latency/GFLOPS does the naive implementation achieve?

## Mathematical contract

For `A[M x K]`, `B[K x N]`, and `C[M x N]` in row-major storage:

```text
C[m,n] = sum(k=0..K-1) A[m*K+k] * B[k*N+n]
```

V0 computes this on CPU in FP32 and is used as the numerical reference. V1 uses the same operation and accumulation type.

## V1 mapping and launch

The initial launch uses `block=(16,16,1)` and
`grid=(ceil(N/16), ceil(M/16), 1)`. A thread owns exactly one output element:

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
float acc = 0.0f;
for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];
C[row * N + col] = acc;
```

The bounds guard makes partial blocks safe. The choice is an explanatory baseline, not a performance conclusion.

### Concrete warp address analysis

For warp lanes `threadIdx.y=0`, `threadIdx.x=0..31` in a 16x16 block, lanes 0-15 are the first row and lanes 16-31 are the second row (the warp is linearized in x fastest, then y). For lanes 0-15 at fixed `k`, `A[row*K+k]` is the same address within that row: source-level repeated loads provide a possible warp broadcast/cache opportunity. Lanes 16-31 repeat the same pattern for the next row. `B[k*N+col]` is contiguous for each 16-lane half-warp and the next half-warp starts at the next row's columns; it is not a single 32-lane contiguous span with this geometry. `C[row*N+col]` stores are contiguous within each 16-lane half-warp, with a row-stride jump between halves. This is theoretical source addressing only. L1/L2 cache behavior, compiler load generation, memory sectors and DRAM traffic require Nsight Compute evidence and are not inferred here.

## Optimization ladder (planned, not implemented)

V2 shared-memory tiled FP32 GEMM; V3 coalesced/sequential tile traversal and reduced redundant loads; V4 register accumulation refinements if needed; V5 WMMA/Tensor Core comparison in an explicitly separate precision contract. Double buffering/`cp.async` is deferred until a later design review.

## Correctness Gate A1

The harness runs V0 and V1 for tiny (`1x1x1`, `2x3x4`), dimensions around 15/16/17 and 31/32/33, rectangular/non-power-of-two cases, and a large `512^3` case. Every case uses deterministic FP32 inputs, guarded output allocation, CUDA error checks, and reports max absolute error, max relative error, RMSE, tolerance, and PASS/FAIL. Bitwise equality is not required. Compute Sanitizer availability is recorded separately.

Tolerance is `abs_error <= atol + rtol * max_abs_reference`, with `atol=1e-3` and `rtol=1e-4`; the scale uses the largest reference magnitude, and RMSE is reported for diagnosis.

## Benchmark Gate (initial V1 only)

Each shape is calibrated, warmed for about 1000 ms by elapsed CUDA work, then measured in an adaptive 500 ms target window with a 2x safety factor. Seven trials are retained. CUDA Events surround only repeated V1 launches; allocation, initialization, H2D, D2H and CPU reference are excluded. FLOPs are `2*M*N*K`, and GFLOPS is computed from mean kernel latency. All raw trials and calibration values are preserved.

Initial shapes are `256^3`, `512^3`, `1024^3`, plus one rectangular `512x384x640` case. `2048^3` is deferred unless a later authorized run demonstrates acceptable duration.

## Nsight Gate (later)

The environment record captures CUDA and Nsight Compute versions. A later Gate C will select only sections/metrics confirmed by `ncu --list-sections` on Jetson 2024.3.1. Candidate observations include achieved occupancy, SM/memory throughput, eligible warps, scheduler/warp stalls, branch efficiency, and supported global load/store transaction sectors. No V1 compute-bound or memory-bound conclusion is made from this stage alone.

## Hypotheses

- H1: V1 is correct for arbitrary partial dimensions when the output bounds guard is active.
- H2: One-thread-per-output V1 performs redundant source-level A loads across lanes sharing a row.
- H3: The 16x16 mapping gives contiguous B loads and C stores only within each 16-lane half-warp; later tile layouts can improve warp-wide coalescing and data reuse.
- H4: Adaptive timing with a sufficiently long window produces lower run-to-run CV than a short fixed launch count on this platform.
- H5: Shared-memory tiling can reduce redundant global loads, but its synchronization/resource cost means larger blocks are not assumed faster.

H1 is testable now; H2-H5 remain hypotheses pending code changes and profiler evidence.
