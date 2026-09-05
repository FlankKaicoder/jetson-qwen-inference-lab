# Phase 4-D Baseline Comparison

## Standalone reference

Primary exact-layout reference is `torch.matmul` with contiguous
`A[1,1024]` and `B[1024,3072]` in Half:

| Metric | Value |
| --- | ---: |
| Median | `0.086364701 ms` |
| Mean | `0.086365059 ms` |
| Sample std | `0.000039214 ms` |
| CV | `0.045405%` |
| Min | `0.086306362 ms` |
| Max | `0.086426028 ms` |
| Trials | 7 |

Secondary `torch.nn.functional.linear` with `W[3072,1024]` gives median
`0.083020223 ms`, CV `0.035288%`. Its logical computation is equivalent, but its
physical weight layout differs from the target `B[K,N]`, so it is not the
primary exact-layout reference.

## TensorRT baseline

No pure TensorRT `up_proj` per-call baseline exists. The closest committed
context is Phase 3-E h16816 aggregate:

| Item | Value |
| --- | ---: |
| Mixed steady-state kernel time | `39.590976 ms` |
| Share of Mixed steady-state kernel time | `26.781320%` |
| Operator match | `gate_proj;up_proj` |
| NCU mean duration | `7601.557 us` |
| NCU memory/L2 | `97.06%` |
| NCU HMMA active | `39.673148%` |
| NCU achieved occupancy | `24.78%` |

These values must not be interpreted as pure `up_proj` latency.

## Difference

The standalone PyTorch reference and TensorRT path differ in implementation,
operand provenance, launch context, memory layout, and possible fusion. Their
latencies cannot be directly equated. The standalone baseline establishes only
a trusted library comparison point for the exact logical GEMM.
