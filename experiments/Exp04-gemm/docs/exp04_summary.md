# Exp04 GEMM Summary

## Gates and hypotheses

- Gate A1: `PASS`, inherited V1 13/13.
- Gate A2: `PASS`, V2-T8/T16/T32 each 13/13 on the same matrix set (39/39).
- Gate A3: `PASS`, WMMA aligned set 8/8; canaries intact and CUDA checks passed.
- Gate B: `PASS` for the final retained cross-version set after the post-warmup `256^3` V1 correction. All retained trials have CV <= 0.50% and actual windows >= 500 ms.
- Gate C: `PASS` for the defined NCU/SASS evidence set.

H1 (V2 tiling improves FP32 performance): `SUPPORTED` by V2-T16 versus V1 latency/GFLOPS.
H2 (performance is not determined by occupancy alone): `SUPPORTED`; T8/T16/T32 use different block resources and T16 wins rather than the largest tile being fastest.
H3 (tiling trades barriers/shared/register cost for reuse): `SUPPORTED` by code structure, timing, shared counters and register data.
H4 (WMMA executes Tensor Core MMA): `SUPPORTED` by non-zero tensor-pipe activity and `HMMA.16816.F32` SASS.
H5 (V3 speedup accompanies a changed execution structure but is not Tensor-Core-only attribution): `SUPPORTED`; V3 changes input precision to FP16 and uses FP32 accumulation, so the speedup is a mixed-precision WMMA implementation result.
H6 (GEMM performance needs movement, reuse, compute, scheduler and occupancy together): `SUPPORTED` as the analysis framework; no single counter is treated as a complete causal explanation.

## Closure scope

V2 Shared Memory Tiling and V3 Tensor Core WMMA are complete for this V2.0 scope. Double buffering is not required; `cp.async` and advanced swizzling are not started. Exp05 is not started.
