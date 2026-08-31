# Exp04 Nsight Compute Analysis

NCU 2024.3.1.0 profiled one isolated `1024^3` launch for V1, V2-T16 and V3-WMMA-FP16 on Orin CC 8.7. Sections were selected from the installed list: LaunchStats, Occupancy, SpeedOfLight, MemoryWorkloadAnalysis, MemoryWorkloadAnalysis_Tables, SchedulerStats, WarpStateStats and SourceCounters. The `.ncu-rep` files remain on Jetson under `/tmp/exp04_ncu_20260831/`; processed CSV exports are committed in `benchmark/raw/`.

| Metric | V1 | V2-T16 | V3-WMMA-FP16 |
|---|---:|---:|---:|
| Block | 16x16 (256) | 16x16 (256) | 32x1 (warp) |
| GPU compute/memory throughput (%) | 95.299 | 84.341 | 99.010 |
| SM throughput (%) | 63.488 | 76.438 | 12.398 |
| Tensor pipe active (%) | 0.046 | 1.384 | 6.127 |
| Registers/thread | 40 | 37 | 40 |
| Eligible warps/cycle | 1.276 | 2.350 | 0.116 |
| Active warps/cycle | 11.933 | 11.928 | 3.987 |
| Shared bank conflicts | 0 | 26 | 0 |
| Shared excessive wavefronts | 0 | 0 | 0 |
| Global excessive-sector derived field | 0 | 0 | 8.388608 Mbyte field in NCU export |

These are profiler counters/derived fields for isolated launches, not formal latency. V2 has a small measured shared conflict count and no excessive wavefronts in this run; the source layout is therefore retained without transpose-style padding. V3 has non-zero tensor-pipe activity and the SASS dump contains `HMMA.16816.F32` instructions in `_Z10wmmaKernelPK6__halfS1_Pfiii`, providing instruction-level evidence that WMMA reached Tensor Core MMA code generation. The NCU export does not expose a single directly comparable Tensor Core utilization percentage beyond the reported pipe activity.

The source-level reuse model is conceptual: V1 has approximately `2*M*N*K` float loads plus `M*N` stores, while a T-tiled stage loads `2*T*T` values and performs `T*T*T` FMAs, giving approximately T-fold reuse and arithmetic intensity near `T/4` FLOP/B (V1 near 0.25 FLOP/B). These are not DRAM traffic measurements; cache and physical memory behavior are only asserted where NCU counters support them.

The dual-reference WMMA closure harness did not modify measured GPU kernels and did not rerun NCU or SASS. Gate C remains supported by the existing committed NCU exports and `HMMA.16816.F32` SASS evidence.
