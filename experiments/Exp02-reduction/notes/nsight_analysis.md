# Exp02.3 Nsight Microarchitecture Analysis

## Environment and scope

Nsight Compute `2024.3.1.0` on CUDA 12.6 was available with sudo non-interactively. Sections used were `LaunchStats`, `Occupancy`, `SpeedOfLight`, `MemoryWorkloadAnalysis_Tables`, `SchedulerStats`, `WarpStateStats` and `SourceCounters`. Common mechanism profiles used `N=16,777,229`, `B=256`, `launch_skip=0`, and the first-stage kernel. V5 was additionally profiled at B=64/128/256/512. NCU replay duration is not benchmark latency; Gate B CUDA Event results remain authoritative for performance.

## Common-block metrics

| Profile | Achieved occupancy % | Active warps/SM | SM throughput % | Memory throughput % | L2 % | Eligible warps/scheduler | Barrier stall ratio | Divergent branches | Shared bank conflicts | Shared loads/stores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 81.95 | 39.34 | 23.01 | 77.50 | 77.50 | 0.30 | 0.00 | 0 | 0 | 0 / 0 |
| V2 | 95.19 | 45.69 | 71.72 | 35.18 | 11.52 | 1.95 | 1.78 | 0.03 | 68,526 | 6,226,015 / 3,604,535 |
| V3 | 91.48 | 43.91 | 56.74 | 81.60 | 17.42 | 1.02 | 3.41 | 0.03 | 6,952,266 | 1,638,425 / 1,310,740 |
| V4 | 91.00 | 43.68 | 58.54 | 66.21 | 19.74 | 1.00 | 3.18 | 0 | 48,955 | 1,638,425 / 1,310,740 |
| V5 | 90.77 | 43.57 | 62.84 | 70.59 | 39.00 | 1.26 | 3.04 | 0 | 37,608 | 819,225 / 655,380 |
| V6 | 73.18 | 35.13 | 37.69 | 33.39 | 24.99 | 0.72 | 2.51 | 0.03 | 121,318 | 1,638,425 / 1,310,740 |
| V7 | 77.01 | 36.96 | 40.79 | 28.31 | 28.31 | 0.91 | 1.22 | 0.25 | 10,508 | 65,537 / 524,296 |

Theoretical occupancy was 100% for all common B=256 profiles; achieved occupancy is the measured NCU value. Direct DRAM throughput was not available in this integrated platform and is not estimated.

## Hypotheses

- H1 `SUPPORTED` for the combined benchmark and launch metadata: V1 has 25 passes and 25,000 measured kernel launches across B3 rows, while V5 has 4 passes and 4,000 launches; V5 is faster. First-stage NCU also confirms global-only V1 has zero shared traffic. The exact full-pipeline traffic ratio is not inferred from one profiled pass.
- H2 `PARTIALLY_SUPPORTED`: V2 is much slower than V3/V4 and has the modulo source structure, but the supported NCU branch metrics report 100% branch efficiency and only 0.03 average divergent branches. Source/benchmark evidence supports a penalty, not a strong measured divergence attribution.
- H3 `SUPPORTED`: V3 reports 6,952,266 shared bank conflicts versus 48,955 for V4 under the same B=256 first-stage profile, with the same shared load/store instruction counts. This directly supports sequential addressing reducing the observed conflict count on this run.
- H4 `SUPPORTED`: B3 V5/B128 is fastest at 1.626439 ms; its pipeline metadata has 4 passes versus V1's 25. This is an end-to-end benchmark result, not a DRAM claim.
- H5 `PARTIALLY_SUPPORTED`: V6 barrier stall ratio (2.51) is below V5 (3.04), but V6 is slower in B3 (2.655211 vs 1.626439 ms) and still has shared traffic. The profiler does not isolate a complete barrier-count causal chain.
- H6 `SUPPORTED`: V7 reduces shared loads/stores to 65,537/524,296 from V6's 1,638,425/1,310,740 and lowers barrier stall ratio from 2.51 to 1.22. The result supports reduced warp-tail shared activity; it does not claim all block synchronization disappeared.
- H7 `SUPPORTED`: Gate B paired V6-V7 delta is 0.333762 ms, 95% CI [0.316896, 0.350628] ms, excluding zero in favor of V7.
- H8 `SUPPORTED`: B1 is non-monotonic; V1 selects B512 while V2-V7 select B128. V5 NCU sweep achieved occupancy is 59.31/89.92/90.78/91.87% for B64/128/256/512, while Gate B selected B128 by benchmark.
- H9 `INCONCLUSIVE`: B2 shows size-dependent latency and pass counts, but launch overhead, memory traffic, and DVFS are not independently isolated by the timing data.

## Limitations

NCU profile clocks can vary with device state; no clocks or power mode were changed. Bank-conflict and divergence metrics are those actually exposed by NCU 2024.3.1; unsupported or unavailable metrics are not reconstructed. The common profile captures one first-stage kernel, while Gate B measures the complete pipeline.

Gate C is `PASS`: V1-V7 common-block profiles and V5 block sweep completed, raw/details/manifest evidence is saved, and benchmark and profiler latency are kept separate.