# Exp03.3 Nsight Compute Analysis

## Method

Gate C profiled one isolated 4096x4096 launch for each V1-V4 with Nsight Compute 2024.3.1.0 on Jetson Orin Nano Super (CC 8.7), using the existing non-interactive `sudo -n` permission path. Sections were selected from the installed set: SpeedOfLight, MemoryWorkloadAnalysis, MemoryWorkloadAnalysis_Tables, Occupancy, WarpStateStats, SchedulerStats, LaunchStats and SourceCounters. Profiler duration is instrumentation data and is not the Formal Benchmark V2 latency.

## Evidence

| Version | Profile SM MHz | Profile duration ms | Memory throughput % | Achieved occupancy % | Eligible warps/scheduler | Long scoreboard | Barrier | Global store sectors/request | Shared bank conflicts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 407.98 | 4.290 | 67.29 | 79.20 | 0.408 | 23.768 | 0.000 | 4.000 | 0 |
| V2 | 510.00 | 17.724 | 99.63 | 79.26 | 0.065 | 62.116 | 0.000 | 32.105 | 0 |
| V3 | 509.99 | 4.786 | 98.37 | 94.28 | 0.452 | 9.224 | 6.424 | 4.000 | 16,349,795 |
| V4 | 407.98 | 3.098 | 90.18 | 94.47 | 0.839 | 15.881 | 1.985 | 4.000 | 53,953 |

Global load requests and sectors were 524,288 and 2,097,152 for every version. V1, V3 and V4 had 524,288 global store requests and 2,097,152 sectors. V2 also had 524,288 requests but 16,832,524 sectors, with 14,735,372 excessive sectors and a 32.105 sectors/request ratio. V1/V3/V4 had zero excessive global sectors.

V3 executed 16,335,476 shared-load and 14,319 shared-store bank conflicts, 16,349,795 total, with 16,252,928 excessive shared wavefronts. V4 reduced this to 25,841 shared-load and 28,112 shared-store conflicts, 53,953 total, with zero excessive shared wavefronts. Both tiled versions used 524,288 shared loads and 524,288 shared stores; their achieved occupancy was nearly identical (94.28% vs 94.47%).

## Interpretation

- H1 is `SUPPORTED`: V1 has coalesced 4-sector loads and stores, zero excessive global sectors, and the Formal Benchmark V2 establishes the copy control at 1.961324 ms / 68.432 useful GB/s. The NCU profile is memory-throughput dominated; direct DRAM throughput is not exposed on this platform.
- H2 is `SUPPORTED`: V2 has the same load traffic as V1 but 32.105 store sectors/request and 14,735,372 excessive sectors. Its Formal Benchmark V2 latency is 5.108x V1. The low eligible-warp rate and high long-scoreboard ratio are consistent with the memory transaction penalty.
- H3 is `SUPPORTED`: V3 restores coalesced global stores (4 sectors/request, zero excessive sectors) and is 3.707x faster than V2 in Formal Benchmark V2. The added shared-memory and barrier work is visible in the profile.
- H4 is `SUPPORTED`: V4 preserves coalesced global traffic while reducing total shared bank conflicts from 16,349,795 to 53,953 and excessive wavefronts to zero. V4 is 1.896x faster than V3 in Formal Benchmark V2. This is direct counter evidence for the padding mechanism, not a claim that padding is the only performance factor.

The V4 Formal Benchmark V2 latency is 1.425565 ms versus V1 at 1.961324 ms (1.376x lower latency). This is a measured result, not a claim that V4 has higher physical DRAM bandwidth. V4 performs global loads/stores, shared stores/loads, synchronization and extra address work; NCU also profiled V1 and V4 at different SM frequencies (407.98 MHz), so profiler duration is not a clean cross-version latency comparison.

## Gate C

C1 global-memory evidence: `PASS` (V1-V4 request/sector counters and excessive-sector comparison).

C2 shared-memory evidence: `PASS` (V3/V4 bank-conflict and excessive-wavefront counters).

C3 occupancy/warp-state evidence: `PASS` (achieved occupancy, eligible warps and scoreboard/barrier data collected for all versions).

Gate C: `PASS`. All H1-H4 are supported by the combined Formal Benchmark V2 and NCU evidence. No direct DRAM throughput value is claimed because the integrated platform reports it as unavailable.
