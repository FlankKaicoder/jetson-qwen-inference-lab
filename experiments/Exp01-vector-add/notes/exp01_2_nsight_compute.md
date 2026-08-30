# Exp01.2 Nsight Compute Gate Closure

## Scope

本轮只恢复 Exp01 的 Gate C。继承 Original Exp01 的 correctness 与 benchmark，以及 Exp01.1 的五轮稳定性结果；没有重跑 Gate A/B、没有修改 kernel、N、数据类型、warmup、repetitions 或 block 集合，也没有开始 Exp02。

## Nsight Environment

- Platform: Jetson Orin Nano Super (`Orin`, compute capability 8.7).
- CUDA / nvcc: 12.6 / V12.6.68.
- Nsight Compute: 2024.3.1.0, build 34702747.
- `ncu --version`: PASS.
- `sudo -n ncu --version`: PASS.
- `sudo -n ncu --list-sections`: PASS.
- Git state before profiling: Windows, GitHub and Jetson all at `exp/01-vector-add@e10f2c06c7d90844cbf425e5ef6c32a413e314ec`, clean.

No sudoers, SSH or system performance configuration was changed.

## Profiling Methodology

The profiled workload is the existing `--single` path:

- `N=16,777,216`, FP32, the existing one-element-per-thread `vectorAddKernel`.
- Blocks: 32, 128, 256 and 1024.
- Warmup: 20; repetitions: 200.
- Kernel filter: `regex:vectorAddKernel`.
- `launch-skip=20`, `launch-count=1`: profile the first measured launch after the original 20 warmup launches.
- The target continues after the profiled launch and performs its existing correctness check; all four runs reported `correctness=PASS`, `max_abs_error=0`.
- Sections: `LaunchStats`, `Occupancy`, `SpeedOfLight`, `MemoryWorkloadAnalysis_Tables`, `SchedulerStats`, `WarpStateStats`, `SourceCounters`.
- Each block required 29 replay passes. NCU replay time and the latency printed by the profiled process are not used as benchmark latency. The comparison table uses the existing five-round stability means.

The `.ncu-rep` files remain outside Git on the Jetson:

- `/tmp/jetson-qwen-exp01-ncu/20260830T144618Z/block32.ncu-rep`
- `/tmp/jetson-qwen-exp01-ncu/20260830T144903Z/block128.ncu-rep`
- `/tmp/jetson-qwen-exp01-ncu/20260830T144903Z/block256.ncu-rep`
- `/tmp/jetson-qwen-exp01-ncu/20260830T144903Z/block1024.ncu-rep`

Git stores the imported details TXT, raw CSV and `benchmark/ncu_profile_summary_20260830T144903Z.csv`. Block 32's first raw export used an unsupported option and produced the preserved 79-byte `block32_raw.csv` error record; the valid export is `block32_raw_metrics.csv`. Profiling itself succeeded and was not repeated.

## Block Comparison

| Metric | 32 | 128 | 256 | 1024 |
| --- | ---: | ---: | ---: | ---: |
| Existing benchmark mean latency (ms) | 6.698908 | 2.207088 | 2.257612 | 3.331965 |
| Theoretical occupancy (%) | 33.33 | 100.00 | 100.00 | 66.67 |
| Achieved occupancy (%) | 24.69 | 85.85 | 83.37 | 57.25 |
| Registers/thread | 16 | 16 | 16 | 16 |
| Driver shared memory/block (KB) | 1.02 | 1.02 | 1.02 | 1.02 |
| Static / dynamic shared memory | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Theoretical active blocks/SM | 16 | 12 | 6 | 1 |
| Theoretical active warps/SM | 16 | 48 | 48 | 32 |
| Achieved active warps/SM | 12.59 | 40.36 | 39.36 | 28.04 |
| Profile SM frequency (MHz) | 510.00 | 509.98 | 407.99 | 509.99 |
| SM throughput (%) | 6.65 | 21.21 | 21.14 | 15.07 |
| Memory throughput (%) | 32.75 | 82.45 | 87.38 | 48.88 |
| DRAM throughput | N/A | N/A | N/A | N/A |
| L1TEX throughput (%) | 6.58 | 19.04 | 19.00 | 13.81 |
| L2 throughput (%) | 32.75 | 82.45 | 87.38 | 48.88 |
| Sysmem read / write (GB/s) | 12.23 / 6.11 | 31.60 / 15.80 | 26.73 / 13.36 | 18.58 / 9.29 |
| Main warp stall | long scoreboard | long scoreboard | long scoreboard | long scoreboard |
| Long-scoreboard NCU ratio | 22.89 | 40.60 | 39.25 | 28.80 |
| Eligible warps/scheduler | 0.12 | 0.25 | 0.28 | 0.54 |
| Global load/store bytes per sector | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 |
| L2 theoretical / ideal global sectors | 6,291,456 / 6,291,456 | same | same | same |

Sysmem GB/s is a direct unit conversion of NCU `sector/us × 32 bytes/sector`; it is profiler evidence, not the existing benchmark's effective-bandwidth measurement. `DRAM Throughput` is `N/A` in this NCU section on the integrated Jetson platform and is not estimated.

## Occupancy Analysis

- Block 32 is limited to 16 theoretical active warps/SM and achieves 24.69% occupancy.
- Blocks 128 and 256 both have 48 theoretical active warps/SM and 100% theoretical occupancy. Achieved occupancy is close: 85.85% versus 83.37%; achieved active warps/SM is 40.36 versus 39.36.
- Block 1024 permits one resident block and 32 theoretical active warps/SM; achieved occupancy is 57.25%.
- Block 128 has 12 resident blocks/SM versus 6 for block 256, but both expose the same theoretical 48 warps/SM. The profiler does not show an occupancy separation large enough to establish the observed benchmark gap's cause.

## SM Analysis

Blocks 128 and 256 have essentially the same SM throughput, 21.21% and 21.14%. This does not support a compute-utilization explanation for block 128's lower benchmark latency. Block 32 and 1024 are lower at 6.65% and 15.07%, consistent with their reduced resident-warps opportunity, but this is not the deciding evidence for 128 versus 256.

## Memory Analysis

- Memory/L2 throughput is 82.45% for block 128 and 87.38% for block 256, far above their 21% SM throughput.
- Block 256's normalized memory throughput is higher, not lower, so the NCU percentage does not explain why its existing benchmark mean is slower.
- Direct DRAM throughput is unavailable (`N/A`). L1TEX throughput is nearly identical for 128/256 (19.04%/19.00%).
- The profile SM clock differs materially: 509.98 MHz for block 128 and 407.99 MHz for block 256. The device was not clock-locked in the existing experiment. This is a profiling-state confounder; it must not be turned into a post-hoc explanation for the independently collected benchmark gap.

The combination of high memory throughput relative to SM throughput and long-scoreboard dominance supports a memory-subsystem-bound classification, without claiming a direct DRAM percentage.

## Warp Stall Analysis

Long scoreboard is the dominant reported stall for every block. For 128/256 the NCU ratios are 40.60/39.25, while eligible warps per scheduler are 0.25/0.28. These values are similar, and block 256 is not worse on either metric. No scheduler or stall metric supplies a reliable causal chain for the 128/256 benchmark difference.

## Coalescing Analysis

Every block reports 32 bytes per sector for global loads and stores, the maximum reported by the section. L2 theoretical global sectors equal ideal sectors (`6,291,456`) and excessive sectors are zero for all four profiles. Therefore the existing contiguous address pattern is supported as efficiently coalesced, and block size does not create a visible coalescing-efficiency difference.

## 128 vs 256 Final Analysis

Observed benchmark phenomenon: block 128 mean `2.207088 ms`, block 256 mean `2.257612 ms`; block 128 is faster by `0.050524 ms` (about 2.238%) in five directionally consistent rounds.

Profiler evidence:

- theoretical occupancy: 100% / 100%; achieved occupancy: 85.85% / 83.37%;
- SM throughput: 21.21% / 21.14%;
- memory throughput: 82.45% / 87.38%;
- long-scoreboard ratio: 40.60 / 39.25;
- global load/store bytes per sector: 32/32 for both;
- registers/thread and shared memory/block are identical;
- theoretical resident blocks differ 12 / 6, but theoretical resident warps are both 48;
- profile SM clocks differ 509.98 / 407.99 MHz.

The metrics establish that both configurations are memory-latency dominated and well coalesced, but they do not isolate a reliable mechanism that makes block 128 faster in the independent benchmark. The clock difference additionally prevents a clean causal comparison of profiler duration or raw traffic rate.

**Final Case C conclusion: `microarchitectural cause remains inconclusive`.**

## H1–H4 Final Status

| Hypothesis | Final status | Evidence |
| --- | --- | --- |
| H1: block size affects performance; larger is not necessarily faster | `SUPPORTED` | Existing five-round benchmark: block 128 is fastest 5/5; 256/1024 are larger and slower. NCU also shows distinct occupancy/throughput regimes across 32/128/1024. |
| H2: higher occupancy does not automatically mean higher performance | `SUPPORTED` | Blocks 128 and 256 both have 100% theoretical occupancy and close achieved occupancy, but different stable benchmark means. Occupancy alone does not explain the gap. |
| H3: Vector Add is memory-bound | `SUPPORTED` | For 128/256, memory throughput is 82.45%/87.38% versus SM throughput 21.21%/21.14%; long scoreboard is the dominant stall. Direct DRAM percentage remains N/A. |
| H4: contiguous addresses have good coalescing | `SUPPORTED` | 32 B/sector for global loads/stores; L2 theoretical sectors equal ideal; excessive sectors are zero for all blocks. |

## Gate Closure

| Gate | Final status | Evidence |
| --- | --- | --- |
| Gate A — Correctness | `PASS / FROZEN` | Original 77/77 sweep, maximum error 0; not repeated. All four profiler target runs also passed correctness. |
| Gate B — Stability | `PASS` | Existing 6 blocks × 5 rounds, 30/30 correctness PASS; not repeated. |
| Gate C — Nsight | `PASS` | Four requested blocks profiled; key launch, occupancy, SM, memory, stall and coalescing evidence saved and formally analyzed. |
| Exp01 Overall | `PASS` | Gates A, B and C are all resolved. The Case C microarchitectural conclusion is a valid evidence-backed outcome. |

## Final Conclusion

Exp01 is closed with complete correctness, stability and profiler evidence. Vector Add is supported as memory-subsystem bound and efficiently coalesced on the recorded Jetson setup. Block 128 remains the stable observed fastest configuration, but the precise 128-versus-256 microarchitectural cause remains inconclusive. Exp01 is `READY_FOR_EXP02`; Exp02 was not started in this session.
