# Project State

本文件是无历史上下文的新会话恢复项目状态的入口。完整实验方法与分析以实验报告和 raw artifacts 为准；若本页与更高优先级证据冲突，以 Git 和 raw artifacts 为准。

## Current State

| Field | Verified value |
| --- | --- |
| Project | `jetson-qwen-inference-lab` / Jetson Qwen Transformer AI Infra Optimization Lab |
| Current date | `2026-08-31` |
| Repository | `FlankKaicoder/jetson-qwen-inference-lab` |
| Windows path | `E:\nvidia-qwen` |
| Jetson path | `/home/nvidia/projects/jetson-qwen-inference-lab` |
| GitHub | `https://github.com/FlankKaicoder/jetson-qwen-inference-lab` |
| Current phase | Phase 0 — GPU execution model and CUDA foundations |
| Current experiment | No active experiment; Exp02 is closed. Exp03 has not started. |
| Current branch | `exp/02-reduction` |
| Current HEAD | Resolve from `git rev-parse HEAD`; this branch's latest commit is the Exp02 closeout. |
| Main HEAD | `d42ab4aeabc751723a4a2c1036b93a5ed16d3d01` |
| Last completed experiment | Exp02 — CUDA Reduction initialization, benchmark and Nsight audit |
| Experiment status | `PASS` |
| Current Gate | Exp02 Overall `PASS`: Gate A `PASS`, Gate B `PASS`, Gate C `PASS` |
| Readiness | `READY_FOR_EXP03`; Exp03 has not started. |

## Confirmed Findings

- Baseline is FP32 `C[i] = A[i] + B[i]`, one element per thread, with bounds checking.
- Original correctness sweep: 11 sizes × 7 supported block sizes = 77/77 `PASS`; maximum absolute error `0`.
- Original benchmark: `N = 2^20, 2^22, 2^24`; block sizes `16, 32, 64, 128, 256, 512, 1024`; warmup `20`; repetitions `200`; CUDA Event kernel-only timing.
- Exp01.1 stability: `N=16,777,216`, six block sizes, five independent rounds; 30/30 correctness PASS. Block 128 was fastest in 5/5 rounds with mean `2.207088 ms`; block 256 mean was `2.257612 ms`.
- Nsight Compute 2024.3.1 successfully profiled blocks 32, 128, 256 and 1024 using the original workload. All four target runs passed correctness.
- Theoretical occupancy for 32/128/256/1024 is `33.33/100.00/100.00/66.67%`; achieved occupancy is `24.69/85.85/83.37/57.25%`.
- SM throughput is `6.65/21.21/21.14/15.07%`; memory/L2 throughput is `32.75/82.45/87.38/48.88%`.
- Long scoreboard is the main reported warp stall for every profiled block.
- Global loads/stores use 32 B/sector; L2 theoretical global sectors equal ideal sectors and excessive sectors are zero for all four blocks.
- H1, H2, H3 and H4 are `SUPPORTED` under the final evidence set.

## Rejected / Disproven Hypotheses

- “Larger block size is necessarily faster” is disproven; block 128 outperformed 256, 512 and 1024 under the recorded benchmark conditions.
- “Equal or higher occupancy guarantees equal or better latency” is disproven; blocks 128 and 256 both have 100% theoretical occupancy and close achieved occupancy, but different stable benchmark means.

No repository evidence records a formally `REJECT`-status experiment.

## Remaining Inconclusive / N/A Items

- 128 vs 256 precise cause: `microarchitectural cause remains inconclusive`. Occupancy, SM, stall and coalescing metrics are close; 256's normalized memory throughput is higher, not lower.
- Profile SM frequency differed for 128/256 (`509.98/407.99 MHz`), so profiler duration and raw traffic rate cannot provide a clean causal comparison of the independent benchmark gap.
- Direct NCU `DRAM Throughput`: `N/A` on this integrated platform; it was not estimated.
- Nsight Systems: `UNKNOWN` and not required for the completed Gate C definition.

## Known Blockers

- No blocker remains for Exp01.
- Exp01 is not merged to `main`; merging requires explicit direction.

## Current Artifacts

- Main report: `experiments/Exp01-vector-add/README.md`
- Stability audit: `experiments/Exp01-vector-add/notes/exp01_1_stability_and_profile.md`
- Nsight closure report: `experiments/Exp01-vector-add/notes/exp01_2_nsight_compute.md`
- Original correctness/benchmark: `experiments/Exp01-vector-add/benchmark/correctness_results.csv`, `experiments/Exp01-vector-add/benchmark/vector_add_benchmark.csv`
- Stability raw/summary: `experiments/Exp01-vector-add/benchmark/stability_raw_20260827T075423Z.csv`, `experiments/Exp01-vector-add/benchmark/stability_summary.csv`
- NCU summary: `experiments/Exp01-vector-add/benchmark/ncu_profile_summary_20260830T144903Z.csv`
- NCU TXT/CSV evidence: `experiments/Exp01-vector-add/benchmark/profiler/20260830T144618Z/`, `experiments/Exp01-vector-add/benchmark/profiler/20260830T144903Z/`
- Reproducible runner: `experiments/Exp01-vector-add/scripts/run_ncu_profile.sh`
- `.ncu-rep` files: Jetson-local `/tmp/jetson-qwen-exp01-ncu/20260830T144618Z/` and `/tmp/jetson-qwen-exp01-ncu/20260830T144903Z/`; intentionally outside Git.
- Important prior commits: `249ddfb0a6873765bc391922111acfdd489e6d5c`, `74087eddc3815c45bae655978b57e99279dd4bd8`, `e10f2c06c7d90844cbf425e5ef6c32a413e314ec`.

## Required Next Action

Review and accept the closed Exp01 evidence. If the project owner explicitly authorizes the next experiment, design Exp02 from the roadmap in a new task. Do not start Exp02 automatically.

## Do-not-repeat Work

- Do not rerun or overwrite Original Exp01 correctness/benchmark, Exp01.1 stability, or Exp01.2 profiler artifacts merely to reconstruct context.
- Do not repeat the completed sudo/NCU permission setup or modify sudoers.
- Do not claim a precise 128/256 microarchitectural cause; the final evidence-backed status is `INCONCLUSIVE`.
- Do not estimate direct DRAM throughput where NCU reports `N/A`.
- Do not start Exp02, merge `main`, change the roadmap, or modify device power/clock state without explicit direction.

## Last Verified Git State

Before Exp01.2 profiling on `2026-08-30`, Windows, GitHub and Jetson were clean at `exp/01-vector-add@e10f2c06c7d90844cbf425e5ef6c32a413e314ec`; local and remote `main` were `d42ab4aeabc751723a4a2c1036b93a5ed16d3d01`. The final closure commit and push/pull state must be verified from Git at session end.

## Exp02.0/Exp02.1 Update (2026-08-31)

- Branch: exp/02-reduction; starting HEAD: d3ee5725374d780fcd3ae84fd9aa57e4d238ffb1.
- Exp02 Gate A: PASS. V1-V7 correctness: 3,087/3,087 executions PASS; max absolute error 1.1754035949707031e-4; max normalized error 2.086155075380606e-8.
- Gate B: PASS. Gate C: PASS. Overall: PASS.
- Compute Sanitizer discovery: N/A because compute-sanitizer is not installed on Jetson.
- Exp01 frozen conclusions unchanged, including 128 vs 256 microarchitectural cause remains inconclusive.

## Exp02.2 Benchmark Update (2026-08-31)

- Gate A PASS; Gate B PASS; Gate C PASS; Overall PASS; READY_FOR_EXP03.
- B1/B2/B3 raw and summary artifacts saved under experiments/Exp02-reduction/benchmark/raw/.
- Final B3 candidates: V1/B512, V2-V7/B128. V5 is fastest at 1.626439 ms mean for N=16777229.
- V6 vs V7 paired delta (V6-V7) = 0.333762 ms, 95% CI [0.316896, 0.350628] ms; significant in favor of V7.
- No Exp03 started; no Exp01 content changed.

## Exp02.3 Nsight / Final Update (2026-08-31)

- Gate A PASS; Gate B PASS; Gate C PASS; Overall PASS; READY_FOR_EXP03.
- NCU 2024.3.1 first-stage common profiles V1-V7 at B256 plus V5 B64/B128/B256/B512 sweep completed. Direct DRAM throughput remains N/A where unsupported; no estimate made.
- H2 is PARTIALLY_SUPPORTED due low measured divergence despite modulo source; H3 SUPPORTED by V3/V4 bank-conflict counters; H5 PARTIALLY_SUPPORTED; H6 SUPPORTED; H7 SUPPORTED; H8 SUPPORTED; H9 INCONCLUSIVE.
- Exp03 was NOT started. Exp01 remains frozen unchanged.
