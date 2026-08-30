# Project State

本文件是无历史上下文的新会话恢复项目状态的入口。完整实验方法与分析以实验报告和 raw artifacts 为准；若本页与更高优先级证据冲突，以 Git 和 raw artifacts 为准。

## Current State

| Field | Verified value |
| --- | --- |
| Project | `jetson-qwen-inference-lab` / Jetson Qwen Transformer AI Infra Optimization Lab |
| Current date | `2026-08-30` |
| Repository | `FlankKaicoder/jetson-qwen-inference-lab` |
| Windows path | `E:\nvidia-qwen` |
| Jetson path | `/home/nvidia/projects/jetson-qwen-inference-lab` |
| GitHub | `https://github.com/FlankKaicoder/jetson-qwen-inference-lab` |
| Current phase | Phase 0 — GPU execution model and CUDA foundations |
| Current experiment | Exp01 — CUDA Vector Add + Exp01.1 stability/profiling audit |
| Current branch | `exp/01-vector-add` |
| Current HEAD | Resolve from `git rev-parse HEAD`; this state was reconstructed from starting HEAD `74087eddc3815c45bae655978b57e99279dd4bd8` and is maintained in the governance commit built on it. |
| Main HEAD | `d42ab4aeabc751723a4a2c1036b93a5ed16d3d01` |
| Last completed experiment | Exp00 — Environment & Repository Bootstrap (`PASS`) |
| Experiment status | `BLOCKED` (profiler evidence blocked; correctness and stability passed) |
| Current Gate | Exp01 Overall `PARTIAL`: Gate A `PASS / FROZEN`, Gate B `PASS`, Gate C `BLOCKED` |

## Confirmed Findings

- Baseline is FP32 `C[i] = A[i] + B[i]`, one element per thread, with bounds checking.
- Original correctness sweep: 11 sizes × 7 supported block sizes = 77/77 `PASS`; maximum absolute error `0`.
- Original benchmark: `N = 2^20, 2^22, 2^24`; block sizes `16, 32, 64, 128, 256, 512, 1024`; warmup `20`; repetitions `200`; CUDA Event kernel-only timing.
- At `N=16,777,216`, original block 128 result was `2.171583 ms` and `92.710 GB/s` effective bandwidth.
- Exp01.1 stability audit used `N=16,777,216`, six block sizes, five independent rounds, warmup `20` and repetitions `200` per configuration; all 30/30 runs passed correctness.
- Block 128 was observed fastest in 5/5 rounds: mean/median/sample standard deviation `2.207088/2.206342/0.002887 ms`; block 256 mean was `2.257612 ms`.
- CUDA Occupancy API reports theoretical occupancy of 100% for blocks 128, 256 and 512; their measured latency differs, supporting H2 that higher/equal theoretical occupancy does not determine performance.
- H1 and H2 are `SUPPORTED`. H3 (memory-bound) and H4 (coalescing) are `PARTIALLY SUPPORTED` under the final Exp01.1 audit.

## Rejected / Disproven Hypotheses

- “Larger block size is necessarily faster” is disproven by the measured sweep; block 128 outperformed 256, 512 and 1024 under the recorded conditions.
- “Equal or higher theoretical occupancy guarantees equal or better latency” is disproven by blocks 128, 256 and 512 sharing 100% theoretical occupancy but having different latency.

No repository evidence records a formally `REJECT`-status experiment.

## Open Questions and Evidence Limits

- Achieved occupancy: `UNKNOWN` — no profiler counter was collected.
- DRAM throughput and SM compute throughput: `UNKNOWN` — no profiler counter was collected.
- Warp stall reasons: `UNKNOWN`.
- Memory request/sector/transaction evidence and actual coalescing efficiency: `UNKNOWN`; H4 remains only `PARTIALLY SUPPORTED` from source address pattern.
- Nsight Systems execution or report: `UNKNOWN`; no repository record or `.nsys-rep` exists.
- Formal current-mode theoretical memory-bandwidth utilization: `INCONCLUSIVE`; the audit intentionally did not infer it from an unverified denominator.
- Whether Exp01 should be accepted without profiler evidence or re-profiled with authorized access remains an external review decision.

## Known Blockers

- Nsight Compute profiling requires interactive sudo or administrator-enabled non-root performance-counter access. `sudo -n true` failed with `sudo: a password is required`; no profile was attempted.
- Exp01 is not merged to `main`.

## Current Artifacts

- Experiment report: `experiments/Exp01-vector-add/README.md`
- Stability audit: `experiments/Exp01-vector-add/notes/exp01_1_stability_and_profile.md`
- Original correctness: `experiments/Exp01-vector-add/benchmark/correctness_results.csv`
- Original benchmark: `experiments/Exp01-vector-add/benchmark/vector_add_benchmark.csv`
- Stability raw/summary: `experiments/Exp01-vector-add/benchmark/stability_raw_20260827T075423Z.csv`, `experiments/Exp01-vector-add/benchmark/stability_summary.csv`
- Environment evidence: `experiments/Exp01-vector-add/benchmark/environment.txt`, `experiments/Exp01-vector-add/benchmark/raw/environment_exp01_1_20260827T075423Z.txt`
- Nsight blockers: `experiments/Exp01-vector-add/benchmark/ncu_sections.txt`, `experiments/Exp01-vector-add/benchmark/ncu_sudo_status.txt`
- Important commits: `249ddfb0a6873765bc391922111acfdd489e6d5c`, `74087eddc3815c45bae655978b57e99279dd4bd8`

## Required Next Action

Obtain the external project review decision for Exp01: either accept the documented `PARTIAL` Gate or explicitly authorize and arrange Nsight Compute access for the missing profiler evidence. Do not start Exp02 until that decision is recorded.

## Do-not-repeat Work

- Do not rerun or overwrite the Original Exp01 correctness/benchmark or Exp01.1 stability artifacts merely to reconstruct context.
- Do not repeat the non-interactive sudo probe without a changed authorization/access condition.
- Do not claim achieved occupancy, DRAM/SM throughput, warp stalls, transaction efficiency or Nsight Systems results without new repository evidence.
- Do not start Exp02, change the roadmap, merge to `main`, or modify device power/clock state without explicit direction.

## Last Verified Git State

On `2026-08-30`, before this governance change: Windows was clean on `exp/01-vector-add@74087eddc3815c45bae655978b57e99279dd4bd8`, tracking `origin/exp/01-vector-add`; local and remote `main` were `d42ab4aeabc751723a4a2c1036b93a5ed16d3d01`. Jetson and GitHub parity was supplied as a known baseline but was not independently contacted in this session; after push, verify remote state from Git and do not infer Jetson state from Windows alone.
