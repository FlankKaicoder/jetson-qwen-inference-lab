# Current State

本文件是辅助交接。独立恢复工作应以 `AGENTS.md`、`docs/PROJECT_STATE.md`、`docs/experiment_index.md`、实验报告和 raw artifacts 为主。

## 当前 branch

`exp/03-matrix-transpose`

## 当前 commit

本轮 starting HEAD 为 `8293d6a203989e981c2f6cf927eaa667681a525a`；Exp03 correctness closure commit 以 `git rev-parse HEAD` 为准。

## 本轮完成

- 重新核验 Windows、GitHub、Jetson 均从 clean 的 `exp/01-vector-add@e10f2c0` 开始。
- `ncu --version`、`sudo -n ncu --version`、`sudo -n ncu --list-sections` 全部 PASS；Nsight Compute `2024.3.1.0`。
- 新增最小 runner，沿用原 `N=16,777,216`、FP32 kernel、warmup 20、repetitions 200，只 profile blocks 32/128/256/1024 的第一个 measured launch。
- 四个 profile 均成功、correctness PASS、最大误差 0；`.ncu-rep` 仅保留在 Jetson `/tmp`，Git 只保存 TXT/CSV。
- 完成 occupancy、SM、memory、warp stall、coalescing 和 128/256 正式分析。
- Exp01 保持冻结；Exp02 Gate A/B/C 均为 `PASS`，Overall 为 `PASS`；Exp03 未开始。

## 关键实验结果

- Benchmark mean latency（既有稳定性数据）32/128/256/1024：`6.698908/2.207088/2.257612/3.331965 ms`。
- Theoretical occupancy：`33.33/100.00/100.00/66.67%`；achieved occupancy：`24.69/85.85/83.37/57.25%`。
- SM throughput：`6.65/21.21/21.14/15.07%`；memory/L2 throughput：`32.75/82.45/87.38/48.88%`。
- 主要 stall 全部为 long scoreboard；128/256 ratio `40.60/39.25`。
- 全部 block 的 global load/store 为 32 B/sector，L2 theoretical sectors 等于 ideal，excessive 为 0。
- 128/256 profile SM frequency 为 `509.98/407.99 MHz`；当前 metrics 无法可靠解释既有 2.238% benchmark 差异。
- 128 vs 256 final：`microarchitectural cause remains inconclusive`。
- H1/H2/H3/H4：`SUPPORTED`；Gate A `PASS / FROZEN`，Gate B `PASS`，Gate C `PASS`，Overall `PASS`。

## 本轮未完成 / 限制

- NCU 直接 `DRAM Throughput` 为 `N/A`；没有估算该值。
- Nsight Systems 未执行，本轮 Gate C 不要求该证据。
- Exp01 未 merge 到 `main`；Exp02 未开始。

## 新增证据

- `experiments/Exp01-vector-add/notes/exp01_2_nsight_compute.md`
- `experiments/Exp01-vector-add/benchmark/ncu_profile_summary_20260830T144903Z.csv`
- `experiments/Exp01-vector-add/benchmark/profiler/20260830T144618Z/`
- `experiments/Exp01-vector-add/benchmark/profiler/20260830T144903Z/`
- `experiments/Exp01-vector-add/scripts/run_ncu_profile.sh`

## 下一步建议

项目所有者审核并接受 Exp01 关闭证据。状态为 `READY_FOR_EXP02`，但只有收到明确新任务后才能设计 Exp02；不得自行开始。

## Push / 工作区状态

最终 commit、GitHub push、Jetson fast-forward 与三端 clean 状态必须由本轮结束时的 Git 命令验证，不能从本段预先推断。

## Exp02.0/Exp02.1 (2026-08-31)
- Branch: exp/02-reduction initialized from d3ee572 and synchronized to GitHub/Jetson.
- V1-V7 implemented; correctness matrix 3,087/3,087 PASS. Max absolute error 1.1754035949707031e-4 (V5 signed N=1048589 B=256); max normalized error 2.086155075380606e-8.
- Compute Sanitizer discovery: N/A, command not installed. Gate A/B/C PASS; Overall PASS; READY_FOR_EXP03.
- No Exp01 files or conclusions modified; no benchmark or Nsight run.

## Exp02.2 Benchmark Gate (2026-08-31)
- Gate B PASS after B1 block survey, B2 scaling and B3 five-round stability; Gate C not started at checkpoint.
- Final blocks V1/B512, V2-V7/B128. V5 fastest 1.626439 ms mean (N=16777229); V7 vs V6 paired CI [0.316896, 0.350628] ms.
- Artifacts: benchmark/raw/block_survey_20260830T172929Z.csv, scaling_20260830T172929Z.csv, stability_20260830T172929Z.csv, summaries, block_candidates and paired_comparisons.
- Next permitted action is Gate C only after checkpoint commit/push; Exp03 not started.

## Exp02 Final Closeout (2026-08-31)
- Exp02 Gate A/B/C all PASS; Overall PASS; READY_FOR_EXP03. Exp03 was then initialized on a new branch; its Gate A is PASS and Gates B/C are not started.
- Final benchmark winner V5/B128, 1.626439 ms mean at N=16777229; V7 vs V6 paired CI [0.316896, 0.350628] ms.
- NCU common profiles V1-V7 and V5 B64/B128/B256/B512 sweep saved under benchmark/profiler/20260831T020000Z; .ncu-rep remains Jetson /tmp only.
- H1 SUPPORTED, H2 PARTIALLY_SUPPORTED, H3 SUPPORTED, H4 SUPPORTED, H5 PARTIALLY_SUPPORTED, H6 SUPPORTED, H7 SUPPORTED, H8 SUPPORTED, H9 INCONCLUSIVE.

## Exp03.0/Exp03.1 (2026-08-31)
- Branch: exp/03-matrix-transpose; initialized from Exp02 closeout `8293d6a`.
- Final correctness artifact: `experiments/Exp03-matrix-transpose/benchmark/raw/correctness_20260831T044304Z.csv`; summary: `correctness_summary_20260831T044304Z.csv`.
- V1/V2/V3/V4 each passed 207/207; total 828/828 bitwise exact; guard and CUDA failures 0.
- Initial V1/V2 non-tiled grid_y bug was fixed and all cases rerun; failed first attempt retained on Jetson.
- Gate A PASS; Gate B/C NOT_STARTED; Overall IN_PROGRESS; READY_FOR_EXP03_BENCHMARK.
- Formal benchmark, Nsight Compute and Exp04 were NOT started.
