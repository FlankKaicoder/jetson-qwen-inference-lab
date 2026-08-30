# Current State

本文件是辅助交接。独立恢复工作应以 `AGENTS.md`、`docs/PROJECT_STATE.md`、`docs/experiment_index.md`、实验报告和 raw artifacts 为主。

## 当前 branch

`exp/01-vector-add`

## 当前 commit

本轮 starting HEAD 为 `74087eddc3815c45bae655978b57e99279dd4bd8`。治理变更的最终 commit 以 `git rev-parse HEAD` 为准。

## 本轮完成

- 从 Git、两个 Exp01 提交、源码、脚本、实验报告、CSV 和 profiler 权限记录重建项目状态，不依赖聊天历史。
- 增补 `AGENTS.md`，固定 source-of-truth、stateless session、evidence、Git、实验和 session start/end 规则。
- 新增 `docs/PROJECT_STATE.md`，作为新会话的当前状态入口。
- 复用并扩展 `docs/experiment_index.md`，登记仓库中实际存在的 Exp00 与 Exp01；未提前登记 Exp02。
- 更新 README 导航与 Changelog；未修改 CUDA 实现、benchmark 数据、profiler 数据或历史实验结果。

## 本轮未完成

- Exp01 Nsight Compute profile 仍未执行；交互式 sudo/管理员权限 blocker 未变化。
- Exp01 尚未 merge 到 `main`；Exp02 未开始。
- 本轮未连接 Jetson，不能从 Windows 单独证明 Jetson 当前 HEAD。

## 关键实验结果

- Gate A — Correctness：`PASS / FROZEN`；Original Exp01 77/77，最大误差 0。
- Gate B — Stability：`PASS`；30/30 独立测量 correctness PASS。
- Block 128 mean/median/sample-std/CV = `2.207088/2.206342/0.002887 ms/0.131%`，五轮最快 5/5。
- Block 256 mean = `2.257612 ms`；`256 - 128` 均值差 `0.050524 ms`。
- Gate C — Nsight：`BLOCKED`；achieved occupancy、DRAM/SM throughput、warp stalls 与 transaction/sector counter 为 `UNKNOWN`。
- Nsight Systems：`UNKNOWN`；仓库中没有执行记录或结果。
- H1/H2 `SUPPORTED`；H3/H4 `PARTIALLY SUPPORTED`；Overall Gate `PARTIAL`。

## 下一步建议

等待 Exp01 外部审核：接受已记录的 `PARTIAL` Gate，或明确授权并安排 Nsight Compute profiling 权限。审核决定记录前不要开始 Exp02。

## Push 状态

`NOT PUSHED`。向 `origin/exp/01-vector-add` 的 push 未执行，因为外部写入审批拒绝导出包含本地路径与环境元数据的治理文档；未采用绕过方式。远端仍停留在 `74087eddc3815c45bae655978b57e99279dd4bd8`。

## 工作区状态

本地 `exp/01-vector-add` 在本轮治理 commit 上，working tree 应为 clean，并领先 `origin/exp/01-vector-add` 1 个提交。最终状态由本轮结束时的 Git 命令验证。
