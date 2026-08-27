# Current State

## 当前 branch

`exp/01-vector-add`

## 当前 commit

本文件随实验分支当前 `HEAD` 维护；Exp01 基于 `main@d42ab4aeabc751723a4a2c1036b93a5ed16d3d01`，最终提交以 `git rev-parse HEAD` 为准。

## 本轮完成

- 在 Jetson 从 `main` 创建 `exp/01-vector-add`，未修改或 merge `main`。
- 实现 FP32 一元素一线程 CUDA Vector Add、CPU reference、错误检查、device-property 输出与 Occupancy API 分析。
- 完成 11 个 N × 7 个受支持 block size 的 correctness sweep。
- 完成 3 个 performance N × 7 个 block size 的 CUDA Event kernel-only benchmark。
- 保存 canonical CSV、带时间戳 raw 数据、环境摘要、console log、源码、脚本与完整实验报告。
- 检查现有 Nsight Compute 2024.3.1；未安装或修改任何软件。

## 本轮未完成

- Nsight representative profile 因当前用户 GPU profiling 权限不足而未执行；`ncu --list-sections` 已被拒绝。
- Exp01 尚未 merge 到 `main`，等待用户/ChatGPT 审核。
- Exp02 Reduction 未开始。

## 关键实验结果

- Correctness Gate：`PASS`；77/77 配置通过，最大绝对误差 0。
- Benchmark Gate：`PASS`；21/21 行完成，warmup=20、repetitions=200。
- Nsight Gate：`BLOCKED`；错误为 `Insufficient privileges to launch app for profiling`。
- Overall Gate：`PARTIAL`。
- 三个测试规模最快配置均为 block 128。
- 代表性 `N=16777216`：block 128，2.171583 ms，92.710 decimal GB/s。
- Occupancy API：block 128/256/512 均为 100%，但 block 128 最快，说明 higher occupancy 不自动等于 higher performance。
- 假设：H1 SUPPORTED；H2 SUPPORTED；H3 SUPPORTED；H4 INCONCLUSIVE（Nsight memory metrics 被阻塞）。

## 下一步建议

等待 ChatGPT 审核 Exp01 后决定是否接受 Nsight Gate BLOCKED，或由管理员明确开放 profiling 权限后补做代表性 profile；审核前不得进入 Exp02 Reduction。

## 工作区状态

Jetson 位于 `exp/01-vector-add`；完成提交与 push 后工作树应为 clean 并跟踪 `origin/exp/01-vector-add`。Windows 保持 `main@d42ab4a` clean；未 merge `main`。
