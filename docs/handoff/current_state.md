# Current State

## 当前 branch

`exp/01-vector-add`

## 当前 commit

本文件随实验分支当前 `HEAD` 维护；Exp01.1 从 `249ddfb0a6873765bc391922111acfdd489e6d5c` 开始，最终审计提交以 `git rev-parse HEAD` 为准。

## 本轮完成

- 只读记录当前 nvpmodel、GPU devfreq、可读取的 memory clock、jetson_clocks 权限状态与前后 tegrastats 温度。
- 修复 `csvEscape()` 双引号多写问题，`abc` 与 `a"b` 最小测试 PASS。
- 固定 `N=16777216`、warmup 20、repetitions 200，按确定性轮换顺序完成 6 blocks × 5 个独立 round。
- 保存时间戳 raw CSV、summary、环境状态、console log 与 Exp01.1 审计笔记；未覆盖 Original Exp01 数据。
- 按授权执行 `sudo -n true`；非交互认证失败后停止 Nsight 部分，未提示密码、未修改系统配置。
- 未修改 nvpmodel/jetson_clocks，未 merge `main`，未开始 Exp02。

## 本轮未完成

- Block 32/128/256/1024 的 Nsight profile 未执行：`sudo -n true` 需要交互式认证。
- Achieved occupancy、DRAM/SM throughput、warp stalls 与 memory transaction/sector counter 未获得。
- Exp01 尚未 merge 到 `main`；Exp02 Reduction 未开始。

## 关键实验结果

- 当前 nvpmodel：`25W`、mode ID `1`。
- Clock 状态：GPU pre-run current/min/max = `306/306/918 MHz`；CUDA memory clock property `918000 kHz`；非 root `jetson_clocks --show` 不可用，未锁频或切换模式。
- tegrastats：pre-run GPU 约 `49.8–50.0°C`，post-run 最高样本 `54.718°C`。
- Gate A — Correctness：`PASS / FROZEN`；Original Exp01 77/77、最大误差 0，本轮未重跑完整 sweep。
- Gate B — Stability：`PASS`；30/30 独立测量 correctness PASS，raw/summary/设备状态完整。
- Block 128 mean/median/sample-std/CV = `2.207088/2.206342/0.002887 ms/0.131%`，五轮最快分布 `5/5`。
- Block 256 mean = `2.257612 ms`；`256 - 128` 均值差 `0.050524 ms`，五轮均为正且明显大于 run-to-run variation，差异稳定。
- Gate C — Nsight：`BLOCKED`；`interactive sudo authentication required`，没有 profiler counter。
- Theoretical occupancy 与 achieved occupancy 已严格区分；后者未采集，tegrastats GPU utilization 未当作 occupancy。
- Memory-bound：`PARTIALLY SUPPORTED`；低算术强度与 benchmark 支持，但缺少 DRAM/SM counter。
- Coalescing：`PARTIALLY SUPPORTED`；源码连续访问支持，但缺少 transaction/sector counter。
- H1 `SUPPORTED`；H2 `SUPPORTED`；H3 `PARTIALLY SUPPORTED`；H4 `PARTIALLY SUPPORTED`。
- Overall Gate：`PARTIAL`。

## 下一步建议

等待 ChatGPT 审核 Exp01。

## 工作区状态

Jetson 位于 `exp/01-vector-add`；本轮提交并 push 后应 clean 且跟踪 `origin/exp/01-vector-add`。Windows 保持 `main@d42ab4a` clean；未 merge `main`。
