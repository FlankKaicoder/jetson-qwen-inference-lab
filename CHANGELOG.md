# Changelog

本文件记录项目管理、实验和基础设施的重要变化。

## [Unreleased]

### Added

- 初始化项目结构、实验规范、Git 工作流与交接文档。
- 完成 Exp00 Windows/Jetson 仓库 bootstrap、SSH 验证与离线 Git 历史同步。
- 创建 Public GitHub 仓库并完成 Windows、GitHub、Jetson 三端 Git 同步。
- 完成 Exp01 CUDA Vector Add correctness 与 block-size benchmark；Nsight Gate 因 profiling 权限不足记录为 BLOCKED。
- 完成 Exp01.1 五轮稳定性与 Nsight 权限审计：block 128 为 5/5 observed fastest，csvEscape 修复验证通过，Nsight 仍因非交互 sudo 认证受阻。
- 建立无状态、仓库驱动的 Codex 工作流，新增项目状态入口并扩展实验索引与证据规则。
- 关闭 Exp01 Nsight Compute Gate：完成 blocks 32/128/256/1024 profiling，H1–H4 最终 SUPPORTED，128 vs 256 微架构成因保持 INCONCLUSIVE，Overall Gate PASS。
- 初始化 Exp02 CUDA Reduction V1-V7，并完成 correctness Gate A：3,087/3,087 执行通过；compute-sanitizer 在 Jetson 上不可用，记录为 N/A；Gate B/C 未开始。
- 完成 Exp02.2 Benchmark/Stability Gate：B1/B2/B3 全部完成，所有 timed configurations correctness PASS，V5/B128 为 N=16,777,229 下最快稳定配置；Gate B PASS，Gate C 待 checkpoint 后开始。
- 完成 Exp02.3 Nsight Gate 与最终收口：V1-V7 common-block profile、V5 block sweep 和 H1-H9 分析完成；Gate A/B/C 均 PASS，Overall PASS，READY_FOR_EXP03；Exp03 未开始。
- 初始化 Exp03 CUDA Matrix Transpose（V0-V4）并完成 Correctness Gate：828/828 bitwise exact PASS，修复 V1/V2 非 tiled grid_y 边界 bug；Gate A PASS，Gate B/C 未开始，READY_FOR_EXP03_BENCHMARK。

- 完成 Exp03.2 两轮正式 Benchmark；correctness/timing PASS，但多项配置 CV 持续超过 5%，Gate B INCONCLUSIVE，Gate C 未开始。
- 完成 Exp03.2b stability diagnosis 与 Formal Benchmark V2：time-based warmup、adaptive CUDA Event window、7 trials 和只读 telemetry 使全部配置 CV <= 0.791%；Gate B PASS，Gate C 待执行。
- 完成 Exp03.3 Nsight Compute 与收口：V1-V4 4096x4096 profile 取得 global sector、shared bank-conflict、occupancy、warp-state 证据；H1-H4 SUPPORTED，Gate A/B/C PASS，Exp03 CLOSED，READY_FOR_EXP04。
