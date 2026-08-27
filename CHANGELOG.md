# Changelog

本文件记录项目管理、实验和基础设施的重要变化。

## [Unreleased]

### Added

- 初始化项目结构、实验规范、Git 工作流与交接文档。
- 完成 Exp00 Windows/Jetson 仓库 bootstrap、SSH 验证与离线 Git 历史同步。
- 创建 Public GitHub 仓库并完成 Windows、GitHub、Jetson 三端 Git 同步。
- 完成 Exp01 CUDA Vector Add correctness 与 block-size benchmark；Nsight Gate 因 profiling 权限不足记录为 BLOCKED。
- 完成 Exp01.1 五轮稳定性与 Nsight 权限审计：block 128 为 5/5 observed fastest，csvEscape 修复验证通过，Nsight 仍因非交互 sudo 认证受阻。
