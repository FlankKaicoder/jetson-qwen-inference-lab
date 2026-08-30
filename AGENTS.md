# Codex Working Agreement

本项目采用 **STATELESS / REPOSITORY-DRIVEN DEVELOPMENT**。每次会话都必须假设没有此前聊天、模型记忆或 terminal scrollback；仓库必须独立提供恢复工作所需的事实。

## Project Source of Truth

事实冲突时按以下顺序裁决：

1. Git commit / branch
2. raw result artifacts
3. experiment report
4. `docs/PROJECT_STATE.md`
5. Codex conversation

Codex conversation 永远不能作为唯一事实来源。`docs/handoff/current_state.md` 只是辅助交接，不得成为恢复项目状态的必要条件。

## Stateless Session Rule

- 不得依赖“此前讨论”“上一个会话已确认”或模型记忆。
- 任何状态、结论和下一步都必须能从 Git、项目状态文档、实验报告或结果文件重新验证。
- 不保存聊天全文；只保存 decision、evidence、state、result 和 rationale。

## Session Start Protocol

开始任何工作前必须：

1. 确认工作目录。
2. 检查 `git status`、当前 branch、HEAD、最近提交和 remote。
3. 阅读 `README.md`、`ROADMAP.md`、本文件和 `docs/project_management.md`。
4. 阅读 `docs/PROJECT_STATE.md` 和 `docs/experiment_index.md`。
5. 阅读当前实验报告及其 raw result artifacts；handoff 仅作为辅助。
6. 对照 Git 和 result artifacts 验证 `PROJECT_STATE`，不一致时先记录差异，不得凭聊天补全。
7. 明确本轮任务后再修改或执行实验。

## Evidence Rule

- 任何性能、正确性、Gate 或 profiler 结论必须有 repository evidence，例如 CSV、TXT、JSON、benchmark log、profiler output、Nsight summary 或实验 Markdown。
- 缺少证据时必须写 `UNKNOWN`；证据存在但不足以形成结论时必须写 `INCONCLUSIVE` 或项目既有的受限判断（例如 `PARTIALLY SUPPORTED`）。
- 理论 occupancy 不得冒充 achieved occupancy；effective bandwidth 不得冒充 DRAM counter；设备 utilization 不得冒充 occupancy。
- 禁止为了“让结果好看”修改实验口径。`REJECT`、`BLOCKED` 和 `INCONCLUSIVE` 都是有效结果。

## Git Rule

执行任何新实验前必须记录：

- branch
- HEAD
- working tree

实验或修改完成后必须记录：

- modified files
- result files
- validation
- commit
- push status

禁止 `git reset --hard`、`git clean -fd`、force push 或历史重写，除非用户明确要求。禁止覆盖旧实验结果、删除历史实验或把大型 Nsight 原始报告、模型权重、TensorRT Engine、凭据提交到 Git。

## Experiment Rule

实验必须形成完整证据链：

```text
Code
→ Build / Execution
→ Benchmark
→ Profiling
→ Analysis
→ Gate
→ Markdown report
→ Git commit
```

- 性能优化必须有 baseline。
- CUDA 优化尽可能进行正确性对比。
- Benchmark 必须保存硬件、软件版本、输入 shape、warmup、repetitions 和 timing 口径。
- 代码“能跑”不等于实验完成；缺少关键证据时不得标记 `PASS`。
- 禁止 Codex 擅自改变研究方向、增加未经要求的大型实验、大范围重构无关代码或自行开展下一实验。

## Session End Protocol

任何产生实际修改的会话结束前必须：

1. 完成与变更风险相称的验证。
2. 如涉及实验，更新对应 experiment report。
3. 更新 `docs/PROJECT_STATE.md`。
4. 如实验状态变化，更新 `docs/experiment_index.md` 和 `results/experiment_registry.csv`。
5. 更新辅助交接 `docs/handoff/current_state.md`。
6. 检查 `git status`、staged diff、大文件、凭据和无关文件。
7. 创建清晰 commit。
8. 网络与权限允许时 push；禁止把“未 push”写成“已同步”。
9. 最终汇报 branch、starting HEAD、new commit、blockers、push status、working tree 和 exact next action。

完成当前任务后停止，不自行开始下一实验。
