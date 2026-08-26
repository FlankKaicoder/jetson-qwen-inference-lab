# Codex Working Agreement

## 开始工作前

必须先阅读：

1. `README.md`
2. `ROADMAP.md`
3. `docs/project_management.md`
4. `docs/handoff/current_state.md`

修改前必须检查：

- `git status`
- 当前 branch
- 最近 commit

## 不可破坏的规则

- 禁止覆盖旧实验结果。
- 禁止删除历史实验。
- 禁止为了“让结果好看”修改实验口径；`REJECT` 也是有效结果。
- 禁止 Codex 擅自改变研究方向、增加未经要求的大型实验或大范围重构无关代码。
- 不允许用“能跑”替代实验分析。
- 不允许 Codex 自行开展下一实验；完成当前任务后停止。

## 实验与 Benchmark

- 性能优化必须有 baseline。
- CUDA 优化尽可能进行正确性对比。
- Benchmark 必须保存必要的硬件、软件版本、输入 shape、warmup、repetitions 和 timing 口径。
- 大模型权重、TensorRT Engine、大型 Nsight 原始报告不得进入 Git。

## 每轮交接

每轮工作结束必须更新 `docs/handoff/current_state.md`，至少包含：

- 当前 branch
- 当前 commit
- 本轮完成
- 本轮未完成
- 关键实验结果
- 下一步建议
- 工作区状态
