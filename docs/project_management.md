# 项目管理与协作规范

## 角色分工

### ChatGPT

- 项目路线
- 知识讲解
- 实验设计
- Gate 判断
- 实验结果复盘
- 下一步决策

### Codex

- 工程实现
- 脚本
- 编译
- 调试
- Benchmark
- 环境检查
- Git 操作
- 实验材料整理

Codex 不得擅自改变研究方向、添加未经要求的大型实验、用“能跑”替代实验分析，或大范围重构无关代码。

### 用户

- 学习
- 理解
- 审核
- 最终掌握项目

## 实验完成标准

每个实验必须至少经历：

```text
问题定义
↓
理论背景
↓
Baseline
↓
实现
↓
正确性验证
↓
Benchmark
↓
性能分析
↓
结论
↓
学习复盘
↓
Git归档
```

代码跑通不等于实验完成。缺少关键证据时不得将实验标记为 `PASS`。

## 实验状态

- `PLANNED`：已定义，尚未开始
- `RUNNING`：正在执行或证据尚未收齐
- `PASS`：完成既定验证并通过 Gate
- `REJECT`：方案经验证无收益或不满足 Gate；这是有效实验结果，不得隐藏
- `BLOCKED`：受环境、依赖、权限或外部条件阻塞
- `SUPERSEDED`：由后续实验替代，历史仍须保留

## 可复现性与证据

每次 Benchmark 至少记录硬件、软件版本、输入 shape、warmup 次数、repetitions、计时口径、baseline 和变更内容。结果摘要（CSV、JSON、Markdown）可提交 Git；模型权重、Engine、大型原始 profile 和敏感信息不得提交。

## Gate 与归档

实验结论必须对应预先定义的判断口径。若改变口径，必须说明原因并保留前后版本。完成后更新实验 README、`docs/experiment_index.md`、`results/experiment_registry.csv`、`CHANGELOG.md` 和交接文档，并创建清晰的 Git 提交。
