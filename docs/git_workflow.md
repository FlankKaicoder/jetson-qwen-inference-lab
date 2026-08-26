# Git 工作流

## 分支与提交

- 默认分支为 `main`。
- 一个实验或一个清晰的基础设施变更使用独立、可审查的提交；较大实验可使用独立分支。
- 提交信息使用简洁的 Conventional Commits 风格，如 `chore: ...`、`docs: ...`、`feat: ...`、`perf: ...`。
- 开始工作前检查状态、分支和最近提交；提交前检查 staged diff。

## 实验历史

- 实验目录随项目推进逐个创建，不预建大量空目录。
- 不覆盖或删除旧实验，不通过重写结果美化结论。
- 被否决或替代的实验分别保留为 `REJECT` 或 `SUPERSEDED`。

## 文件边界

- 代码、构建配置、可复现脚本、实验说明和结果摘要进入 Git。
- 模型权重、TensorRT Engine、缓存、构建产物、日志和大型 Nsight 原始报告不进入 Git。
- SSH 密码、API token、私钥、`.env` 或任何凭据禁止进入 Git。

## 多端同步

Windows、Jetson 与未来 GitHub 应共享同一提交历史。GitHub remote 仅在用户创建空仓库并明确要求后配置；不得猜测 URL。离线同步时可使用临时 Git bundle，验证接收端 HEAD 后删除 bundle。

## 提交前检查

```text
git status --short
git diff --cached --stat
git diff --cached
```

确认没有凭据、大文件、模型、Engine、编译产物或无关个人文件后再提交。
