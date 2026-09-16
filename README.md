# Jetson Qwen Transformer AI Infra Optimization Lab

本仓库是一个面向长期学习与工程实践的 Transformer 推理优化实验室。目标是从“会在 Jetson 上部署模型”，提升到“能够分析 Transformer 推理瓶颈，并通过量化、CUDA Kernel、TensorRT Plugin、Attention 优化和 Runtime 优化解决问题”。

## 主实验平台与研究对象

- 主实验平台：Jetson Orin Nano Super
- 研究对象：Qwen3 系列模型

## Final Results Summary

项目已于 2026-09-16 进入 `PROJECT_COMPLETE` closure。最终实验为 Phase 9.3-B2
Qwen3-VL decoder bottleneck attribution；最终报告见
[docs/PROJECT_FINAL_REPORT.md](docs/PROJECT_FINAL_REPORT.md)，最终状态见
[docs/FINAL_STATUS.md](docs/FINAL_STATUS.md)。

最重要的 bounded 结论是：静态 Qwen3-VL Vision Encoder 的 TensorRT FP16 路径在
固定隔离边界约 `2.7x` 快于 PyTorch FP16，但端到端生成没有可声称的有效加速。
Phase 9.3-B2 记录 decode 占固定 workload 生成的 `87.17536311733093%`，decode
kernel time 中 `78.3598%` 属于 GEMM-class；exact attention share、DRAM
counters、achieved bandwidth 与 power 保持 `UNKNOWN`。TensorRT-LLM 是研究
方向，不是已完成的成功迁移结果。

## Repository Organization

- [docs/PROJECT_FINAL_REPORT.md](docs/PROJECT_FINAL_REPORT.md): final project closeout.
- [docs/FINAL_STATUS.md](docs/FINAL_STATUS.md): final gate and limitations.
- [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md): recovery-first project state.
- [docs/experiment_index.md](docs/experiment_index.md): experiment report index.
- [results/experiment_registry.csv](results/experiment_registry.csv): experiment registry.
- `experiments/`: experiment reports, harnesses, and compact evidence.
- `results/`: compact result artifacts and registries.
- Large model weights, ONNX files, TensorRT engines, and raw profiler reports
  remain Jetson-local and are not committed.

## 长期能力方向

- CUDA Kernel
- GPU 性能分析
- TensorRT
- TensorRT Plugin
- TensorRT-LLM
- Quantization
- RMSNorm
- Attention / FlashAttention
- KV Cache
- LLM Runtime

## 项目原则

```text
学习原理
    ↓
提出实验问题
    ↓
Codex工程实现
    ↓
Jetson真实Benchmark
    ↓
Nsight分析
    ↓
ChatGPT复盘
    ↓
用户能够独立解释
```

代码跑通不等于实验完成。每个实验还必须有正确性验证、可复现 Benchmark、性能分析、结论、学习复盘和 Git 归档。

新会话应先阅读 [AGENTS.md](AGENTS.md) 和 [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)，再从 [docs/experiment_index.md](docs/experiment_index.md) 进入当前实验报告与结果证据。实验与协作规则见 [docs/project_management.md](docs/project_management.md)，路线见 [ROADMAP.md](ROADMAP.md)。[docs/handoff/current_state.md](docs/handoff/current_state.md) 仅作为辅助交接，不是项目状态的唯一来源。
