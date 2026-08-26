# Roadmap

本路线描述能力建设方向，不预设实验结论。具体实验编号和执行顺序可以根据前序实验结果调整。

## Phase 0：GPU 执行模型与 CUDA 基础

- Vector Add
- Reduction
- Matrix Transpose
- GEMM
- Softmax 基础

## Phase 1：Qwen3 Baseline 与 TensorRT-LLM

建立可复现的 Qwen3 推理基线，理解 TensorRT-LLM 的构建、运行与测量链路。

## Phase 2：LLM 量化

- FP16
- INT8
- Mixed Precision
- INT4 / Weight-only

## Phase 3：Transformer 算子优化

- Softmax
- LayerNorm
- RMSNorm

## Phase 4：Attention 优化

- Naive Attention
- Tiled Attention
- FlashAttention 思想
- TensorRT / TensorRT-LLM Attention 实现分析

## Phase 5：TensorRT Plugin

围绕已识别的算子瓶颈学习并实现可验证的 TensorRT Plugin。

## Phase 6：LLM Runtime

- KV Cache
- CUDA Graph
- Memory Management
- TensorRT-LLM Runtime
- llama.cpp
- vLLM

## Phase 7：Qwen3-VL 多模态迁移验证

将前序能力迁移到多模态推理链路，并验证适用范围与限制。
