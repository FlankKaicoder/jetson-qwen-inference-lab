# Module Index

本文档是快速模块索引。实际实现主要在 `experiments/`；顶层 `cuda/`、`tensorrt/`、`transformer/`、`runtime/`、`benchmark/` 是学习范围索引。

## CUDA

**用途:** 学习 CUDA kernel 基础、memory access、reduction、transpose、GEMM 和 RMSNorm 的 correctness / performance 演进。

**关键文件:**

- `experiments/Exp01-vector-add/src/vector_add.cu`
- `experiments/Exp02-reduction/src/reduction.cu`
- `experiments/Exp03-matrix-transpose/src/transpose.cu`
- `experiments/Exp03-matrix-transpose/src/transpose_benchmark.cu`
- `experiments/Exp04-gemm/src/gemm.cu`
- `experiments/Exp04-gemm/src/gemm_all.cu`
- `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/rmsnorm_kernel.cuh`
- `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/benchmark.cpp`

**学习重点:** kernel launch 配置、CUDA event timing、correctness gate、memory coalescing、shared memory、tile / padding、GEMM / WMMA，以及 profiler 结果的边界。

## TensorRT

**用途:** 研究 ONNX export / audit、TensorRT engine build、FP16 与 Q/DQ INT8、Plugin V3、engine execution 和 Qwen3-VL vision adapter。

**关键文件:**

- `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/build_engine.py`
- `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/run_engine.py`
- `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/trt_block_parse.py`
- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.h`
- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.cpp`
- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin_demo.cpp`
- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/qwen3_integration/qwen3_rmsnorm_integration.cpp`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_2B1/export_vision_onnx.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_2B2/parse_vision_onnx.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_2C1/build_vision_onnx_fp16.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_3A/run_end_to_end_vision_integration.py`

**学习重点:** network 构建与 ONNX parse 的差异、FP16 / INT8 边界、Plugin V3 capability interface、plugin creator 注册、`execute_async_v3` buffer binding、TensorRT engine 不入 Git 的 artifact 规则。

## Transformer

**用途:** 建立 Qwen3 / Qwen3-VL 的 PyTorch reference、decoder block 拆解和 vision encoder 集成路径。

**关键文件:**

- `experiments/Phase1-qwen3-baseline/src/hf_bf16_reference.py`
- `experiments/Phase1-qwen3-baseline/src/hf_bf16_benchmark.py`
- `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/qwen3_block.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_1A/run_fp16_smoke.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_1B/run_fp16_baseline.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_3A/run_end_to_end_vision_integration.py`

**学习重点:** tokenizer / processor / model loading、Qwen3-like decoder block、vision encoder input shape、PyTorch module replacement contract、端到端 baseline 与 isolated component benchmark 的区别。

## Runtime

**用途:** 分析 TensorRT runtime context、stream ownership、object lifetime、prefill / decode split 和 CUDA Graph feasibility。

**关键文件:**

- `experiments/Phase2-qwen3-quantization/src/phase3a/phase3a_runtime_profile.py`
- `experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py`
- `experiments/Phase2-qwen3-quantization/src/phase3c/phase3c_residual_runtime.py`
- `experiments/Phase2-qwen3-quantization/src/phase3d0/phase3d0_cuda_graph.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_3B2/run_decoder_bottleneck_profile.py`
- `docs/phase2_2c_runtime_architecture.md`
- `docs/phase3a_runtime_bottleneck_attribution.md`
- `docs/phase3b_runtime_object_lifetime.md`

**学习重点:** `LegacyContext` 与 `PersistentContext` 的资源生命周期、`execute_async_v3` stream 语义、CUDA Graph capture / replay 的约束、prefill 与 decode 的不同瓶颈。

## Benchmark

**用途:** 提供从 CUDA microbenchmark 到 Hugging Face baseline、TensorRT isolated benchmark 和 end-to-end stage breakdown 的 measurement path。

**关键文件:**

- `experiments/Phase1-qwen3-baseline/src/hf_bf16_benchmark.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_2D1/run_vision_latency_benchmark.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_1B/run_fp16_baseline.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_3B1/run_end_to_end_stage_breakdown.py`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_3B2/run_decoder_bottleneck_profile.py`
- `experiments/Exp01-vector-add/src/vector_add.cu`
- `experiments/Exp02-reduction/src/reduction.cu`
- `experiments/Exp03-matrix-transpose/src/transpose_benchmark.cu`
- `experiments/Exp04-gemm/src/gemm_all.cu`
- `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/benchmark.cpp`
- `benchmark/README.md`

**学习重点:** timing scope、warmup / repetitions、correctness gate、hardware / software记录、CSV / JSON artifact 命名，以及 isolated speedup 与 end-to-end speedup 的区别。

## Results

**用途:** 保存 compact result artifacts、experiment registry 和可复核证据；大权重、TensorRT engine 和原始 profiler 报告保持在 Jetson-local，不进入 Git。

**关键文件:**

- `results/README.md`
- `results/experiment_registry.csv`
- phase-specific directories under `results/`
- `docs/experiment_index.md`
- `docs/PROJECT_FINAL_REPORT.md`
- `docs/FINAL_STATUS.md`

**学习重点:** 先读 registry 找 experiment id，再读对应 experiment report 和 result directory；不要跳过 protocol 和 limitation 直接引用数字。`UNKNOWN` 表示缺少证据，不表示结论失败。
