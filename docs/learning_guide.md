# Learning Guide

本指南为 GPT 或后续读者提供从 CUDA 到 TensorRT/Qwen 优化的阅读顺序。建议每读一个入口，同时查看对应 `README.md`、experiment report 和 `results/` evidence，而不是只看代码得出性能结论。

## Stage 1 CUDA基础

阅读：

1. `experiments/Exp01-vector-add/src/vector_add.cu`
2. `experiments/Exp02-reduction/src/reduction.cu`
3. `experiments/Exp03-matrix-transpose/src/transpose.cu`
4. `experiments/Exp03-matrix-transpose/src/transpose_benchmark.cu`
5. `experiments/Exp04-gemm/src/gemm.cu`
6. `experiments/Exp04-gemm/src/gemm_all.cu`

原因：

- `vector_add.cu` 是最小 CUDA 入口，可以看到 kernel launch、block size、CUDA event timing 和 correctness loop。
- `reduction.cu` 展示同一问题从简单实现到多种 reduction 变体的演进，适合理解 warp、shared memory、bank conflict 和 branch 差异。
- transpose 代码展示 memory coalescing、tiled access、padding 和 adaptive benchmark 的实际取舍。
- GEMM 代码把问题推进到 compute-intensive kernel，并可对照 cuBLASLt / CUTLASS feasibility 结果。

## Stage 2 Transformer

阅读：

1. `experiments/Phase1-qwen3-baseline/src/hf_bf16_reference.py`
2. `experiments/Phase1-qwen3-baseline/src/hf_bf16_benchmark.py`
3. `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/qwen3_block.py`
4. `experiments/Phase9-qwen3-vl-migration/src/phase9_1A/run_fp16_smoke.py`
5. `experiments/Phase9-qwen3-vl-migration/src/phase9_1B/run_fp16_baseline.py`

原因：

- Phase 1 先建立可复现的 Hugging Face Qwen3 reference 和 baseline。
- `qwen3_block.py` 把 decoder block 拆成可导出的 `RMSNorm`、attention、MLP 结构，是连接模型概念和 TensorRT graph 的关键。
- Phase 9 展示从 Qwen3 text model 到 Qwen3-VL vision model 的加载、smoke test 和 baseline 差异。

## Stage 3 Quantization

阅读：

1. `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/build_engine.py`
2. `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/run_engine.py`
3. `experiments/Phase2-qwen3-quantization/src/phase2_3b/evaluate_calibration.py`
4. `experiments/Phase2-qwen3-quantization/src/phase2_3e/build_mixed_runtime.py`
5. `experiments/Phase2-qwen3-quantization/src/phase2_3f/phase2_3f_compare.py`
6. `experiments/Phase9-qwen3-vl-migration/src/phase9_2C1/build_vision_onnx_fp16.py`

原因：

- `trt_precision_probe` 用最小 linear network 隔离 FP16 与 explicit Q/DQ INT8 的行为。
- calibration、mixed runtime 和 compare 脚本展示从 scale / calibration 到 mixed engine 再到结果对比的路径。
- Phase 9 的 FP16 vision build 是一个较完整的 TensorRT conversion 案例，但不要把 FP16 结果当作 INT8 结论。

## Stage 4 TensorRT

阅读：

1. `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/trt_block_parse.py`
2. `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.h`
3. `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.cpp`
4. `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin_kernel.cu`
5. `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin_demo.cpp`
6. `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/qwen3_integration/qwen3_rmsnorm_integration.cpp`
7. `experiments/Phase9-qwen3-vl-migration/src/phase9_2B1/export_vision_onnx.py`
8. `experiments/Phase9-qwen3-vl-migration/src/phase9_2B2/parse_vision_onnx.py`
9. `experiments/Phase9-qwen3-vl-migration/src/phase9_2C1/build_vision_onnx_fp16.py`
10. `experiments/Phase9-qwen3-vl-migration/src/phase9_3A/run_end_to_end_vision_integration.py`

原因：

- `trt_block_parse.py` 说明 ONNX graph 如何进入 TensorRT network。
- RMSNorm plugin 覆盖 Plugin V3 接口、creator 注册、CUDA kernel launch、demo engine 和 Qwen3 primitive/plugin 对比。
- Phase 9 覆盖 vision encoder 的 export、graph audit、FP16 build、engine execution 和 PyTorch adapter replacement，是本仓库最完整的 TensorRT 集成路径。

## Stage 5 Optimization

阅读：

1. `experiments/Phase5-cuda-feasibility/src/phase5a_cublaslt_benchmark.cu`
2. `experiments/Phase5-cuda-feasibility/src/phase5a_cutlass_benchmark.cu`
3. `experiments/Phase6-attention-matmul-attribution/` 下的 report、script 和 compact results
4. `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/benchmark.cpp`
5. `experiments/Phase2-qwen3-quantization/src/phase3a/phase3a_runtime_profile.py`
6. `experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py`
7. `experiments/Phase2-qwen3-quantization/src/phase3d0/phase3d0_cuda_graph.py`
8. `experiments/Phase9-qwen3-vl-migration/src/phase9_2D1/run_vision_latency_benchmark.py`
9. `experiments/Phase9-qwen3-vl-migration/src/phase9_3B1/run_end_to_end_stage_breakdown.py`
10. `experiments/Phase9-qwen3-vl-migration/src/phase9_3B2/run_decoder_bottleneck_profile.py`

原因：

- Phase 5 用 cuBLASLt 和 CUTLASS 建立 GEMM feasibility baseline。
- Phase 6 学习如何把 runtime 时间归属到 operator / kernel，并接受 exact attention share 可能保持 `UNKNOWN`。
- Phase 8 比较 RMSNorm CUDA 版本和 TensorRT Plugin 的 correctness 与性能。
- Phase 3 runtime 工作展示 context、CUDA Graph、stream ownership 和 profiling 边界。
- Phase 9 把 isolated vision speedup 与 end-to-end stage breakdown 分开，避免把组件加速误读为完整生成加速。

## Evidence Boundary

- `PASS` 只在对应 protocol、correctness gate 和 benchmark 边界内有效。
- `REJECT`、`BLOCKED`、`INCONCLUSIVE` 和 `UNKNOWN` 是有效状态，不应被解释成失败或成功。
- profiler 工具名称不等于 DRAM bandwidth、achieved occupancy 或 power 的直接证据；这些结论必须回到对应 CSV / JSON / NCU / NSYS summary。
