# Code Architecture

本文档是 `final-analysis` 分支的代码阅读地图，面向后续上传 ZIP 后的 GPT 分析。它只描述仓库结构和代码入口，不改变任何实验代码。

## 1. Overall Architecture

```text
Model
  Qwen3 / Qwen3-VL checkpoints
  Hugging Face PyTorch reference and baseline
        |
        v
Conversion
  ONNX export and graph audit
  TensorRT direct-network / QDQ / FP16 build
        |
        v
Runtime
  Hugging Face inference path
  TensorRT engine execution and adapter replacement
  persistent context, stream ownership, CUDA Graph
        |
        v
Optimization
  CUDA kernel variants
  RMSNorm TensorRT plugin
  mixed-precision and quantization studies
  kernel/operator attribution
        |
        v
Benchmark
  correctness gates
  CUDA-event latency / throughput
  NCU / NSYS compact summaries
  CSV, JSON, Markdown reports, and bounded gates
```

重要边界：仓库中最强的结论是隔离边界结论。例如 Qwen3-VL Vision Encoder 的 TensorRT FP16 路径在固定 workload 下约 `2.7x` 快于 PyTorch FP16，但不能直接推广为端到端生成加速。decode 主导 workload 时，isolated speedup 不等于 end-to-end speedup。

## 2. Directory Map

| 目录 | 作用 | 核心文件 / 子目录 |
| --- | --- | --- |
| `docs/` | 项目状态、最终报告、实验索引、架构说明和学习文档。 | `README.md`, `PROJECT_FINAL_REPORT.md`, `FINAL_STATUS.md`, `PROJECT_STATE.md`, `experiment_index.md`, `final_analysis_readme.md`, `code_architecture.md`, `learning_guide.md`, `module_index.md` |
| `experiments/` | 实际实现、实验报告、脚本和 compact evidence 的主树。 | `Exp01-vector-add/`, `Exp02-reduction/`, `Exp03-matrix-transpose/`, `Exp04-gemm/`, `Phase1-qwen3-baseline/`, `Phase2-qwen3-quantization/`, `Phase5-cuda-feasibility/`, `Phase6-attention-matmul-attribution/`, `Phase8-rmsnorm-optimization/`, `Phase9-qwen3-vl-migration/` |
| `results/` | compact result artifacts 和实验注册表；大权重、engine 和原始 profiler 报告不进 Git。 | `README.md`, `experiment_registry.csv`, phase-specific result directories |
| `cuda/` | CUDA 学习范围索引，不是实际 kernel 实现目录。 | `README.md` |
| `tensorrt/` | TensorRT、plugin 和 TensorRT-LLM 学习范围索引。 | `README.md` |
| `transformer/` | Transformer operator、attention 和量化概念索引。 | `README.md` |
| `runtime/` | KV cache、CUDA Graph、context 生命周期等 runtime 索引。 | `README.md` |
| `benchmark/` | benchmark 口径和标准索引；实际 benchmark 代码在实验树内。 | `README.md` |
| `scripts/` | 仓库级操作和学习脚本索引。 | `README.md` |
| `references/` | 参考资料。 | README 和 reference notes |

Phase 3 的 runtime 代码保留在 `experiments/Phase2-qwen3-quantization/src/phase3*/` 下，因为那些 runtime 研究从 Phase 2 TensorRT 工作演化而来。顶层 `cuda/`、`tensorrt/`、`transformer/`、`runtime/`、`benchmark/` 主要是索引，不是代码主目录。

## 3. Important Entry Points

### CUDA

| 入口文件 | 入口函数 | 调用关系 |
| --- | --- | --- |
| `experiments/Exp01-vector-add/src/vector_add.cu` | device: `vectorAddKernel`; host: `runCase`, `main` | `main` 解析参数并进入 `runCase`；`runCase` 分配输入输出、launch `vectorAddKernel`、做 CUDA-event timing 和正确性检查。 |
| `experiments/Exp02-reduction/src/reduction.cu` | `executeVersion`, `runCorrectness`, `main` | `main` 按 mode 进入 benchmark 或 correctness；`executeVersion` 选择 reduction V1-V7；`runCorrectness` 输出 raw 和 summary。 |
| `experiments/Exp03-matrix-transpose/src/transpose.cu` | `launchTranspose`, `runCase`, `main` | `main` 进入 correctness path；`runCase` 生成 pattern、launch transpose kernel 并比较结果。 |
| `experiments/Exp03-matrix-transpose/src/transpose_benchmark.cu` | `runCalibration`, `runLegacy`, `runAdaptive`, `main` | `main` 分发 calibration、legacy benchmark、adaptive benchmark 或 sanity path；入口先做 correctness sanity。 |
| `experiments/Exp04-gemm/src/gemm.cu`, `experiments/Exp04-gemm/src/gemm_all.cu` | `runCase`, `timed`, `main` | `main` 分发 info、correctness、WMMA correctness 和 benchmark；`runCase` 执行 GEMM/WMMA kernel 并比较 reference。 |
| `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/benchmark.cpp` | `run_case`, `run_correctness`, `main` | `main` 分发 correctness 和 benchmark；`run_case` 执行 FP16/BF16 RMSNorm 版本并保存结果。 |
| `experiments/Phase5-cuda-feasibility/src/phase5a_cublaslt_benchmark.cu`, `experiments/Phase5-cuda-feasibility/src/phase5a_cutlass_benchmark.cu` | `main` | cuBLASLt / CUTLASS feasibility benchmark 入口，用于对比库级 GEMM 实现和记录 profiling 边界。 |

### TensorRT

| 入口文件 | 入口函数 / 类 | 调用关系 |
| --- | --- | --- |
| `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/build_engine.py` | `build_engine`, `main` | `main` 调用 `build_engine`；函数直接搭建 TensorRT network，构造 FP16 或 explicit Q/DQ INT8 candidate，并调用 `build_serialized_network`。 |
| `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/run_engine.py` | `main` | 反序列化 engine，分配 device buffer，设置 tensor address，调用 `execute_async_v3` 并记录 finite / correctness 状态。 |
| `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/trt_block_parse.py` | `main` | 解析导出的 Qwen3-like decoder block ONNX graph，搭建 TensorRT engine，执行并验证输出。 |
| `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.h`, `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.cpp` | class `RMSNormPlugin`; method `enqueue` | `RMSNormPlugin` 实现 TensorRT Plugin V3 接口；`enqueue` 调用 `launchRMSNormPluginKernel`。creator 在 `rmsnorm_plugin_creator.cpp` 注册。 |
| `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin_demo.cpp` | `main` | 注册 plugin，构建 primitive 与 plugin 两个 engine，反序列化并执行对比。 |
| `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/qwen3_integration/qwen3_rmsnorm_integration.cpp` | `buildPrimitive`, `buildPlugin`, `runEngine`, `main` | `main` 注册 plugin；`buildPrimitive` / `buildPlugin` 分别构造 primitive RMSNorm 和 plugin RMSNorm；`runEngine` 执行两个 engine 并比较结果。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_2B1/export_vision_onnx.py` | class `StaticQwen3VLVision`; `main` | 加载 Qwen3-VL visual module，包成 static-shape wrapper，导出 ONNX。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_2B2/parse_vision_onnx.py` | `main` | 审计 vision ONNX graph，记录 TensorRT 可解析性和节点形态。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_2C1/build_vision_onnx_fp16.py` | `build_serialized_network` path; `main` | 解析 ONNX，配置 FP16，调用 `builder.build_serialized_network` 生成 vision engine。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_3A/run_end_to_end_vision_integration.py` | class `TensorRTVisionAdapter`; `main` | 反序列化 vision engine，包装为 `TensorRTVisionAdapter`，替换 PyTorch 模型中的 `model.model.visual`，再做 PyTorch 与 TensorRT 的 correctness / timing 对比。 |

### Transformer / Qwen

| 入口文件 | 入口函数 / 类 | 调用关系 |
| --- | --- | --- |
| `experiments/Phase1-qwen3-baseline/src/hf_bf16_reference.py` | `main` | 使用 `AutoTokenizer` 和 `AutoModelForCausalLM` 建立 BF16/FP16 Hugging Face reference。 |
| `experiments/Phase1-qwen3-baseline/src/hf_bf16_benchmark.py` | `load_runtime`, `run_request`, `main` | `main` 调用 `load_runtime` 载入 tokenizer/model，`run_request` 执行请求级 benchmark 和 stage capture。 |
| `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/qwen3_block.py` | class `RMSNorm`, class `Qwen3LikeDecoderBlock`; `make_block` | 构造可导出的 Qwen3-like decoder block，包含 RMSNorm、attention 和 MLP，用于 ONNX / TensorRT 对比。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_1A/run_fp16_smoke.py` | `main` | Qwen3-VL FP16 loading / forward smoke test。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_1B/run_fp16_baseline.py` | `main` | 加载 processor 与 model，执行 staged timer 和 end-to-end FP16 baseline。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_3A/run_end_to_end_vision_integration.py` | class `TensorRTVisionAdapter`; `main` | 将 PyTorch vision encoder 替换为 TensorRT adapter，验证多模态模型集成边界。 |

### Runtime

| 入口文件 | 入口函数 / 类 | 调用关系 |
| --- | --- | --- |
| `experiments/Phase2-qwen3-quantization/src/phase3a/phase3a_runtime_profile.py` | `bench_runtime`, `profile_workload`, `main` | 按 prefill / decode / embedding / logits 分段测量 FP16 与 mixed runtime。 |
| `experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py` | class `LegacyContext`, class `PersistentContext`; `load_pipeline`, `functional_mode`, `benchmark_mode`, `profile_workload`, `main` | 比较 legacy 与 persistent context 的生命周期、stream ownership 和 profile 行为。 |
| `experiments/Phase2-qwen3-quantization/src/phase3c/phase3c_residual_runtime.py` | `main` | residual runtime profiling 入口。 |
| `experiments/Phase2-qwen3-quantization/src/phase3d0/phase3d0_cuda_graph.py` | class `GraphDecodeWindow`; `capture`, `replay`, `main` | 验证 TensorRT decode window 的 CUDA Graph capture / replay 可行性与正确性。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_3B2/run_decoder_bottleneck_profile.py` | class `TensorRTVisionAdapter`; `main` | 在 Qwen3-VL workload 中拆分 vision / prefill / decode，做 decoder bottleneck attribution。 |

### Benchmark

| 入口文件 | 入口函数 | 调用关系 |
| --- | --- | --- |
| `experiments/Phase1-qwen3-baseline/src/hf_bf16_benchmark.py` | `load_runtime`, `run_request`, `main` | Hugging Face Qwen3 end-to-end / staged baseline。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_2D1/run_vision_latency_benchmark.py` | `main` | 固定 vision workload 的 TensorRT / PyTorch isolated latency benchmark。 |
| `experiments/Phase9-qwen3-vl-migration/src/phase9_1B/run_fp16_baseline.py` | `main` | Qwen3-VL end-to-end FP16 baseline。 |
| `experiments/Exp01-vector-add/src/vector_add.cu`, `experiments/Exp02-reduction/src/reduction.cu`, `experiments/Exp03-matrix-transpose/src/transpose_benchmark.cu`, `experiments/Exp04-gemm/src/gemm_all.cu`, `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/benchmark.cpp` | `main` 系列 | CUDA operator-level correctness 与 benchmark 入口。 |

`benchmark/README.md` 是口径索引；实际 benchmark 代码和 evidence 在 `experiments/` 与 `results/`。
