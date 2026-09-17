# GPT Analysis Manifest

This manifest is the final pre-upload reading list for a GPT code-analysis ZIP. It is intentionally smaller than the full project-state document: it identifies what to read first, what is optional, and what should not be treated as code evidence.

## Repository Overview

The repository is a bounded Transformer inference optimization lab for Qwen3 / Qwen3-VL on Jetson Orin Nano Super. It moves through CUDA fundamentals, Hugging Face baselines, TensorRT conversion and runtime, quantization studies, RMSNorm CUDA/TensorRT-plugin work, and Qwen3-VL vision-encoder migration. The final analytical claim is isolated and bounded: the static Qwen3-VL Vision Encoder TensorRT FP16 path was about `2.7x` faster than the PyTorch FP16 boundary, while end-to-end generation speedup was not claimed because decode dominated the workload.

### Repository Size Report

Audit snapshot: branch `final-analysis`, commit `c227d31f8ff152f9a3593b1a8c71f6423d098761`, taken before adding this manifest. This manifest adds one small Markdown file to the final ZIP.

| Scope | Files | Size |
| --- | ---: | ---: |
| Git-tracked working-tree content | 1,486 | 31.33 MiB |
| Untracked, non-ignored local content | 52 | 0.66 MiB |
| `.git/` object database | 3,785 | 13.02 MiB |

Top-level tracked distribution:

| Directory | Files | Size |
| --- | ---: | ---: |
| `experiments/` | 978 | 21.14 MiB |
| `results/` | 471 | 9.72 MiB |
| `docs/` | 25 | 0.41 MiB |
| Root documents and `.gitignore` | 5 | 0.05 MiB |
| Index README directories (`benchmark/`, `cuda/`, `references/`, `runtime/`, `scripts/`, `tensorrt/`, `transformer/`) | 7 | <0.01 MiB |

Largest tracked files are compact JSON/CSV evidence, not binaries:

| Size | File |
| ---: | --- |
| 1.420 MiB | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3d_20260904T001100Z/policy_p0_all_int8.json` |
| 1.419 MiB | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3d_20260904T001100Z/policy_p1_outlier_guard.json` |
| 1.417 MiB | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3d_20260904T001100Z/policy_p2_family_guard_prevalidated.json` |
| 1.417 MiB | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3d_20260904T001100Z/mixed_precision_policy_primary_fixed.json` |
| 1.392 MiB | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3e_20260904T034300Z/engine_precision_summary.json` |
| 0.964 MiB | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3c_20260903T220600Z/portable_sensitivity_per_target.json` |
| 0.800 MiB | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3d_20260904T001100Z/full_policy_component_validation.json` |
| 0.766 MiB | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3c_20260903T220600Z/portable_sensitivity_per_sample.json` |
| 0.723 MiB | `results/phase3a_runtime_attribution/20260904T063649Z_nsys/stats2/mixed_nvtx_pushpop_trace.csv` |
| 0.668 MiB | `results/phase4a_operator_attribution/20260904T132820Z/onnx_node_inventory.csv` |

### Local-Only Risk Scan

The following large or transient files exist locally but are excluded by `.gitignore`. They must not enter the GPT ZIP.

| Category | Local findings | Size | Status |
| --- | ---: | ---: | --- |
| PyTorch checkpoint / captured tensor payload | 2 `.pt` files | 1,214.188 MiB | ignored by `*.pt` |
| ONNX conversion artifact | 3 `.onnx` files | 90.079 MiB | ignored by `*.onnx` |
| TensorRT engine | 3 `.engine` files | 91.038 MiB | ignored by `*.engine` |
| Python cache | 55 `.pyc` files under `__pycache__/` | 0.898 MiB | ignored by `__pycache__/` and `*.py[cod]` |
| Log files | 217 `.log` files | 2.570 MiB | ignored by `*.log` / `logs/` |
| Build products | no `.o`, `.obj`, `.so`, `.dll`, or `.exe` found in working-tree scan | 0 B | clean |
| Raw Nsight reports | no `.nsys-rep`, `.ncu-rep`, or `.qdrep` found in working-tree scan | 0 B | clean |

Generate the GPT ZIP from Git history, not by copying the whole folder:

```bash
git archive --format=zip -o ../nvidia-qwen-final-analysis.zip HEAD
```

This includes clean tracked content and excludes local model/engine payloads, caches, logs, untracked audit files, and `.git/`.

## Reading Order

Read in this order:

1. `README.md` — project purpose, final bounded result, and repository conventions.
2. `docs/final_analysis_readme.md` — read-oriented repository map and evidence chain.
3. `docs/code_architecture.md` — architecture flow, directory map, and verified entry points.
4. `docs/module_index.md` — compact module-to-file index.
5. `docs/learning_guide.md` — staged CUDA-to-TensorRT reading route.
6. `docs/PROJECT_FINAL_REPORT.md` and `docs/FINAL_STATUS.md` — final gates, bounded claims, and limitations.
7. `docs/experiment_index.md` and `results/experiment_registry.csv` — map experiments to reports and result artifacts.
8. CUDA source in `experiments/Exp01-*` through `experiments/Exp04-*`.
9. Qwen3 baseline and decoder block source under `experiments/Phase1-*` and `experiments/Phase2-*`.
10. TensorRT quantization, runtime, plugin, and vision-integration sources under `experiments/Phase2-*`, `experiments/Phase8-*`, and `experiments/Phase9-*`.
11. Optimization/attribution reports and compact results under `experiments/Phase5-*`, `experiments/Phase6-*`, `results/`, and Phase 3 runtime reports in `docs/`.

Priority classes:

| Priority | Scope |
| --- | --- |
| Must read | Project overview docs, final report/status, experiment index, registry, CUDA fundamentals, Qwen3 baseline/block, TensorRT precision probe, RMSNorm plugin, Qwen3-VL export/build/integration. |
| Recommended | Phase 3 runtime reports, Phase 5/6 feasibility and attribution material, experiment READMEs/scripts, compact CSV/JSON evidence used to verify a specific conclusion. |
| Can skip initially | Top-level index READMEs other than their scope notes, experiment templates, bulk raw CSV/JSON unless validating a claim, local ignored binaries/caches/logs, `.git/`, and untracked local audit reports. |

## Essential Files

| 模块 | 文件 | 作用 | 阅读原因 |
| --- | --- | --- | --- |
| Project map | `README.md` | Project summary and final result pointer. | Establishes scope and the bounded final conclusion. |
| Project map | `docs/final_analysis_readme.md` | Read-oriented repository guide. | Explains the split between index directories and evidence directories. |
| Project map | `docs/code_architecture.md` | Architecture and entry-point map. | Connects model, conversion, runtime, optimization, and benchmark paths. |
| Project map | `docs/learning_guide.md` | Five-stage learning route. | Gives a concrete reading order from CUDA to optimization. |
| Project map | `docs/module_index.md` | Module and key-file index. | Fast navigation when analyzing a specific subsystem. |
| Final evidence | `docs/PROJECT_FINAL_REPORT.md` | Project closeout report. | Separates supported, partially supported, and unknown conclusions. |
| Final evidence | `docs/FINAL_STATUS.md` | Final gate and limitations. | Prevents overclaiming end-to-end speedup. |
| Final evidence | `docs/experiment_index.md` | Experiment-to-report index. | Finds the report behind each experiment. |
| Final evidence | `results/experiment_registry.csv` | Compact experiment registry. | Maps experiment IDs to status and evidence. |
| CUDA | `experiments/Exp01-vector-add/src/vector_add.cu` | Vector-add kernel and benchmark entry. | Minimal launch, timing, and correctness example. |
| CUDA | `experiments/Exp02-reduction/src/reduction.cu` | Reduction V1-V7 implementation and validation. | Shows progressive reduction optimization and tradeoffs. |
| CUDA | `experiments/Exp03-matrix-transpose/src/transpose.cu` | Transpose correctness path. | Demonstrates coalescing, tiling, and padding. |
| CUDA | `experiments/Exp03-matrix-transpose/src/transpose_benchmark.cu` | Transpose calibration/adaptive benchmark. | Shows disciplined warmup and timing methodology. |
| CUDA | `experiments/Exp04-gemm/src/gemm_all.cu` | GEMM/WMMA correctness and benchmark. | Bridges memory-bound learning to compute-bound GEMM. |
| CUDA | `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/benchmark.cpp` | RMSNorm FP16/BF16 benchmark. | Connects CUDA kernel variants to a Transformer operator. |
| Transformer | `experiments/Phase1-qwen3-baseline/src/hf_bf16_reference.py` | Hugging Face Qwen3 reference. | Establishes the model-loading correctness baseline. |
| Transformer | `experiments/Phase1-qwen3-baseline/src/hf_bf16_benchmark.py` | Hugging Face Qwen3 benchmark. | Defines request-level baseline timing. |
| Transformer | `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/qwen3_block.py` | Qwen3-like decoder block. | Makes RMSNorm/attention/MLP exportable for comparison. |
| Transformer | `experiments/Phase9-qwen3-vl-migration/src/phase9_1B/run_fp16_baseline.py` | Qwen3-VL FP16 baseline. | Adds processor, vision model, and staged timing context. |
| Quantization | `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/build_engine.py` | FP16 and explicit Q/DQ INT8 TensorRT probe. | Isolates precision behavior on a minimal network. |
| Quantization | `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/run_engine.py` | TensorRT engine execution probe. | Shows binding allocation and `execute_async_v3` validation. |
| Quantization | `experiments/Phase2-qwen3-quantization/src/phase2_3b/evaluate_calibration.py` | Calibration evaluation entry. | Connects calibration data to mixed-precision policy work. |
| Quantization | `experiments/Phase2-qwen3-quantization/src/phase2_3e/build_mixed_runtime.py` | Mixed-precision runtime builder. | Shows policy-to-engine construction. |
| Quantization | `experiments/Phase2-qwen3-quantization/src/phase2_3f/phase2_3f_compare.py` | Mixed runtime comparison. | Verifies precision-policy effects under bounded conditions. |
| TensorRT | `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/trt_block_parse.py` | Parse and execute decoder block graph. | Bridges exported model structure to TensorRT execution. |
| TensorRT | `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.cpp` | RMSNorm Plugin V3 implementation. | Shows plugin capability interface and `enqueue`. |
| TensorRT | `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin_kernel.cu` | Plugin CUDA kernel. | Connects plugin call path to device execution. |
| TensorRT | `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/qwen3_integration/qwen3_rmsnorm_integration.cpp` | Primitive vs plugin comparison. | Validates plugin integration at a controlled boundary. |
| TensorRT | `experiments/Phase9-qwen3-vl-migration/src/phase9_2B1/export_vision_onnx.py` | Static Qwen3-VL vision ONNX export. | Shows the conversion starting point. |
| TensorRT | `experiments/Phase9-qwen3-vl-migration/src/phase9_2B2/parse_vision_onnx.py` | Vision ONNX graph audit. | Explains parseability and graph assumptions. |
| TensorRT | `experiments/Phase9-qwen3-vl-migration/src/phase9_2C1/build_vision_onnx_fp16.py` | Vision ONNX to TensorRT FP16 build. | Gives the full engine-build path. |
| TensorRT | `experiments/Phase9-qwen3-vl-migration/src/phase9_3A/run_end_to_end_vision_integration.py` | TensorRT vision adapter integration. | Shows PyTorch-to-TensorRT module replacement and comparison. |
| Runtime | `experiments/Phase2-qwen3-quantization/src/phase3a/phase3a_runtime_profile.py` | Prefill/decode stage profiling. | Locates runtime bottlenecks by stage. |
| Runtime | `experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py` | Legacy vs persistent context study. | Clarifies context lifetime and stream ownership. |
| Runtime | `experiments/Phase2-qwen3-quantization/src/phase3d0/phase3d0_cuda_graph.py` | CUDA Graph capture/replay feasibility. | Shows graph constraints around TensorRT execution. |
| Runtime | `experiments/Phase9-qwen3-vl-migration/src/phase9_3B2/run_decoder_bottleneck_profile.py` | Qwen3-VL decoder attribution. | Separates vision, prefill, decode, and kernel-class evidence. |
| Benchmark | `experiments/Phase9-qwen3-vl-migration/src/phase9_2D1/run_vision_latency_benchmark.py` | Isolated vision latency benchmark. | Gives the bounded 2.7x comparison context. |
| Benchmark | `experiments/Phase9-qwen3-vl-migration/src/phase9_3B1/run_end_to_end_stage_breakdown.py` | End-to-end stage breakdown. | Prevents isolated speedups from being generalized. |

## Ignore List

Do not prioritize the following during GPT analysis:

- `.git/`: source-control object database, not project evidence.
- `__pycache__/`, `*.pyc`, `.cache/`, and Hugging Face caches: runtime cache, not source or evidence.
- `*.pt`, `.pth`, `.safetensors`, `.onnx`, `.engine`, `.plan`, and other model/engine payloads: large Jetson-local artifacts excluded by `.gitignore`; the repo reports summarize their use and result.
- `build/`, `CMakeFiles/`, `*.o`, `*.obj`, `*.so`, `*.dll`, `*.exe`: build products, not experimental evidence.
- `logs/`, `*.log`, and temporary files: useful only when reproducing a run; not primary reading material.
- Raw Nsight reports (`*.nsys-rep`, `*.ncu-rep`, `.qdrep`): absent from the tracked snapshot; compact summaries are the portable evidence.
- Top-level `cuda/`, `tensorrt/`, `transformer/`, `runtime/`, `benchmark/`, and `scripts/` beyond their READMEs: these are scope indexes, while implementation lives under `experiments/`.
- Bulk raw CSV/JSON during the first pass: consult them only after a report points to a specific result.
- Untracked local audit files such as `docs/branch_audit_report.md` and root sync/audit Markdown files: not part of the tracked final-analysis ZIP unless explicitly added later.

Do not interpret `UNKNOWN`, `INCONCLUSIVE`, `BLOCKED`, or `REJECT` as missing documentation. They are valid evidence states.
