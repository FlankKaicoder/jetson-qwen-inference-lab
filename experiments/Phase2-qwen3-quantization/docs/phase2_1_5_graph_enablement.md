# Phase 2.1.5 TensorRT Graph Pipeline Enablement

Date: 2026-09-01. This is a bounded synthetic graph-pipeline audit. It validates `PyTorch -> ONNX -> TensorRT parser -> engine -> CUDA execution` on Jetson; it does not export or quantize Qwen3 and it contains no performance, memory, power, INT8/INT4, TensorRT-LLM, or production-backend claim.

## Scope and environment

The isolated tool environment is `/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools` with `--system-site-packages`. It uses the existing NVIDIA PyTorch installation and does not modify the Phase 1 HF or Phase 2 quantization venvs. The only compatibility adjustments were local to this venv: NumPy `1.26.4`, `typing_extensions 4.15.0`, and `ml_dtypes 0.5.4`; `pip check` passes.

Recorded environment: Jetson Orin, Python 3.10.12, NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08` from `/usr/local/lib/python3.10/dist-packages/torch`, CUDA `12.6`, TensorRT Python `10.3.0`, and Nsight Compute `2024.3.1.0`. Full evidence is in `artifacts/phase2_1_5_20260901/environment.txt`.

## Synthetic graphs

Three small FP16 modules were exported with PyTorch opset 17 and a dynamic first dimension with profile `[1, 32]`:

| Graph | Structure | ONNX nodes |
| --- | --- | ---: |
| `linear` | `Linear(1024,3072)` | 1 `Gemm` |
| `mlp` | `Linear(1024,3072) -> GELU(tanh) -> Linear(3072,1024)` | 15 |
| `rmsnorm_linear` | `RMSNorm-like arithmetic -> Linear(1024,3072)` | 10 |

The ONNX files are generated on Jetson under `/tmp/phase2_1_5_graph_pipeline/artifacts_20260901/` and are intentionally not tracked (`*.onnx` is ignored). `onnx.checker.check_model` passed for all three. Export and graph metadata are retained in `artifacts/phase2_1_5_20260901/linear_export.txt`, `mlp_export.txt`, `rmsnorm_linear_export.txt`, and `onnx_check.txt`.

## TensorRT parse and CUDA execution

Each graph was parsed with `trt.OnnxParser`, built with an FP16 TensorRT builder configuration, and executed with `execute_async_v3` for `M=1` and `M=32`. The corrected harness allocates the output using the context-resolved shape and TensorRT-declared dtype (`DataType.HALF`); it does not use a mismatched FP32 pointer. Engines remain Jetson-local under `/tmp` and are not Git artifacts.

The six-case result is summarized in `artifacts/phase2_1_5_20260901/graph_pipeline_summary.csv`. All six parser/build/execute cases passed and all outputs were finite. The RMSNorm graph emitted TensorRT's generic warning that FP16 Reduce/Pow may overflow; this is recorded as a graph-precision risk and was not converted into an accuracy or performance conclusion. TensorRT also warned that the default CUDA stream may add synchronization overhead; no timing was collected.

## Gates

- Gate A — environment and toolchain: `PASS` (`pip check`, imports, CUDA availability, TensorRT and NCU versions recorded).
- Gate B — graph export/checker: `PASS` (3/3 exports and 3/3 ONNX checker validations).
- Gate C — TensorRT parser/build/CUDA execution: `PASS` (6/6 M/profile cases; finite, correctly typed outputs).
- Overall Phase 2.1.5: `PASS` for the bounded synthetic graph-pipeline enablement objective.

This PASS does not change Phase 2.0 or Phase 2.1 status and does not establish Qwen3 exportability, quantization support, kernel selection, latency, memory, power, or production readiness. Phase 2.2 and Phase 3 remain unstarted.

## Reproducibility and stop point

Scripts are under `src/phase2_1_5_graph_pipeline/`. Run them only in the isolated tool venv and keep generated ONNX/engine files outside Git. No benchmark or Nsight Compute collection was performed. The next action requires explicit authorization; stop here.
