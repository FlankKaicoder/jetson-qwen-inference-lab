# Phase 2.1 TensorRT INT8 / INT4 Capability Audit

Date: 2026-09-01. This is a synthetic Linear capability audit only. No Qwen3 weights were loaded, exported, quantized, or benchmarked. Phase 1 BF16 evidence is unchanged.

## Environment

Jetson Orin Nano Super (SM 8.7), CUDA 12.6, NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`, TensorRT Python `10.3.0`, and `/usr/src/tensorrt/bin/trtexec` banner `TensorRT v100300` were verified in the frozen `/home/nvidia/.venvs/jetson-qwen-phase2-quant` environment. The TensorRT runtime libraries are present. No package or system component was changed.

The venv has no `onnx`, `polygraphy`, or `onnx_graphsurgeon` module. `build_linear_onnx.py` therefore records `BLOCKED_NO_ONNX_PACKAGE`; no ONNX file was fabricated and no ONNX parser conclusion is claimed.

## Probe and evidence

The probe uses direct TensorRT Python-network construction as a clearly-labeled fallback for the unavailable ONNX route. It creates deterministic synthetic FP16 Linear layers `1024 -> 1024/2048/3072` with dynamic M profile `[1,32]`. Engines are Jetson-local under `/tmp/phase2_1_trt_engines/20260901T/` and are not Git artifacts.

FP16 engines built successfully for all three output sizes. `run_engine.py` executed M=1 and M=32 for all six cases with `execute_async_v3=true`, CUDA input/output, finite output, and FP32 binding-aware output allocation. `trtexec --dumpLayerInfo --dumpProfile` passed for all three engines and reported Compute Capability 8.7, TensorRT 10.3.0, a `linear_matmul` layer, and GPU profile totals.

The explicit Q/DQ candidate inserts TensorRT `add_quantize`/`add_dequantize` around activation and deterministic weight tensors, with INT8 builder flag. All three engines built and all six M/output combinations executed on `cuda:0` with finite output. Against the deterministic FP16 PyTorch reference, observed max absolute error was approximately `0.6362-0.8789`, mean absolute error `0.1383-0.1489`; relative error is large near zero and is not used as a quality claim. These are numerical probe values, not a performance benchmark. `trtexec` layer profiles passed, but layer names/profile output alone do not prove a particular INT8 GEMM kernel or exclude every internal reformat.

## INT4 investigation

`trtexec --help` exposes `--int4`; Python exposes `DataType.INT4`, `BuilderFlag.INT4`, and a module-level `int4` symbol. No public TensorRT 10.3 API for constructing a packed weight-only INT4 Linear was identified in the frozen environment. No INT4 engine was forced from an FP16 graph, so INT4 weight-only build/GPU execution is `UNKNOWN`.

## Decision matrix

| Path | Build | GPU execute | Correctness | Status |
| --- | --- | --- | --- | --- |
| FP16 direct TensorRT Linear | PASS (3/3) | PASS (6/6) | finite and reference error recorded | SUPPORTED for synthetic primitive |
| INT8 explicit Q/DQ Linear | PASS (3/3) | PASS (6/6) | finite; error recorded; no formal tolerance gate | PARTIALLY SUPPORTED |
| INT4 weight-only Linear | no public construction path identified | NOT RUN | UNKNOWN | PARTIALLY SUPPORTED / BLOCKED for this audit |
| Requested ONNX parser route | blocked by missing `onnx` | NOT RUN | UNKNOWN | BLOCKED_NO_ONNX_PACKAGE |

## Gates

- Gate A Environment: `PASS` (CUDA/TensorRT usable; frozen Python environment preserved).
- Gate B FP16 TensorRT: `PARTIAL PASS` for the direct-network fallback; requested ONNX parse is `BLOCKED_NO_ONNX_PACKAGE`.
- Gate C INT8: `PARTIAL PASS` (explicit Q/DQ build and CUDA execution are demonstrated; numerical error is recorded, but no formal accuracy threshold or true Qwen Linear kernel claim is made).
- Gate D INT4: `PARTIALLY_SUPPORTED / BLOCKED` (surface flags/types exist; weight-only packed construction and execution were not identified).
- Overall Phase 2.1: `INCONCLUSIVE` for selecting a production Qwen backend.

## Recommendation and stop point

The next authorized investigation should prioritize TensorRT INT8 only after an ONNX-capable isolated environment or an explicitly approved equivalent graph path and a defined accuracy threshold. TensorRT INT4 weight-only remains blocked pending a documented supported packing/API path. This audit makes no speed, memory, or power claim.

No Qwen3 quantization was performed. No TensorRT-LLM was built. No Qwen engine was created. No Phase 2.2 or Phase 3 was started. The BF16 reference remains frozen.
