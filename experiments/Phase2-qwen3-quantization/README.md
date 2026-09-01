# Phase 2 Quantization Backend Feasibility and TensorRT Capability Audit

Phase 2.0 status: `BLOCKED`; Phase 2.1 status: `INCONCLUSIVE`

This experiment audits quantization backend feasibility on Jetson Orin Nano Super without quantizing Qwen3 or running a formal quantized benchmark. The Phase 1 BF16 reference remains frozen.

## Scope

- Isolated venv: `/home/nvidia/.venvs/jetson-qwen-phase2-quant`
- NVIDIA PyTorch 2.5.0a0+872d972e41.nv24.08 and CUDA 12.6 are preserved.
- TorchAO `0.12.0` is the only installed quantization package.
- bitsandbytes and TensorRT are survey-only in this phase.

## Current checkpoint

TorchAO wheel installation succeeded with `--no-deps`, but import failed before API discovery because the NVIDIA PyTorch build does not provide `torch._C._distributed_c10d`, required by TorchAO's eagerly imported Float8 modules. No Torch/PyTorch component was replaced. The INT8/INT4 CUDA micro-probes are therefore `BLOCKED_BY_IMPORT_GATE` rather than claimed as supported.

See `docs/phase2_0_quantization_backend_audit.md` and timestamped artifacts for evidence.

## Phase 2.1 TensorRT capability audit

The authorized follow-on audit is documented in `docs/phase2_1_tensorrt_capability_audit.md`. The frozen Jetson environment exposes TensorRT 10.3 FP16 and explicit Q/DQ INT8 construction and synthetic CUDA execution. The ONNX route is blocked because `onnx` is absent and package installation is forbidden. INT4 flags/types are visible, but no public packed weight-only construction path was identified. No Qwen3 quantization or formal performance benchmark was run.

## Phase 2.1.5 TensorRT graph pipeline enablement

Phase 2.1.5 is a bounded synthetic follow-up and is `PASS`. An isolated tools venv at `/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools` supplies ONNX 1.22, GraphSurgeon, Polygraphy and the existing NVIDIA Torch/TensorRT stack. Three FP16 opset-17 graphs (Linear, MLP/GELU, and RMSNorm-like arithmetic plus Linear) passed PyTorch export, `onnx.checker`, TensorRT parsing/building, and CUDA execution for dynamic `M=1` and `M=32` (6/6 cases, finite correctly typed outputs).

This validates only the synthetic `PyTorch -> ONNX -> TensorRT -> CUDA` plumbing. It does not claim Qwen3 exportability, INT8/INT4 support, performance, memory, power, or production readiness. Evidence and the stop-point rationale are in `docs/phase2_1_5_graph_enablement.md` and `artifacts/phase2_1_5_20260901/`. Phase 2.2 and Phase 3 remain unstarted.

## Phase 2.1.8 Qwen3-like decoder block feasibility

A single Qwen3-like block with the recorded Qwen3-0.6B dimensions and synthetic FP16 weights passed opset-17 ONNX checking, TensorRT 10.3 parsing/building and dynamic batch-1/2 CUDA execution. This is a graph feasibility result only. It makes no full checkpoint/export, quantization, benchmark, TensorRT-LLM, Phase 2.2 or Phase 3 claim. The report is `docs/phase2_1_8_qwen3_block_feasibility.md`; FP16 normalization/default-stream warnings remain explicit limitations.
