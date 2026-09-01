# Phase 2.0 - Qwen3 Quantization Backend Feasibility

Status: `RUNNING`

This experiment audits quantization backend feasibility on Jetson Orin Nano Super without quantizing Qwen3 or running a formal quantized benchmark. The Phase 1 BF16 reference remains frozen.

## Scope

- Isolated venv: `/home/nvidia/.venvs/jetson-qwen-phase2-quant`
- NVIDIA PyTorch 2.5.0a0+872d972e41.nv24.08 and CUDA 12.6 are preserved.
- TorchAO `0.12.0` is the only installed quantization package.
- bitsandbytes and TensorRT are survey-only in this phase.

## Current checkpoint

TorchAO wheel installation succeeded with `--no-deps`, but import failed before API discovery because the NVIDIA PyTorch build does not provide `torch._C._distributed_c10d`, required by TorchAO's eagerly imported Float8 modules. No Torch/PyTorch component was replaced. The INT8/INT4 CUDA micro-probes are therefore `BLOCKED_BY_IMPORT_GATE` rather than claimed as supported.

See `docs/phase2_0_quantization_backend_audit.md` and timestamped artifacts for evidence.
