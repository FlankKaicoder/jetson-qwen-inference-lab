# Phase 2.2-B2.0 — Controlled Runtime Toolchain Environment Bridge

Date: 2026-09-02

## Result

The previous B2 preflight stopped because `phase2-trt-tools` could not import `transformers`. This audit confirms a safe dual-environment route. `phase1-hf` contains the frozen HF stack and shared NVIDIA PyTorch/TensorRT; `phase2-trt-tools` contains ONNX/Polygraphy/TensorRT but lacks Transformers and Safetensors. Both environments pass `pip check` and preserve NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08` from `/usr/local/lib/python3.10/dist-packages/torch` with CUDA 12.6.

## Solution evaluation

- Solution A (one existing environment): rejected; neither environment contains the complete required import set.
- Solution B (controlled extension): not attempted. The selected pip does not support `pip install --dry-run`, so there is no auditable resolver plan. Installing packages without that evidence would violate the task boundary.
- Solution C (dual environment): selected. Future B2 work should run real Qwen3 semantics and Layer 0 extraction in `phase1-hf`, then use an explicit bounded handoff to `phase2-trt-tools` for ONNX/TensorRT operations. The handoff must contain only Layer 0 metadata/tensors, explicit names, shapes, dtype and hashes; no full model duplication.

## Gates

ENV-A existing environment audit: PASS.
ENV-B NVIDIA PyTorch identity: PASS.
ENV-C HF dependencies in one selected environment: BLOCKED; dual-env route available.
ENV-D ONNX/TensorRT capability: PASS in `phase2-trt-tools`.
ENV-E pip check: PASS in both environments.

Overall: `SOLUTION_C_DUAL_ENV_REQUIRED`; `B2_ENVIRONMENT_READY` is not claimed for a single combined environment.

## Stop point

No real checkpoint tensor was loaded. No Layer 0 integration, real ONNX export, TensorRT engine build, benchmark, Nsight, INT8, INT4 or TensorRT-LLM work was performed. Phase 2.2-B2 scientific execution and later phases remain not started. Phase 1 BF16 reference and all existing environments are unchanged.
