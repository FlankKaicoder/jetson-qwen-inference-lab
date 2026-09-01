# Phase 1 - Qwen3 Baseline Deployment

Status: `IN PROGRESS`

Phase 1 starts with an audit-first runtime decision. Phase 1.1 is now in progress: its isolated Hugging Face dependency environment is established, while model acquisition and inference remain pending.

## Phase 1.0 result

- Gate P1.0: `PASS WITH CONSTRAINTS`.
- Recommended Phase 1.1 path: Hugging Face Transformers on the existing NVIDIA Jetson PyTorch/CUDA stack, initially using Qwen3-0.6B BF16 at a bounded short context.
- Latest TensorRT-LLM has native Qwen3 support, but its current official hardware/software matrix does not provide a supported Jetson Orin SM87 intersection with this JetPack 6.2 environment.
- TensorRT-LLM `v0.12.0-jetson` targets Jetson Orin/CUDA 12.6 but accepts only Qwen, Qwen2 and Qwen2-MoE model types; Qwen3 would require a source backport and is not the baseline path.
- Plain TensorRT/ONNX is not a turnkey decoder runtime because autoregressive control, dynamic sequence handling, KV cache and sampling remain application/runtime responsibilities.

Evidence and full rationale are in [docs/phase1_0_runtime_feasibility_audit.md](docs/phase1_0_runtime_feasibility_audit.md).

## Phase 1.1 dependency checkpoint

- Formal environment: `/home/nvidia/.venvs/jetson-qwen-phase1-hf` with `system-site-packages=true`.
- NVIDIA PyTorch was reused unchanged: `2.5.0a0+872d972e41.nv24.08`, CUDA 12.6, BF16 available on SM87.
- Transformers `4.57.3`, Accelerate `1.14.0`, Hugging Face Hub `0.36.2`, Safetensors `0.8.0`, Tokenizers `0.22.2`, Regex `2026.9.3`, and hf-xet `1.6.0` were installed with wheel-only `--no-deps` operations after ordinary resolution attempted PyPI Torch.
- `pip check` passes. Dependency evidence is in `artifacts/phase1_1_*`.
- The earlier `/home/nvidia/.venvs/jetson-qwen-phase1` remains preserved and untouched as `FAILED_PARTIAL_ENV`.
- Gate A: `PASS`. Gates B/C/D: `NOT STARTED`. No model was downloaded and no inference or benchmark was run.
