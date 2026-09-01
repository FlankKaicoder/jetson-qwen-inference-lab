# Phase 1 - Qwen3 Baseline Deployment

Status: `IN PROGRESS`

Phase 1 starts with an audit-first runtime decision. Phase 1.0 does not install software, download model weights, build an engine, or run inference.

## Phase 1.0 result

- Gate P1.0: `PASS WITH CONSTRAINTS`.
- Recommended Phase 1.1 path: Hugging Face Transformers on the existing NVIDIA Jetson PyTorch/CUDA stack, initially using Qwen3-0.6B BF16 at a bounded short context.
- Latest TensorRT-LLM has native Qwen3 support, but its current official hardware/software matrix does not provide a supported Jetson Orin SM87 intersection with this JetPack 6.2 environment.
- TensorRT-LLM `v0.12.0-jetson` targets Jetson Orin/CUDA 12.6 but accepts only Qwen, Qwen2 and Qwen2-MoE model types; Qwen3 would require a source backport and is not the baseline path.
- Plain TensorRT/ONNX is not a turnkey decoder runtime because autoregressive control, dynamic sequence handling, KV cache and sampling remain application/runtime responsibilities.

Evidence and full rationale are in [docs/phase1_0_runtime_feasibility_audit.md](docs/phase1_0_runtime_feasibility_audit.md). Phase 1.1 has not started.
