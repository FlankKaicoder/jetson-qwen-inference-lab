# Phase 1 - Qwen3 Baseline Deployment

Status: `IN PROGRESS`

Phase 1 starts with an audit-first runtime decision. Phase 1.1 is complete; Phase 1 remains in progress and Phase 1.2 has not started.

## Phase 1.0 result

- Gate P1.0: `PASS WITH CONSTRAINTS`.
- Recommended Phase 1.1 path: Hugging Face Transformers on the existing NVIDIA Jetson PyTorch/CUDA stack, initially using Qwen3-0.6B BF16 at a bounded short context.
- Latest TensorRT-LLM has native Qwen3 support, but its current official hardware/software matrix does not provide a supported Jetson Orin SM87 intersection with this JetPack 6.2 environment.
- TensorRT-LLM `v0.12.0-jetson` targets Jetson Orin/CUDA 12.6 but accepts only Qwen, Qwen2 and Qwen2-MoE model types; Qwen3 would require a source backport and is not the baseline path.
- Plain TensorRT/ONNX is not a turnkey decoder runtime because autoregressive control, dynamic sequence handling, KV cache and sampling remain application/runtime responsibilities.

Evidence and full rationale are in [docs/phase1_0_runtime_feasibility_audit.md](docs/phase1_0_runtime_feasibility_audit.md).

## Phase 1.1 result

- Formal environment: `/home/nvidia/.venvs/jetson-qwen-phase1-hf` with `system-site-packages=true`.
- NVIDIA PyTorch was reused unchanged: `2.5.0a0+872d972e41.nv24.08`, CUDA 12.6, BF16 available on SM87.
- Transformers `4.57.3`, Accelerate `1.14.0`, Hugging Face Hub `0.36.2`, Safetensors `0.8.0`, Tokenizers `0.22.2`, Regex `2026.9.3`, and hf-xet `1.6.0` were installed with wheel-only `--no-deps` operations after ordinary resolution attempted PyPI Torch.
- `pip check` passes. Dependency evidence is in `artifacts/phase1_1_*`.
- The earlier `/home/nvidia/.venvs/jetson-qwen-phase1` remains preserved and untouched as `FAILED_PARTIAL_ENV`.
- Exact model: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`; weight SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Qwen3 loaded in BF16 with all parameters on `cuda:0`; finite forward and all three bounded deterministic generations passed semantic sanity.
- Minimum checkpoint MemAvailable was 2,746,023,936 bytes; successful-run swap grew by 68,157,440 bytes and then remained stable; no OOM or offload occurred.
- PyTorch 2.5 requires the built-in eager attention path because its SDPA API does not accept Transformers' `enable_gqa` argument. This is recorded as a functional compatibility constraint, not a performance conclusion.
- Gate A/B/C/D: `PASS`. Phase 1.1: `PASS / CLOSED`. Full report: [docs/phase1_1_hf_bf16_reference.md](docs/phase1_1_hf_bf16_reference.md).
