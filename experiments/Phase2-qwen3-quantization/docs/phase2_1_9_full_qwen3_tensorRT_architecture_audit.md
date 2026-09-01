# Phase 2.1.9 Full Qwen3 TensorRT Architecture Audit

Date: 2026-09-02

## Scope and environment

This is a read-only architecture audit. It uses the frozen Qwen3-0.6B configuration and Phase 1 artifacts. It does not load checkpoint tensors, export full Qwen3 ONNX, build a Qwen3 engine, install packages, run inference, benchmark, calibrate INT8, use INT4, or build TensorRT-LLM.

The existing Jetson tools environment passed import and dependency checks: NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`, CUDA `12.6`, TensorRT `10.3.0`, ONNX `1.22.0`, Nsight Compute `2024.3.1.0`, GPU `Orin`, Python `3.10.12`, `pip check` clean.

## Model configuration and architecture

The exact model is `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`. Config evidence is copied to `artifacts/phase2_1_9_20260901/qwen3_full_config.json`. The model has 28 decoder layers, hidden size 1024, 16 Q heads, 8 KV heads, head dimension 128, intermediate size 3072, vocabulary 151,936, max positions 40,960, RoPE theta 1,000,000, RMSNorm epsilon 1e-6, SiLU and tied word embeddings.

Each layer contains input RMSNorm, Q/K/V projections, Q/K normalization, RoPE, grouped-query causal attention, output projection and residual, followed by post-attention RMSNorm, SwiGLU-style MLP and residual. The full map is in `docs/qwen3_full_architecture.md`.

## Weight memory estimate

The frozen safetensors manifest reports 751,632,384 serialized elements and 1,503,300,328 bytes. The conservative serialized lower bounds are 2.800 GiB FP32 and 1.400 GiB FP16/BF16. With `tie_word_embeddings=true`, the loaded unique-parameter lower bound is 1,192,099,840 bytes (1.110 GiB) at two bytes per element. These are theoretical weight bounds, not TensorRT engine or runtime measurements. Detailed arithmetic is in `artifacts/phase2_1_9_20260901/weight_memory_estimate.json`.

## KV cache

For batch 1, FP16/BF16 cache storage is:

```text
2(K,V) * 28 layers * 8 KV heads * 128 head_dim * 2 bytes
= 114,688 bytes = 112 KiB per token
```

That is 112 MiB at 1,024 tokens, 448 MiB at 4,096, 896 MiB at 8,192, 3.5 GiB at 32,768 and 4.375 GiB at 40,960. Batch size multiplies the result. These values exclude metadata, padding, activations and workspace; they are not a capacity claim. Full values are in `artifacts/phase2_1_9_20260901/kv_cache_estimate.json`.

## Dynamic shapes and runtime contract

Prefill consumes `input_ids [B,S]`, `attention_mask [B,S]` and `position_ids [B,S]`; it produces logits and K/V for all input positions. Decode consumes `input_ids [B,1]`, `position_ids [B,1]` and persistent per-layer K/V cache state, then appends one K/V position and produces `[B,1,vocab_size]` logits. A practical design needs separate prefill and decode optimization profiles, for example batch 1-2 with prefill S 1-4096 and decode S 1 plus cache lengths 0-40959. This is a design proposal, not a tested profile.

The runtime must own cache allocation, sequence offsets, causal masking, stream synchronization, logits post-processing and sampling. A stateless ONNX graph alone does not provide these semantics.

## Operator support mapping

| Operator / feature | Phase 2.1.8 evidence | Full-model concern |
| --- | --- | --- |
| RMSNorm arithmetic | Parsed and executed synthetic | FP16 Reduce/Pow overflow warning; precision policy needed |
| MatMul / projections | Parsed and executed synthetic | 28-layer constant size and tactic/workspace growth |
| RoPE Sin/Cos | Parsed and executed synthetic | Position offsets with cache and long context |
| GQA / KV repetition | Parsed and executed synthetic | Cache layout should avoid materialized repetition where possible |
| Softmax / causal mask | Parsed and executed synthetic | Dynamic decode mask and numerical stability |
| KV cache | Not tested | Persistent state, append, reuse and memory pressure |
| Dynamic decode | Not tested | Separate profile/engine and runtime scheduling |
| Embedding / LM head | Not tested in TensorRT | Tied-weight preservation and large vocabulary projection |

## Gates and decision

- Gate A - environment: `PASS`.
- Gate B - architecture mapping: `PASS`.
- Gate C - memory/KV analysis: `PASS` as theoretical planning evidence; no runtime capacity claim.
- Gate D - production route decision: `BLOCKED_NEEDS_RUNTIME_WORK`.

The recommended direction is a separately authorized native TensorRT FP16 engine design with explicit prefill/decode and KV-cache contracts, while retaining the Phase 1 HF BF16 path as the correctness oracle. The synthetic block result supports parser/operator feasibility only; it does not make full Qwen3 deployment ready.

## Stop condition

No full Qwen3 export.

No TensorRT engine.

No INT8.

No INT4.

No TensorRT-LLM build.

No benchmark.

No Phase 2.2.

No Phase 3.

BF16 reference unchanged.
