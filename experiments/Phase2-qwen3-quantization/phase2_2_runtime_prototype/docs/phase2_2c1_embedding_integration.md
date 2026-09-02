# Phase 2.2-C1 — Qwen3 Embedding Integration

## Objective

Validate the real Qwen3-0.6B embedding as an independent TensorRT FP16 component and attempt one controlled handoff into the existing B4.2 28-layer prefill engine. Final RMSNorm, LM head, sampling, generation, quantization, plugins, benchmark and Nsight are out of scope.

## Starting State

Branch `phase/02-qwen3-quantization`, starting HEAD `10eb423` (C0 architecture audit). B4.2 decoder code and engines were not modified. Frozen model revision is `c1899de289a04d12100db370d81485cdf75e47ca`, checkpoint SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

## Real Qwen3 Embedding Audit

The pinned safetensors key is `model.embed_tokens.weight`, module path `model.embed_tokens`. Shape is `[151936, 1024]`, dtype BF16, 155,582,464 elements and 311,164,928 bytes. The config records tied word embeddings; the LM head shares this matrix. BF16 tensor SHA256 is `8f29acf519434862d95613b2b4f6b9d14933a5e4d16baebf8ac0b33b410acfb6`.

## TensorRT Implementation

The independent graph is one native `Gather` over the real weight, exported as opset 17 and converted to FP16. ONNX checker passed with one node/one initializer; serialized ONNX is 311,165,133 bytes. TensorRT 10.3 parsed with zero errors and built a 311,199,084-byte FP16 engine. Input is Int64 `input_ids [B,S]` with dynamic batch/sequence; output is FP16 `hidden_states [B,S,1024]`.

## Numerical Validation

Three deterministic cases were evaluated: short `[1,3]`, B=1,S=8 prefill `[0..7]`, and an alternate eight-token pattern. All outputs were finite, CUDA resident, and shape-correct. Against the FP16 reference, max absolute/relative-L2 errors were zero for all cases. Against BF16 reference, the alternate case had max absolute `1.49e-8`, relative-L2 `6.66e-9`; the other cases were zero. Embedding-only Gate C1.4 therefore passes.

## 28-Layer Integration

The TensorRT embedding output was passed once to the unchanged B4.2 prefill engine (`B=1,S=8`). The decoder output remained finite and shape-correct (`[1,8,1024]`, CUDA FP16), but comparison with an independently loaded portable FP16 28-layer reference failed the predeclared bounded criterion: max absolute `4268.0`, RMSE `56.9263`, relative-L2 `2.00425`, cosine `0.534268` at Layer 27.

This is an integration numerical mismatch, not an embedding-only mismatch. No decoder code, engine, cache logic or B4.2 artifact was changed. The likely next diagnostic is to compare the embedding TensorRT output byte-for-byte with the reference immediately before the decoder, then verify that the decoder reference uses the identical input tensor and stream/engine profile. That diagnostic requires explicit follow-up; no alternative implementation was attempted in C1.

## Memory Lifecycle

The embedding weight was loaded in a separate HF environment and serialized only to Jetson `/tmp`; it was not committed. Validation used the existing B4.2 engines from `/tmp/phase2_2b4_2_20260902T082326Z`. Runtime snapshots recorded minimum `MemAvailable` 3,221,467,136 bytes, maximum RSS approximately 1,439,820 KiB, and no exit 137/OOM. The final snapshot includes the portable reference stack, so it is not a production capacity claim. No all-layer HF model plus duplicate handoff was retained.

## Results

Embedding audit, export, ONNX checker, TensorRT build and embedding-only numerical validation passed. The single embedding→B4.2 decoder handoff produced finite tensors but failed end-to-end hidden-state numerical agreement.

## Gate Decision

| Gate | Result |
| --- | --- |
| C1.1 real weight audit | PASS |
| C1.2 HF reference | PASS |
| C1.3 TensorRT engine | PASS |
| C1.4 embedding numerical agreement | PASS |
| C1.5 B4.2 integration | BLOCKED |
| C1.6 final hidden agreement | BLOCKED |
| Memory lifecycle | PASS / no OOM |

Overall C1: **BLOCKED** by the decoder integration numerical mismatch. C2 must not start.

Evidence: `artifacts/phase2_2c1_20260902T090000Z/`. TensorRT warnings about default stream and Int64 binding remain recorded in `embedding_build.log` and `validation.log`.

## Limitations and Stop

No final RMSNorm, LM head, sampling loop, token generation, benchmark, Nsight, INT8, INT4, TensorRT-LLM, plugin, CUDA custom kernel or performance optimization was run. Phase 1 BF16 reference and B4.2 decoder implementation remain unchanged.
