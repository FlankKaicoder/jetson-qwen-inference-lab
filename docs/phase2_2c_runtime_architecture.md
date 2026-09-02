# Phase 2.2-C0 Runtime Architecture Audit

Status: **C0 architecture audit only**
Branch: `phase/02-qwen3-quantization`
Baseline: Phase 2.2-B4.2, commit `fa16fc0`

This document defines the full Qwen3 runtime boundary. It does not implement or execute C1-C5.

## 1. Phase 2.2-C Objective

Phase 2.2-B4.2 demonstrated a real Qwen3-0.6B decoder stack: layers 0-27 execute with TensorRT FP16, prefill and dynamic decode update independent K/V caches, and the bounded numerical and memory checks pass. That stack accepts hidden states and position/cache tensors; it does not accept text and it does not produce a token.

The distinction is important:

| Runtime level | Responsibility | Current status |
| --- | --- | --- |
| Decoder runtime | Execute decoder layers and maintain per-layer KV state for supplied hidden states | **DONE / bounded** in B4.2 |
| Full generation runtime | Convert text to IDs, embed IDs, run the decoder, normalize/ project logits, sample, and manage the next-token loop | **TODO** in C1-C5 |

C0 therefore answers an architecture question, not an end-to-end quality or latency question: what components must surround the verified decoder before a Qwen3 model can generate tokens?

## 2. Full Qwen3 Runtime Pipeline

```text
Text
  |
Tokenizer
  |
input_ids [B,S]
  |
Embedding (embed_tokens)
  |
hidden states [B,S,H]
  |
28-Layer TensorRT Decoder + persistent KV cache
  |
final hidden states [B,S,H]
  |
Final RMSNorm
  |
LM Head / tied embedding projection
  |
logits [B,S,V]
  |
Sampling policy
  |
next token id
```

Prefill consumes the prompt sequence and creates one K/V history per decoder layer. Decode consumes one new token at a time, reads the existing history, appends one slot per layer, and returns logits for the next sampling decision. The tokenizer and sampling policy are host-side control-plane components; embeddings, decoder execution, normalization, and projection are data-plane components.

## 3. HF Qwen3 Component Mapping

The table records the current mapping without implying that a component is already integrated.

| HF Component | Runtime Status | Backend | Interface / evidence |
| --- | --- | --- | --- |
| `tokenizer` | TODO (C1/C5 dependency) | Hugging Face tokenizer on CPU | Text ↔ `input_ids`; not part of B4.2 |
| `model.embed_tokens` | TODO (C1) | Planned TensorRT FP16 or controlled host/device implementation | `input_ids [B,S]` → hidden `[B,S,1024]` |
| `model.layers.0..27` | DONE / bounded (B4.2) | TensorRT 10.3 FP16 | Real weights; prefill and decode engines; B=1,S=8 and decode 8→12 passed |
| Per-layer KV cache | DONE / bounded (B4.2) | CUDA-resident FP16 buffers owned by runtime | Layout `[B,8,L,128]`; prefix and layer isolation validated |
| `model.norm` (final RMSNorm) | TODO (C2) | Planned TensorRT FP16, with normalization warning review | Decoder output → normalized hidden |
| `lm_head` / tied `embed_tokens` projection | TODO (C3) | Planned TensorRT FP16 or explicit GEMM path | Hidden `[B,S,1024]` → logits `[B,S,151936]` |
| Sampling loop | TODO (C4) | Host control logic with device logits | Greedy first; deterministic token selection |
| End-to-end token agreement | TODO (C5) | HF BF16 reference vs TensorRT runtime | Token IDs plus tensor metrics |

## 4. Current TensorRT Coverage

Completed:

- Real Qwen3-0.6B decoder layers 0 through 27.
- GQA attention, RoPE, Q/K normalization, MLP, prefill, dynamic decode, and persistent per-layer K/V state.
- One 28-layer prefill engine and one 28-layer decode engine, both TensorRT FP16.
- Bounded cache-prefix integrity and selected-layer numerical propagation.

Not completed:

- Tokenizer and text preprocessing.
- Input embedding.
- Final RMSNorm.
- LM head/logit production.
- Sampling and token-generation loop.
- Full text-to-token-to-text agreement.

Consequently, B4.2 must not be described as a complete Qwen3 inference runtime.

## 5. Runtime Memory Ownership

### 5.1 B4.1 lesson

The failed monolithic path retained all 28 CPU layer states, overlapping CUDA copies, reference trees, and handoff payloads. The result was an exit-137 memory failure. B4.1 recovered the path by making layer state lifetime explicit and writing independently hashed layer files. B4.2 then consumed that handoff without retaining the full HF model payload.

The invariant for C1-C5 is therefore:

```text
load or map only the state needed by the active stage
→ execute
→ release temporary references
→ keep only persistent engine, cache, and required activations
```

### 5.2 Ownership model

| Resource | Owner | Lifetime | Notes |
| --- | --- | --- | --- |
| CPU tokenizer state | Runtime host | Process lifetime | Vocabulary/configuration and token buffers |
| CPU weight staging | Loader | Per component/stage | Must not duplicate the complete model unnecessarily |
| TensorRT engine | Runtime engine registry | Session or process lifetime | Prefill/decode engines are immutable after build/load |
| CUDA engine weights/workspace | TensorRT | Engine context lifetime | Accounted separately from KV and activations |
| KV cache | Runtime sequence state | Sequence lifetime | One K/V pair per layer; reset on sequence boundary |
| Hidden activations | Execution context | One operation/step | Reuse buffers where shape contracts allow |
| Logits | Sampling boundary | One sampling decision | Avoid retaining all time-step logits |

For the verified decoder, batch-1 FP16 KV storage is 114,688 bytes per token across 28 layers (`[B,8,L,128]` K and V). This is a layout calculation, not a claim of maximum context capacity. Engine and workspace memory must be measured independently for each runtime configuration.

### 5.3 CUDA synchronization boundary

All enqueue operations and buffer ownership transitions need an explicit stream contract. B4.2 used CUDA-resident direct bindings and synchronization after execution. C1-C5 must preserve that contract, document whether a non-default stream is introduced, and ensure tokenizer/CPU control never observes a buffer before its producing CUDA work completes.

## 6. Phase 2.2-C Implementation Plan

### C1 — Embedding

- **Goal:** Map `input_ids [B,S]` to Qwen3 hidden states `[B,S,1024]` using the exact frozen embedding weights.
- **Input/output:** token IDs and attention/padding metadata in; FP16 hidden states and position IDs out.
- **Validation:** weight hash/mapping audit, finite output, shape/device checks, and per-element/tensor comparison against the HF BF16 reference after an explicitly defined FP16 conversion.
- **Memory constraint:** stream or map embedding weights without constructing an unnecessary second full-model payload.

### C2 — Final RMSNorm

- **Goal:** Integrate `model.norm` after layer 27.
- **Input/output:** final decoder hidden `[B,S,1024]` in; normalized hidden of the same shape out.
- **Validation:** finite values, cosine/max-error/RMSE against HF eager reference, and explicit investigation of TensorRT FP16 Reduce/Pow overflow warnings observed in B4.2.
- **Exit criterion:** normalization behavior is bounded for both prefill and one-token decode.

### C3 — LM Head

- **Goal:** Project normalized hidden states to vocabulary logits, honoring Qwen3 tied embedding weights.
- **Input/output:** `[B,S,1024]` in; `[B,S,151936]` logits out.
- **Validation:** weight tying and shape audit, finite logits, top-k agreement, and numerical metrics before any sampling claim.
- **Memory constraint:** vocabulary projection is large; avoid retaining duplicate transposed/converted matrices unless measured and justified.

### C4 — Greedy Sampling

- **Goal:** Select `argmax(logits[-1])` deterministically and manage sequence position/cache state.
- **Input/output:** one logits row and runtime state in; one next-token ID plus updated position in.
- **Validation:** deterministic repeated calls, EOS handling, sequence reset, and a CPU reference argmax check. Temperature/top-p sampling is out of scope for the first implementation.

### C5 — End-to-End Token Agreement

- **Goal:** Compare the complete text → IDs → decoder → logits → greedy token loop with the frozen HF BF16 reference.
- **Input/output:** bounded prompts in; generated token IDs and diagnostic tensors out.
- **Validation:** fixed prompts, fixed revision, fixed tokenizer settings, token-by-token agreement, and per-stage cosine/max-error/RMSE. Any mismatch must be localized to embedding, decoder, final norm, projection, or sampling rather than hidden by an aggregate score.
- **Exit criterion:** only the predeclared prompt/length matrix may be called PASS; broader quality or capacity claims require separate evidence.

## 7. Validation Protocol

The HF BF16 reference from Phase 1 remains the semantic reference; it is not modified. TensorRT runtime inputs, positions, masks, and cache lengths must be identical at each comparison point.

For tensors, record:

- shape and dtype/device;
- finite status;
- maximum absolute error;
- RMSE;
- relative L2 error;
- cosine similarity.

For generation, record:

- exact tokenizer/model revision;
- prompt and input IDs;
- token-by-token agreement;
- first divergence position;
- logits top-k overlap around the divergence.

The comparison sequence is:

```text
HF embedding ↔ TensorRT embedding
HF layer outputs/KV ↔ B4.2 decoder outputs/KV
HF final norm ↔ TensorRT final norm
HF logits ↔ TensorRT logits
HF greedy IDs ↔ TensorRT greedy IDs
```

Numerical thresholds must be declared before each C-stage run. A cosine or token match alone is insufficient evidence for correctness, and a bounded tensor difference does not prove semantic token agreement.

## 8. Future Extension

Phase 2.3 may investigate quantization after the FP16 runtime boundary is complete. The existing Phase 2.0/2.1 audits leave INT8/INT4 support constrained or inconclusive; no quantization decision is made by C0.

Phase 3 may optimize CUDA operators and runtime scheduling, including RMSNorm, attention, memory movement, CUDA Graphs, or custom TensorRT plugins. Such work requires a stable C5 baseline and its own correctness, benchmark, and profiler evidence.

## C0 Stop Declaration

This audit adds architecture documentation only. No code, model weights, TensorRT engine, benchmark, profiler, quantization backend, or generation loop was modified or executed. C1-C5 remain TODO.
