# Phase 2.2 Qwen3 TensorRT FP16 Runtime Prototype Design

## 1. Problem

A TensorRT engine is an accelerator for a bounded graph invocation, not an autoregressive LLM runtime. Qwen3 generation needs persistent state between invocations: K/V tensors from prompt prefill must be retained, indexed by sequence position, read by each decode step and extended with the new token's K/V. The runtime must also schedule prefill and decode shapes, own GPU memory and turn logits into the next token.

This preparation implements only CPU-side interface and ownership prototypes. It does not load Qwen3, invoke TensorRT, execute CUDA or make an engine.

The precheck and prototype validation were run on Windows and Jetson with ordinary Python. Evidence is in `artifacts/phase2_2_precheck/precheck.json`, `artifacts/kv_memory_estimate.json` and `artifacts/kv_cache_validation.json`.

## 2. Runtime responsibilities

- **KV cache ownership:** allocate per-layer K and V memory once per sequence, expose bindings to future engines and reuse capacity across requests.
- **Sequence management:** distinguish the logical sequence ID, batch slot, token position and per-layer materialization progress.
- **Stream synchronization:** a future implementation must order cache writes before a later decode reads them, without assuming the default stream is safe or efficient.
- **Sampling:** convert final logits to a next token under explicit greedy/sampling policy outside the decoder engine.
- **Memory management:** reserve cache capacity, avoid stale sequence visibility, release or reuse allocation at request completion and account for weights, workspace and activations separately from cache bytes.

## 3. Cache layout and ownership

The prototype owns CPU bytearrays per decoder layer:

```text
K[layer] : [batch, kv_head, capacity_tokens, head_dim]
V[layer] : [batch, kv_head, capacity_tokens, head_dim]
```

The bytearrays are explicit manager-owned storage. `append_kv()` copies a contiguous incoming `[B,H_kv,T,D]` span into the capacity-strided allocation. A layer's length advances only for a contiguous append; the manager's visible `sequence_length` is the shortest length among layers, preventing one partially written layer from advancing a decode position.

`reset_sequence()` invalidates logical contents but retains allocation. It does not erase bytes; a production security policy can choose erasure separately.

## 4. Prefill path

```text
Prompt input_ids [B,S]
  -> embedding
  -> decoder layers
  -> per-layer K/V creation for positions [0,S)
  -> final norm and LM head
  -> logits [B,S,vocab]
```

The future runtime allocates a cache before prefill, validates that `S <= capacity`, runs the prefill execution contract, then appends each layer's K/V span at position zero. It must not declare the prompt ready until all layers expose the same prefix length.

## 5. Decode path

```text
New input_ids [B,1]
  -> decoder attention reads K/V cache prefix
  -> decoder creates K/V for current position
  -> runtime appends new K/V to every layer cache
  -> final logits [B,1,vocab]
  -> sampling selects next token
```

Decode position is the current fully materialized cache length. The runtime supplies `position_ids [B,1]`, cache offsets and causal semantics consistently with RoPE. Position bookkeeping must not be inferred from token IDs.

## 6. Future TensorRT boundary

```text
Runtime (request lifecycle, cache addresses, positions, stream, sampling)
  -> TensorRT engine(s) (prefill/decode graph execution)
  -> CUDA (device allocations and kernels)
```

A likely design has separate optimization profiles, and possibly separate engines, for prefill `[B,S]` and decode `[B,1]`. The runtime would bind model inputs, K/V input/output buffers and logits buffers for each enqueue. It must define whether cache is passed as explicit tensors, persistent device buffers or plugin-owned state before any full-model export is authorized.

## Non-goals

This preparation does not validate TensorRT bindings, engine shapes, cache numerics, attention correctness, latency, throughput or memory capacity. The Phase 1 BF16 reference remains the correctness oracle for later authorized execution work.
