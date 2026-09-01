# Qwen3-0.6B Full Architecture Map

Source: frozen Phase 1 Qwen3-0.6B config at revision `c1899de289a04d12100db370d81485cdf75e47ca`. This is a structural map only; no checkpoint tensor was loaded by Phase 2.1.9.

## Configuration

| Field | Value |
| --- | ---: |
| Hidden size | 1024 |
| Decoder layers | 28 |
| Attention heads / KV heads | 16 / 8 |
| Head dimension | 128 |
| Intermediate size | 3072 |
| Vocabulary size | 151,936 |
| Max positions | 40,960 |
| RoPE theta | 1,000,000 |
| RMSNorm epsilon | 1e-6 |
| Tie word embeddings | true |
| Activation | SiLU / SwiGLU-style gated MLP |

## Dataflow

```text
input_ids [B,S]
  -> token embedding [B,S,1024]
  -> 28 x decoder layer
       input RMSNorm
       Q/K/V projections
       q_norm / k_norm
       RoPE
       grouped-query attention (16 Q heads, 8 KV heads)
       causal softmax
       output projection
       residual add
       post-attention RMSNorm
       gate_proj + SiLU, up_proj, elementwise product, down_proj
       residual add
  -> final RMSNorm
  -> tied LM head [1024,151936]
  -> logits [B,S,151936]
```

For autoregressive decode, each layer consumes the current token query and appends the projected K/V vectors to its cache. The prefill path creates K/V for all input positions; decode uses query length 1 and attends over the cached prefix plus the new token.

## Weight layout per decoder layer

The frozen safetensors header reports Q projection `[2048,1024]`, K/V `[1024,1024]`, O projection `[1024,2048]`, two 128-element Q/K norms, two 1024-element RMSNorm vectors and three MLP matrices (`[3072,1024]`, `[3072,1024]`, `[1024,3072]`). The full serialized header contains 751,632,384 elements. Tied embeddings mean the loaded module has 596,049,920 unique parameter elements while the serialized file contains both embedding and LM-head tensors.

## TensorRT boundaries

The model naturally splits into an embedding input boundary, repeated decoder-layer runtime, final norm and vocabulary projection. A production engine must decide whether to keep the LM head in the same engine, shard or refit it, and how to represent persistent KV-cache buffers between enqueue calls. The synthetic Phase 2.1.8 graph validated arithmetic, attention and MLP operators, but not these runtime boundaries or cache lifetimes.

## Open architecture risks

- Full-model graph size and constant storage are materially larger than the 31-32 MB single-block feasibility engine.
- KV cache is persistent state, not an ordinary stateless tensor; cache indexing and append semantics must be owned by the runtime or a plugin.
- Prefill and decode have different shapes and likely need separate optimization profiles or engines.
- RoPE position handling must remain consistent with cache position offsets.
- FP16 Reduce/Pow normalization warnings from the synthetic graph require an explicit precision policy.
- Tied embedding/LM-head storage may not remain physically shared after ONNX conversion and TensorRT build.
