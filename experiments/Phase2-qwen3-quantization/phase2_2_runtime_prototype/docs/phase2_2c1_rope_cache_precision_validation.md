# Phase 2.2-C1K - Qwen3 RoPE Cache Precision Validation

Date: 2026-09-03
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `4967feb895a5d2675db2571dadedcca0a8d32ff9`

## Objective

Validate one isolated hypothesis from C1J: TensorRT's native RoPE path retains
FP32 cache values while the portable reference casts `cos/sin` to FP16 before
the multiply/add. This diagnostic changes only that cache boundary. The B4.2
28-layer engine is loaded read-only.

## Layer 0 Structure

Input RMSNorm -> Q/K/V projection -> per-head Q/K RMSNorm -> Qwen3 RoPE ->
8 KV heads repeated to 16 Q heads (GQA) -> causal attention -> output
projection/residual -> post-attention RMSNorm -> SwiGLU MLP.

## Input Contract

Canonical embedding hidden input is `B=1,S=8,H=1024`, FP16, with position IDs
`0..7`. Layer 0 uses Q/K heads `16/8`, head dimension and rotary dimension
`128`, theta `1,000,000`, and half-split `rotate_half`.

## RMSNorm, Attention, QK Norm and RoPE Analysis

RMSNorm, projections, Q/K normalization, attention, softmax, residual and MLP
are unchanged. The baseline and corrected graphs both export and build with
TensorRT 10.3. The corrected graph uses fixed FP16 `cos/sin` initializers for
positions `0..7`; it has only a `hidden_states` input, making the fixed-position
contract explicit. Micro engines retain the dynamic `position_ids` input only
for the baseline comparison.

| Component | Baseline TRT rel-L2 vs portable FP16 | Corrected TRT rel-L2 | Cosine corrected |
| --- | ---: | ---: | ---: |
| Q RoPE output | `0.0927107111` | `0.0001572046` | `0.9999992847` |
| K RoPE output | `0.0144213885` | `0.0000268356` | `1.0000004768` |
| Layer 0 final hidden | `0.0347152092` | `0.0038153231` | `0.9999918938` |

The Layer 0 corrected output is relative-L2 `0.0018616138` against the
unchanged B4.2 output, while the B4.2 portable-reference control is
`0.0037876803`. The final Layer 0 error is reduced by `0.0308998860`, a
`9.0989x` reduction.

## First Divergent Operator

`Q_ROPE` remains the first divergent operator observed in C1J. C1K validates
that the cache precision/cast boundary is sufficient to remove nearly all of
the isolated Q/K RoPE mismatch and most of the Layer 0 final mismatch. The
remaining `0.0038153231` is not attributed further in this experiment.

## Root Cause Ranking

1. **Validated major cause:** FP32-vs-FP16 RoPE cache precision/order at the
   TensorRT graph boundary.
2. **Residual unknown:** TensorRT fusion/tactic arithmetic and downstream
   FP16 normalization effects.
3. **Rejected for this mismatch:** position indexing, half-split layout,
   GQA repeat/interleave and softmax precision, based on C1H/C1J controls.

## Memory and Safety

The run completed without OOM or exit 137. Minimum recorded `MemAvailable` was
`2,551,443,456` bytes. Existing B4.2 engine/runtime, C1 embedding and all
historical artifacts were not modified or overwritten. New ONNX/engine files
remain Jetson-local under `/tmp/phase2_2c1k_20260902T163005Z/`; the committed
raw evidence is
`artifacts/c1k_20260902T163005Z/c1k_rope_cache_precision_20260902T163005Z.json`.

## Result

**C1K result: `ROPE_CACHE_FIX_VALIDATED`**
**C1 overall status: `BLOCKED`**. This diagnostic does not authorize repair,
C1L, C2, benchmark, Nsight, quantization or a production 28-layer rebuild.

## Next Step

Stop and await explicit owner direction. No automatic fix or engine replacement
was performed.
