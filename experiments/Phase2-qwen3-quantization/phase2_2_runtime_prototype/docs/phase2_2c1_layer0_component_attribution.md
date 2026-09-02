# Phase 2.2-C1F — Qwen3 Layer 0 Component Attribution

Date: 2026-09-02
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `6e48add7acdb40a64f13920efd7bb27b615ec3bb`
Result: **COMPONENT_LOCALIZATION_BLOCKED**

## Objective

Locate the first divergent operator inside Qwen3 Layer 0 using the canonical
`B=1,S=8,H=1024` FP16 embedding input, the existing B4.2 28-layer TensorRT
prefill engine, and the independently staged Layer 0 weights. This was a
diagnostic only. No engine, runtime, weight, benchmark, profiler, or
quantization artifact was modified.

Raw evidence: [phase2_2c1f_layer0_component_20260902T193000Z.json](../artifacts/phase2_2c1f_layer0_component_20260902T193000Z.json).

## Layer0 Structure

The source-faithful portable implementation confirms the actual Qwen3 block:

`input RMSNorm → Q/K/V projections → per-head Q/K RMSNorm → RoPE → GQA
attention → output projection → attention residual → post-attention RMSNorm →
SwiGLU (SiLU(gate) × up → down) → MLP residual`.

Recorded dimensions are Q=`16×128`, K/V=`8×128`, rotary dimension `128`,
RoPE theta `1,000,000`, and GQA repeat/interleave factor `2` (`16` query
heads / `8` KV heads). This is not the ordinary MHA/Llama assumption.

## Input Contract

Both reference and TensorRT paths were intended to use the byte-identical C1
canonical embedding tensor `[1,8,1024]`, dtype FP16, with
`position_ids=[0,1,2,3,4,5,6,7]`. The Layer 0 handoff file was loaded from the
existing B4.1 stream; no new weights were produced.

## RMSNorm Analysis

The B4.2 engine has no `input_rmsnorm` or other normalization output binding.
The portable reference tensor was computed for shape/evidence only. A
reference-vs-TRT metric is therefore **NOT AVAILABLE** and the first operator
cannot be assigned to RMSNorm.

## Attention Analysis

The B4.2 contract exposes no Q, K, V, QK-normalized, RoPE, attention-score,
context, or attention-output tensor. Consequently all such comparisons are
**NOT AVAILABLE** for the 28-layer artifact.

The pre-existing B3 four-layer engine does expose `attention_l0`; running it
with the same canonical input gives the following bounded, partial evidence:

| Comparison | Max abs | Relative L2 | Cosine | Interpretation |
| --- | ---: | ---: | ---: | --- |
| portable Layer 0 attention output vs B3 `attention_l0` | 0.01171875 | 0.00316544 | 0.99999416 | Existing four-layer artifact only; not 28-layer operator attribution |

This result cannot identify whether an earlier Q/K/V, QK-norm, RoPE, or fused
attention operation diverges in B4.2.

## QK Norm Analysis

Q/K normalization is present in the Qwen3 structure, but neither
`q_norm` nor `k_norm` is a B4.2 TensorRT binding. Status: **NOT AVAILABLE**.

## RoPE Analysis

The portable path uses rotary dimension `128`, theta `1,000,000`, and the
recorded positions `0..7`. No post-RoPE Q/K binding is exposed by B4.2. Status:
**NOT AVAILABLE**.

## MLP Analysis

The B4.2 engine exposes no post-attention hidden, post-attention RMSNorm, gate,
up, SiLU, gate×up, down, or MLP-residual tensors. All MLP rows are therefore
**NOT AVAILABLE** for reference-vs-TRT comparison.

## Component Metrics

The diagnostic recorded reference-only shapes for every requested component;
because the B4.2 artifact has no matching bindings, no fabricated numerical
metrics are reported.

| Component group | TRT binding | Metric status |
| --- | --- | --- |
| Input RMSNorm | absent | NOT AVAILABLE |
| Q/K/V projections | absent | NOT AVAILABLE |
| Q/K normalization | absent | NOT AVAILABLE |
| RoPE Q/K | absent | NOT AVAILABLE |
| Attention output | absent | NOT AVAILABLE (B4.2); B3 partial evidence above |
| Post-attention RMSNorm/residual | absent | NOT AVAILABLE |
| SwiGLU MLP intermediates/residual | absent | NOT AVAILABLE |

## First Divergent Operator

**UNKNOWN.** `COMPONENT_LOCALIZATION_BLOCKED` is the evidence-backed gate:
the existing B4.2 engine is too coarse-grained for internal attribution. C1E
still establishes the first non-zero *layer boundary* difference at Layer 0,
but this run cannot promote that to an operator-level claim.

## Root Cause Ranking

Given C1D's byte-identical embedding and identical host/direct-device failure,
transport, pointer, stream, lifetime, and auxiliary-input defects are not
confirmed. The remaining hypotheses, in decreasing diagnostic priority, are:

1. FP16 normalization semantics/accumulation (input or post-attention RMSNorm,
   including fused Reduce/Pow behavior).
2. Fused attention numerical ordering, including Q/K normalization, RoPE,
   softmax, or GQA repeat/interleave.
3. Fused SwiGLU/MLP accumulation or residual ordering.

This ranking is a hypothesis list, not a root-cause finding.

## Next Step

Keep Phase 2.2-C1 **BLOCKED** and do not start C2. A future, explicitly
authorized diagnostic would need a smallest-scope Layer 0 engine exposing
internal tensors (or an equivalent operator-isolated graph) before a first
divergent operator can be named. No repair was attempted in C1F.
