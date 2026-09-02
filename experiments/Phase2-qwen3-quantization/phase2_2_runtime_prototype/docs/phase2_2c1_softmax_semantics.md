# Phase 2.2-C1H - Attention Softmax Numerical Semantics

Date: 2026-09-02
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `64d8d5eefb832a51edf22e119001d0cfb709010a`

## Objective
Determine whether Layer 0 attention softmax is the first material source of the C1 decoder mismatch. This was diagnostic-only; C1 remains blocked.

## Starting State
The unchanged B4.2 prefill engine, canonical C1 FP16 embedding reference and Layer 0 handoff were reused read-only. No B4.2 runtime or engine was modified, and no 28-layer rebuild, benchmark, profiler, repair or C2 work was run.

## C1G Evidence
C1G found QK-score relative-L2 `1.9496e-7` followed by softmax relative-L2 `0.003432333`, identifying softmax as the first material amplification but not confirming causality.

## Exact Attention Pipeline
Layer 0 is Qwen3-specific: input RMSNorm, Q/K/V projections, per-head Q/K normalization, RoPE (rotary dimension 128, theta 1,000,000), 16-query/8-KV GQA with repeat 2, QK matmul, scale `128**-0.5`, causal mask, softmax, context, output projection and residual, followed by post-attention RMSNorm and SwiGLU.

## Exact Pre-Softmax Input
The fresh probe exposed raw QK, scaled QK and final masked `attention_scores`. All tensors were `[1,16,8,8]`, FP16 and finite. The final masked tensor had min `-65504.0` (FP16 sentinel), max `42.8125`, and no infinities. Its relative-L2 against the portable reference was `1.9496e-7`.

## Mask / Scale / Cast Audit
The portable path scales before masking with `torch.finfo(FP16).min` and computes `torch.softmax(scores, dim=-1, dtype=torch.float32).to(FP16)`. The probe confirms the sentinel and masked pattern survived TensorRT. No pre-softmax transform divergence was found.

## Softmax Micro Isolation
Native TensorRT softmax and an explicit FP32-cast TensorRT softmax were run on identical canonical `[1,16,8,8]` FP16 scores. Both matched portable softmax exactly (`vs_portable` relative-L2 `0`) and both differed from the FP32 oracle by relative-L2 `0.000154608`.

## FP32 Oracle
The oracle is `torch.softmax(scores.float(), dim=-1)`. Portable reference, native TensorRT and explicit FP32 TensorRT results were finite; native and explicit FP32 micro results were numerically indistinguishable at recorded precision.

## TensorRT Native Softmax
Native TensorRT did not reproduce the C1G softmax discrepancy on the same pre-softmax tensor. The C1G jump is therefore coupled to upstream graph execution/rounding rather than an isolated native softmax effect.

## TensorRT FP32 Softmax A/B
The single-variable FP32 softmax micro variant produced no material change. The independent Layer 0 FP32-softmax-only variant had final relative-L2 `0.003787680`, identical to unchanged B4.2 control (`0.003787680`).

## Layer0 Single-Variable A/B
| Candidate | Relative-L2 vs portable Layer 0 | Cosine |
| --- | ---: | ---: |
| Unchanged B4.2 control | 0.003787680 | 0.999992132 |
| Layer 0 explicit FP32 softmax only | 0.003787680 | 0.999992132 |

## Root Cause Decision
**SOFTMAX_HYPOTHESIS_REJECTED**. Same-input native TensorRT did not reproduce a material error, FP32 micro softmax did not improve it, and Layer 0 FP32 softmax did not reduce final hidden divergence.

## Remaining Numerical Sources
Remaining candidates are upstream Q/K projection, Q/K normalization, RoPE and reduction/precision behavior before the masked-score checkpoint, plus TensorRT FP16 normalization warnings. C1G input RMSNorm remains the first non-zero component difference (`0.000491762`), but is not independently confirmed as causal.

## B4.2 Regression
The unchanged B4.2 control remained finite with Layer 0 relative-L2 `0.003787680`; no engine or runtime regression was observed.

## Memory
No OOM or exit 137 occurred. Minimum recorded `MemAvailable` was `2,617,298,944` bytes and maximum process RSS was `3,623,332` KiB. New ONNX/engine binaries remain Jetson-local under `/tmp/phase2_2c1h_20260902T210000Z/`.

## C1 Status
C1 remains **BLOCKED**. This result does not authorize C1I, C2, or any repair.

## Recommended Next Experiment
After explicit authorization, isolate upstream Q/K projection, Q/K-normalization and RoPE precision/cast semantics with a new single-variable diagnostic. Do not modify B4.2 until that experiment is approved.

## Raw Artifacts
- `artifacts/c1h_20260902T210000Z/c1h_softmax_semantics_20260902T210000Z.json`
- `artifacts/c1h_20260902T210000Z/run.log`
- `src/phase2_2c1/c1h_softmax_semantics.py`
