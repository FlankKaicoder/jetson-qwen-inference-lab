# Phase 2.3-E — 28-Layer Mixed-Precision Quantized Runtime

Status: `PASS / BOUNDED` (2026-09-04)

## Objective

Integrate the frozen Phase 2.3-D 196-Linear mixed-precision policy and the
target-specific activation scales into the existing Qwen3-0.6B 28-layer
TensorRT autoregressive runtime, and verify functional correctness of prefill,
decode, KV Cache, Final RMSNorm, LM Head, and greedy generation.

This phase does not redesign the policy. Primary comparison boundary remains
`TRT MIXED vs TRT FP16`; the existing HF-to-TRT FP16 C1 drift is not re-opened.

## Frozen Policy

- Primary policy: `P2_FAMILY_GUARD_REFINED`
- Manifest: `mixed_precision_policy_primary_final.json`
- Linear assignments: 196 / 196
- FP16 Linear: 63
- PT-W8A8 Linear: 133
- INT8 linear coverage: 67.86%
- INT8 parameter coverage: 56.90%
- Activation scales: 139 entries recorded (133 used + 6 refined-to-FP16
  unused); all 133 primary PT-W8A8 targets have a frozen BOUNDED_MSE_CLIP scale.
- Deployable states: FP16 and PT-W8A8 only. No per-channel, INT4, FP8, or
  parser workaround was attempted.

## Runtime Architecture

The existing validated Phase 2.2 runtime is reused, not replaced:

```text
Tokenizer
  -> Embedding engine
  -> 28-layer mixed TensorRT decoder (prefill + decode)
  -> Final RMSNorm engine
  -> LM Head engine
  -> CPU NumPy greedy sampling
  -> autoregressive loop
```

The B4.2 FP16 prefill/decode ONNX graphs are the source. For each PT-W8A8
target, the existing MatMul is transformed by replacing its FP16 weight
initializer with a per-tensor symmetric INT8 weight, adding a weight
DequantizeLinear, and wrapping the activation in QuantizeLinear/DequantizeLinear
using the frozen target-specific activation scale. FP16 targets are preserved
verbatim. This is the same explicit Q/DQ semantics proven in Phase 2.3-A/B/C.

## Policy Application

- `policy_application_audit.json`: 196 / 196 assignments, 63 FP16, 133 PT-W8A8.
- Every PT-W8A8 target has a real weight initializer, verified weight shape, a
  per-tensor weight scale derived from the actual FP16 ONNX initializer, and a
  frozen activation scale.
- `layer_build_summary.json`: per-layer FP16/PT-W8A8 counts; 28 / 28 layers
  transformed and built.

## Layer Build

- Prefill mixed engine build: `PASS`.
- Decode mixed engine build: `PASS`.
- Total Linear assignments transformed: 196.

## INT8 Engine Evidence

EngineInspector (TensorRT 10.3, detailed) shows 133 INT8 tensor-core GEMM
tactics per engine, matching the 133 PT-W8A8 Linear targets:

- Prefill: 106 `sm80_xmma_gemm_i8i8_i8i32_f32_...` + 27
  `sm80_xmma_gemm_i8f32_i8i32_f32_...`.
- Decode: 106 `sm80_xmma_gemm_i8f32_i8i32_f32_...` + 27
  `sm80_xmma_gemm_i8i8_i8i32_f32_...`.

The `i8i8` and `i8f32` tactics are both SM87 INT8 tensor-core GEMMs with an
Int8 activation input and an Int8 weight input. This is
`INT8_COMPUTE_PROVEN` for the quantized Linear targets under the current
deployable PT-W8A8 path. `up_proj` and `down_proj` are fully FP16 in the
primary policy, so there is no INT8 target for those two families
(`NO_INT8_TARGET_IN_PRIMARY_POLICY` for up_proj/down_proj).

## Memory Lifetime

Build and runtime remained streaming and bounded. Engines were serialized to
disk and released; no all-layer HF residency was introduced. No OOM and no
exit 137 occurred. Peak process RSS was bounded and swap usage was transient
during TensorRT build only.

## Representative Layer Validation

Same-input mixed vs TRT FP16 hidden outputs (integration diagnostics only):

| Layer | relative_L2 | cosine |
| --- | ---: | ---: |
| L0 | 0.0577 | 0.9983 |
| L9 | 0.0083 | 0.999997 |
| L18 | 0.0200 | 0.9998 |
| L27 | 0.1413 | 0.9901 |

## Full Prefill

B=1, S=10 real English prompt. 28 / 28 layers executed, all outputs finite,
no NaN/Inf, hidden shapes `[1,10,1024]`, KV shapes `[1,8,10,128]`, 28-way
K/V pointer isolation. Status `PASS`.

## Decode

4 forced decode steps. Cache lengths progressed 2 -> 3 -> 4 -> 5, all K/V
prefixes exact, all outputs finite. Status `PASS`.

## KV Cache

Free generation cache lengths progressed 1 -> 2 -> 3 -> 4 -> 5, all finite,
28-way K/V pointer isolation preserved, every K/V prefix exact. Status `PASS`.

## Final RMSNorm / LM Head

The existing Final RMSNorm (`fp32_reduce`) and LM Head engines were used
read-only. The full decoder -> Final RMSNorm -> LM Head -> greedy path executed
finite and produced valid tokens. Status `PASS`.

## Same-Prefix Comparison

Definition: `QUANTIZED_RUNTIME_VS_TRT_FP16` (mixed vs FP16, same input tokens).

- Prefill last hidden: relative-L2 `0.008568`, cosine `0.9999999`.
- Prefill last-token logits are degenerate/identical in both runtimes
  (documented C1 background; not a quantization claim).
- Forced decode (same forced continuation `[21806, 0, 358, 2776]`):
  - hidden relative-L2 `0.542` .. `0.603`, cosine `0.819` .. `0.855`.
  - logits relative-L2 `0.324` .. `0.371`, cosine `0.934` .. `0.947`.
  - top-1 agreement `0/4`, top-5 overlap `1..2/5`.

These are quantization-induced component/runtime deltas relative to the current
TRT FP16 runtime; they are not HF drift.

## Autoregressive Generation

Prompt `Hello` (`[9707]`), 4 decode steps, greedy:

- TRT FP16 tokens: `[0, 46309, 46309, 46309, 46309]`.
- Mixed tokens: `[0, 99815, 99815, 99815, 99815]`.
- First divergence step: 1 (after the shared degenerate first token `0`).

Runtime functional status `PASS`. Token mismatch is a result, not a runtime
failure, and no arbitrary token-equivalence gate is applied.

## Gate

`Phase 2.3-E = PASS / BOUNDED`.

- Primary policy loaded and applied 196/196.
- All 133 required INT8 scales present.
- 28/28 mixed layer engines built.
- EngineInspector INT8 GEMM evidence present.
- Prefill / decode / KV Cache / Final RMSNorm / LM Head / generation all
  functional PASS.
- No OOM / exit137.
- No safety violation.

Bounded because substantial mixed-vs-FP16 numerical and token divergence exists.
Runtime classification: `PRIMARY_POLICY_RUNTIME` (no fallback policy was
required).

## Limitations

- The activation scales are frozen from HF BF16 calibration and are applied
  verbatim to the portable FP16 decoder; no recalibration was performed.
- The existing C1 HF-to-TRT FP16 drift remains `CLOSED /
  NUMERICAL_LIMITATION_UNRESOLVED` and is not re-opened.
- Engine storage is 27% smaller than FP16, but this is not yet a full-runtime
  memory or latency claim (deferred to Phase 2.3-F).

## Next Authorized Boundary

Phase 2.3-F — Accuracy / Memory / Performance Comparison (authorized only if
E is PASS or PASS / BOUNDED).

## Evidence

- Script: `src/phase2_3e/build_mixed_runtime.py`
- Artifacts: `artifacts/phase2_3e_20260904T034300Z/`
- Mixed ONNX/engines remain Jetson-local under `/tmp/phase2_3e_20260904T020000Z/`.
