# Phase 2.2-C2 — Qwen3 Final RMSNorm Integration

Date: 2026-09-03
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `cb4aeabddc9b960f0a6b5ab8ae3c7549d8c2b3db`

## Objective

Validate the real pinned Qwen3 final RMSNorm as an independent TensorRT FP16
component and integrate it after the existing 28-layer decoder. C1 decoder
numerical drift is treated as a documented boundary, not reopened.

## Starting State

C1 is `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`: embedding and decoder
execution are functional, while corrected Layer 0 and Layer 27 relative-L2
versus the portable reference are `0.0038153231` and `2.0046324720`.

## Real Final RMSNorm Audit

The pinned checkpoint `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
was audited on Jetson. The module is `model.norm`, key `model.norm.weight`,
shape `[1024]`, 1024 elements, dtype BF16, epsilon `1e-6`, and weight SHA256
`40983d5d6018627e0ea73b2f7650e67c42ab160ceb82404016799eaffdf9d078`.
The checkpoint SHA256 is the frozen `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

## TensorRT Implementation

Two independent opset-17 graphs and TensorRT 10.3 FP16 engines were built:
`native` and `fp32_reduce`. The latter performs the variance reduction and
`rsqrt` in FP32, multiplies by the BF16-derived FP32 weight, then casts to
FP16. TensorRT emitted its known FP16 normalization warning; no historical
engine was changed.

## Precision Variants / Same-Input Numerical Validation

Both variants were finite and shape-correct. Metrics are computed after FP32
conversion against the portable source-faithful oracle.

| Input | Variant | Max abs | Mean abs | RMSE | Relative-L2 | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic `[1,8,1024]` | native | 0.015625 | 0.00136091 | 0.00230935 | 0.000590547 | 0.999999106 |
| Synthetic `[1,8,1024]` | FP32 reduction | 0.015625 | 0.00136091 | 0.00230935 | 0.000590547 | 0.999999106 |
| Synthetic `[1,1,1024]` | native | 0.015625 | 0.00052310 | 0.00137508 | 0.000350695 | 0.999999881 |
| Synthetic `[1,1,1024]` | FP32 reduction | 0.015625 | 0.00052310 | 0.00137508 | 0.000350695 | 0.999999881 |

The two TensorRT variants were numerically identical on this device. The
selected implementation is `fp32_reduce` because it states the pinned source
semantics explicitly and is the safer normalization contract.

## Prefill / Decode Validation

The selected engine produced finite CUDA-resident FP16 outputs for prefill
`[1,8,1024]` and decode `[1,1,1024]`; both shapes and bindings passed.

## Decoder Integration

The exact C1 embedding engine and corrected C1 closeout prefill decoder were
used read-only. Their Layer 27 output was passed into Final RMSNorm and the
execution completed with finite CUDA output. The same Layer 27 tensor sent to
portable and selected TensorRT norm produced max abs `27.109375`, RMSE
`0.32244748`, relative-L2 `0.08413005`, cosine `0.99645412`. This is a bounded
same-input attribution result on a large-magnitude FP16 decoder tensor and is
separate from the synthetic operator gate.

## Full-Path Diagnostic

No token generation, LM head or sampling was run. Any embedding → decoder →
norm comparison is **`END_TO_END_DIAGNOSTIC_ONLY`** because C1 documents
decoder drift; it is not used to reject the Final RMSNorm operator.

## Memory

The run recorded minimum `MemAvailable` of `4,533,264,384` bytes and maximum
RSS `1,400,540` KB in the validation process. No OOM and no exit 137 occurred.
ONNX/engine binaries remain Jetson-local under
`/tmp/phase2_2c2_20260903T/`.

## Gate

`FINAL_RMSNORM_OPERATOR = PASS`. TensorRT build, same-input numerical checks,
prefill/decode shape checks, decoder integration, finite CUDA output and
memory checks all passed. Overall result: **`PASS / BOUNDED`**.

## Limitations

The full 28-layer decoder remains subject to the closed C1 numerical
limitation. This C2 run is not a benchmark, profiler, quantization, LM head,
sampling or generation result.

## Next Step

Stop after C2 and await explicit authorization for C3 (LM Head). No C3 work was
started in this experiment.

## Artifacts

- `src/phase2_2c2/c2_final_rmsnorm.py`
- `artifacts/phase2_2c2_20260903T/c2_final_rmsnorm_validation.json`
- `artifacts/phase2_2c2_20260903T/final_norm_weight_audit.json`
- `artifacts/phase2_2c2_20260903T/final_norm_onnx_summary.json`
- `artifacts/phase2_2c2_20260903T/final_norm_engine_summary.json`
