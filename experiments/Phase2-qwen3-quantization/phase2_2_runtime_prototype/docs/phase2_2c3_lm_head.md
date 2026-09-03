# Phase 2.2-C3 - Qwen3 LM Head Integration

Date: 2026-09-03
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `92e8857a25c974c61ac18901424fabefaba319d4`

## Objective

Validate the pinned Qwen3 LM Head as an independent TensorRT FP16 MatMul and
exercise the read-only `decoder -> final RMSNorm -> LM Head` handoff. This is
not a sampling, generation, benchmark, quantization, or C1 repair experiment.

## Layer0/Model Structure

The pinned config is Qwen3-0.6B: hidden size `1024`, vocabulary `151936`,
28 decoder layers, 16 query heads, 8 KV heads, head dimension `128`, RoPE
theta `1,000,000`, and RMSNorm epsilon `1e-6`. Layer 0 is the Qwen3-specific
`input RMSNorm -> Q/K/V -> per-head Q/K RMSNorm -> RoPE -> GQA attention ->
output projection/residual -> post-attention RMSNorm -> SwiGLU MLP` path.

## Input Contract

Synthetic deterministic hidden inputs use seed `20260903`, B=1, S=8 and
`[1,1,1024]` decode shapes. The LM Head contract is FP16 hidden states
`[B,S,1024]` to FP16 logits `[B,S,151936]`; the engine profile is min
`[1,1,1024]`, opt `[1,8,1024]`, max `[2,16,1024]`.

## Weight Audit

Jetson direct audit confirmed module `lm_head`, checkpoint key
`lm_head.weight`, shape `[151936,1024]`, BF16, 155,582,464 elements, no bias.
`model.embed_tokens.weight` has the same shape/dtype and an identical BF16
SHA256 (`8f29acf519434862d95613b2b4f6b9d14933a5e4d16baebf8ac0b33b410acfb6`),
confirming tied embeddings. The checkpoint SHA256 remains
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

## Portable Oracle and TensorRT

The portable oracle computes `hidden @ lm_head.weight.T`. An independent
opset-17 ONNX graph with one initializer and one MatMul was checked and built
by TensorRT 10.3 as a native FP16 engine. Existing C1 embedding/decoder and C2
Final RMSNorm engines were not modified.

## Component Metrics

Metrics are FP32 comparisons of portable and TensorRT outputs from
`artifacts/phase2_2c3_20260903T/c3_lm_head_validation.json`.

| Path | Max abs | Mean abs | RMSE | Relative-L2 | Cosine | Argmax | Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Synthetic prefill `[1,8,1024]` | 0.01953125 | 0.00109410 | 0.00160778 | 0.00169018 | 1.00001049 | 8/8 equal | 5/5 all tokens |
| Synthetic decode `[1,1,1024]` | 0.01562500 | 0.00101164 | 0.00147516 | 0.00164510 | 0.99999183 | 1/1 equal | 5/5 |
| Last-token `[1,1,1024]` | 0.01757812 | 0.00111600 | 0.00162776 | 0.00168146 | 0.99999166 | 1/1 equal | 5/5 |

All outputs were finite and shape-correct. Full prefill logits are
`[1,8,151936]`; last-token logits are `[1,1,151936]`.

## Decoder -> Final RMSNorm -> LM Head

The existing corrected 28-layer prefill engine, C1 embedding engine and C2
`fp32_reduce` norm engine were consumed read-only. The integrated output was
finite CUDA FP16 with shapes `[1,8,1024] -> [1,8,1024] -> [1,8,151936]`.
Using the identical Layer 27 tensor, portable-vs-TRT Final RMSNorm had
relative-L2 `0.08413005`; the resulting logits comparison had relative-L2
`0.04533300`, cosine `0.99895149`, argmax agreement `7/8`, and mean top-5
overlap `4.375/5` (minimum `0`). These values are explicitly
`END_TO_END_DIAGNOSTIC_ONLY` because C1 decoder drift remains closed and
unresolved; they do not reject the independent LM Head operator.

## Memory / Runtime

No OOM or exit 137 occurred. The validation trace recorded minimum
`MemAvailable` `3,094,806,528` bytes and maximum RSS `2,015,540 KB`.
TensorRT's default-stream warning remains recorded in the validation log.

## Gate

`LM_HEAD_OPERATOR = PASS`. Weight audit, ONNX checker, TensorRT build, native
FP16 MatMul, prefill/decode/last-token shapes, finite CUDA outputs, argmax and
top-5 checks all passed. Overall C3 result: **PASS / BOUNDED**.

## First Divergent Operator / Root Cause Ranking

No divergence was found within the independent LM Head beyond bounded FP16
MatMul rounding. The end-to-end diagnostic differences originate upstream in
the known C1 decoder/final-norm path; C3 does not localize or repair them.

## Next Step

Stop after C3. Do not start C4, sampling, generation, benchmark, Nsight,
quantization, or weight-streaming redesign without explicit authorization.

## Artifacts

- `src/phase2_2c3/c3_lm_head.py`
- `artifacts/phase2_2c3_20260903T/lm_head_weight_audit.json`
- `artifacts/phase2_2c3_20260903T/lm_head_onnx_summary.json`
- `artifacts/phase2_2c3_20260903T/lm_head_engine_summary.json`
- `artifacts/phase2_2c3_20260903T/c3_lm_head_validation.json`
