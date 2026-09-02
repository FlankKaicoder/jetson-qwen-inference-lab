# Phase 2.2-C1I - Qwen3 Layer 0 Q/K and RoPE Numerics

Date: 2026-09-02  
Branch: `phase/02-qwen3-quantization`  
Starting HEAD: `38660352861bad99b46ddb854e7648b6005935b4`

## Objective

Isolate the first upstream numerical divergence before attention softmax using
new Layer 0 TensorRT probes and same-input micro engines. C1 remains blocked;
no repair, C1J, C2, benchmark, profiler or 28-layer rebuild was performed.

## Starting State

B4.2 prefill was loaded read-only from `/tmp/phase2_2b4_2_20260902T082326Z/prefill_28layer.engine`; canonical FP16 embedding and Layer 0 weights were unchanged. Working tree was clean except the new diagnostic script.

## Exact Attention Pipeline

Qwen3 Layer 0 uses input RMSNorm, Q/K/V projections, reshape/transpose,
per-head Q/K RMSNorm, rotary embedding, 16 Q heads/8 KV heads (GQA repeat 2),
QK matmul, scale/mask/softmax, output projection/residual, post-attention
RMSNorm and SwiGLU MLP. Hidden size and head dimension are 1024 and 128;
rotary dimension is 128, theta is 1,000,000 and epsilon is 1e-6.

## Canonical Input

`B=1, S=8`, FP16 hidden input from the C1 embedding reference, INT64
`position_ids=0..7`; all weights and inputs are identical between portable and
TensorRT paths.

## RMSNorm Micro Results

Input RMSNorm portable-vs-TRT relative-L2 is `0.000491762`, cosine
`0.999999702`. The first non-zero difference is small and agrees with C1G.

## RMSNorm FP32 A/B

The final run used a dedicated single-output wrapper. FP32 input-RMSNorm,
Q-norm, K-norm and Q/K-RoPE variants all produced the same final relative-L2
`0.034715209` versus the portable Layer 0 output (B4.2 control remains
`0.003787680`). No variant improved the control by the predeclared 50% test.

## Q Projection

Relative-L2 `0.001685598`, cosine `0.999997973`; per-head maximum
`0.002067495` (head 8). This is distributed FP16 projection error, not the
first material amplification.

## K Projection

Relative-L2 `0.001933634`, cosine `0.999997735`; per-head maximum
`0.002275725` (head 7). No isolated K projection source is confirmed.

## Q Norm

Relative-L2 `0.001795566`, cosine `0.999997497`; same-input Q-norm micro
relative-L2 `0.000449720`. Q/K normalization is present and numerically close.

## K Norm

Relative-L2 `0.001838832`, cosine `0.999998748`; same-input K-norm micro
relative-L2 `0.000456408`. No K-norm major source is confirmed.

## Q/K Norm Same-Input Micro

Both micro engines were run with contiguous `[B, heads, S, D]` inputs and
matched the portable FP16 boundary to better than `5e-4` relative-L2.

## Q/K Norm Precision A/B

The valid dedicated-wrapper A/B rows show no material final improvement for
any tested FP32 variant; same-input micro results do not indicate a Q/K norm
major source.

## RoPE Q

Q RoPE is the first material divergence: same-input micro relative-L2
`0.092710741`, cosine `0.995700836`; full probe relative-L2 `0.092714824`.
Per-head maximum is `0.119861849` (head 3).

## RoPE K

K RoPE same-input micro relative-L2 is `0.014421386`, cosine `0.999894559`;
full probe relative-L2 `0.014535242`. Per-head maximum is `0.070175037`
(head 1). K RoPE is material but smaller than Q RoPE.

## QK Score Reconciliation

The full probe gives QK raw relative-L2 `0.043750536` and scaled relative-L2
`0.043813497`, cosine `0.9992086`. This is downstream of the Q RoPE jump and
is consistent with RoPE-driven attention input error. The unchanged B4.2
control remains relative-L2 `0.003787680` versus the portable Layer 0 output.

## Error Amplification Table

| Checkpoint | Relative-L2 | Cosine |
| --- | ---: | ---: |
| Input RMSNorm | 0.000491762 | 0.999999702 |
| Q projection | 0.001685598 | 0.999997973 |
| K projection | 0.001933634 | 0.999997735 |
| Q norm | 0.001795566 | 0.999997497 |
| K norm | 0.001838832 | 0.999998748 |
| Q RoPE | 0.092714824 | 0.995700836 |
| K RoPE | 0.014535242 | 0.999894559 |
| QK raw | 0.043750536 | 0.999208927 |

## TensorRT Warning Mapping

Logs retain the known default-stream, INT64 position binding, DLA fallback and
FP16 Reduce/Pow layernorm warnings. They do not prove a specific RMSNorm or
RoPE implementation fault; warning-to-operator mapping is `UNKNOWN`.

## Probe Validity

All outputs were finite and shape-equal. The B4.2 control was unchanged and
finite. The Q/K/RoPE intermediate probe is therefore valid for localization;
the explicitly noted full-layer A/B rows are invalid and excluded.

## Root Cause Decision

**ROPE_MAJOR_SOURCE_CONFIRMED**. Q/K projection and per-head norm remain small,
while same-input Q RoPE shows a 0.0927 relative-L2 jump and K RoPE 0.0144,
followed by QK raw 0.0438. This confirms RoPE as the major upstream source in
this diagnostic, without claiming the exact kernel/cast mechanism.

## B4.2 Regression

B4.2 was read-only; Layer 0 control remains relative-L2 `0.003787680` and no
engine/runtime regression, OOM or exit 137 occurred.

## Memory

The Jetson run completed without OOM or exit 137. New ONNX/engine binaries and
logs remain Jetson-local under `/tmp/phase2_2c1i_20260902T221500Z/`; JSON and
log summaries are retained in the repository artifact directory.

## C1 Status

C1 remains **BLOCKED**. C1J and C2 must not start.

## Recommended Next Experiment

After explicit authorization, run a narrowly scoped RoPE-only precision/cast
diagnostic (FP32 frequency, cosine/sine cache precision, rotary dimension and
position broadcast) with a dedicated single-output Layer 0 wrapper. Do not
modify B4.2 or repair the runtime automatically.

## Raw Artifacts

- `artifacts/c1i_20260902T223000Z.json`
- `artifacts/c1i_20260902T223000Z_run.log`
- Earlier preserved diagnostic attempts: `artifacts/c1i_20260902T220000Z.json` and `artifacts/c1i_20260902T221500Z.json`
- `src/phase2_2c1/c1i_qk_rope_numerics.py`
