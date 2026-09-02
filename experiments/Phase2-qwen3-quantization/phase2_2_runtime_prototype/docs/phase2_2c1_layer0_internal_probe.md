# Phase 2.2-C1G - Qwen3 Layer 0 Internal Tensor Probe

Date: 2026-09-02
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `b8c9bc8edbb123e989a8a734eac6627077608336`

## Objective

Expose a minimal set of Layer 0 intermediate tensors in new, independent
TensorRT diagnostic engines and locate the first material numerical
amplification. B4.2 was loaded read-only and was never rebuilt or overwritten.

## Starting State

C1 remains `BLOCKED`. C1D excluded input handoff, stream, pointer and binding
transport causes. C1E found the first non-zero layer-boundary difference at
Layer 0: max abs `0.01171875`, relative-L2 `0.00378768`, cosine `0.9999921`.
C1F showed that the existing B4.2 artifact had no internal bindings.

## Authorization Scope

Only new Layer 0 diagnostic graphs/engines, timestamped raw evidence and this
report were created. No formal B4.2 runtime, complete 28-layer rebuild, repair,
C2, final norm, LM head, sampling, quantization, benchmark or Nsight work was
performed.

## Diagnostic Engine Construction

Three separate FP16 opset-17 TensorRT 10.3 engines were built from the same
Layer 0 weights and canonical input:

| Group | Exposed checkpoints | ONNX bytes | Engine bytes |
| --- | --- | ---: | ---: |
| A | input RMSNorm, Q/K/V, Q/K norm, RoPE, final hidden | 31,485,654 | 31,813,172 |
| B | attention scores, softmax, context, output projection, attention residual, final hidden | 31,484,467 | 31,882,724 |
| C | post-attention RMSNorm, gate/up, SiLU, gate x up, down, final hidden | 31,484,231 | 31,764,468 |

The engines are diagnostic-only artifacts. Large ONNX/engine binaries remain
outside Git's tracked files; raw JSON summaries are the repository evidence.

## Probe Validity

Each diagnostic final hidden was compared against the original B4.2
`hidden_l0` using the same `[1,8,1024]` FP16 input and `position_ids=0..7`.

| Group | Max abs | Mean abs | RMSE | Relative-L2 | Cosine | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 0 | 0 | 0 | 0 | 0.99999952 | PASS |
| B | 0.0078125 | 0.00040508 | 0.00057357 | 0.00250531 | 0.99999619 | PASS |
| C | 0 | 0 | 0 | 0 | 0.99999952 | PASS |

All three are within the diagnostic validity criterion (relative-L2 <= 0.01,
cosine >= 0.9999). `PROBE_VALIDITY = PASS`; internal tensors are usable for
attribution. The original B4.2 control remained unchanged at relative-L2
`0.00378768`, cosine `0.99999213` against the portable Layer 0 reference.

## Portable Reference Checkpoints

The portable reference reused the C1/B4.2 FP16 oracle semantics: RMSNorm
epsilon `1e-6`, Qwen3 dimensions hidden `1024`, Q=`16`, KV=`8`, head dimension
`128`, rotary dimension `128`, RoPE theta `1,000,000`, GQA repeat `2`, causal
mask, and SwiGLU MLP intermediate `3072`. All compared tensors were finite and
shape-equal.

## Layer0 Structure

Layer 0 follows Qwen3 ordering: input RMSNorm, Q/K/V projections, Q/K
normalization, RoPE, 16-query/8-KV-head GQA (repeat 2), causal attention,
output projection and residual, post-attention RMSNorm, SwiGLU MLP and final
residual add.

## Input Contract

The fixed canonical input is `B=1`, `S=8`, hidden size `1024`, FP16
`hidden_states`, and INT64 `position_ids` containing `0..7`. Identical Layer 0
weights and inputs are used by the portable reference and all diagnostic
engines.

## RMSNorm Analysis

Input RMSNorm is the first non-zero difference (relative-L2 `0.000491762`), but
it is not the first material amplification. Post-attention RMSNorm is
relative-L2 `0.003475933`.

## Attention Analysis

QK scores are nearly identical, then the softmax checkpoint shows the first
material jump; context, projection and residual metrics are in Group B.

## QK Norm Analysis

Q/K normalization is present. Q norm and K norm relative-L2 values are
`0.001795566` and `0.001838832`; neither is the first material amplification.

## RoPE Analysis

RoPE uses rotary dimension `128`, theta `1,000,000`, and positions `0..7`.
RoPE Q/K relative-L2 values are `0.001805041` and `0.001839084`.

## Group A Results

| Probe | Component | MaxAbs | MeanAbs | RMSE | RelL2 | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | input RMSNorm | 0.001953125 | - | - | 0.000491762 | 0.99999970 |
| A | Q projection | 0.013671875 | - | - | 0.001685598 | 0.99999809 |
| A | K projection | 0.015625 | - | - | 0.001933634 | 0.99999756 |
| A | V projection | 0.00390625 | - | - | 0.001863806 | 0.99999744 |
| A | Q norm | 0.09375 | - | - | 0.001795566 | 0.99999750 |
| A | K norm | 1.0 | - | - | 0.001838832 | 0.99999875 |
| A | RoPE Q | 0.09375 | - | - | 0.001805041 | 0.99999815 |
| A | RoPE K | 1.0 | - | - | 0.001839084 | 0.99999869 |
| A | final hidden | 0.01171875 | - | - | 0.003787680 | 0.99999213 |

The first non-zero difference in probe order is input RMSNorm. Its magnitude
is small relative to the final Layer 0 error and is not by itself a material
operator finding.

## Group B Results

| Probe | Component | MaxAbs | MeanAbs | RMSE | RelL2 | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B | attention QK scores | 0.046875 | - | - | 0.000000195 | 1.00000012 |
| B | softmax output | 0.0078125 | - | - | 0.003432333 | 0.99999410 |
| B | attention context/PV | 0.004394531 | - | - | 0.002940988 | 0.99999475 |
| B | output projection | 0.0078125 | - | - | 0.003277720 | 0.99999392 |
| B | attention residual | 0.0078125 | - | - | 0.003254358 | 0.99999392 |
| B | final hidden | 0.0078125 | - | - | 0.003811384 | 0.99999219 |

QK score error is effectively zero, while the softmax checkpoint jumps to
relative-L2 `0.003432333`. This is the first clear material amplification in
the attention path.

## Group C Results

| Probe | Component | MaxAbs | MeanAbs | RMSE | RelL2 | Cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C | post-attention RMSNorm | 0.017578125 | 0.001028744 | 0.001428899 | 0.003475933 | 0.99999332 |
| C | gate projection | 0.021484375 | 0.001792360 | 0.002540119 | 0.002949947 | 0.99999464 |
| C | up projection | 0.0078125 | 0.000967459 | 0.001242032 | 0.003975601 | 0.99999011 |
| C | SiLU(gate) | 0.009765625 | 0.000458458 | 0.000692071 | 0.003361402 | 0.99999273 |
| C | gate x up | 0.014648438 | 0.000233115 | 0.000428867 | 0.005109561 | 0.99998605 |
| C | down projection | 0.0078125 | 0.000570568 | 0.000751165 | 0.005827574 | 0.99998242 |
| C | final hidden | 0.01171875 | 0.000636592 | 0.000867174 | 0.003787680 | 0.99999213 |

MLP nonlinear/product checkpoints amplify the error further, but they occur
after the attention softmax jump and do not establish the original source.

## MLP Analysis

The MLP path adds later amplification: gate x up reaches relative-L2
`0.005109561` and down projection `0.005827574`. These are downstream and are
not treated as the first divergent operator.

## First Nonzero Difference

**Input RMSNorm**, relative-L2 `0.000491762` (Group A). This is a boundary
fact, not a root-cause claim.

## First Material Divergence

**Attention softmax output**, narrowed as the first material amplification:
QK score relative-L2 `1.95e-7` → softmax relative-L2 `3.43e-3` (approximately
four orders of magnitude). The Layer 0 final relative-L2 is `3.79e-3`, so the
softmax checkpoint accounts for the dominant abrupt increase visible in the
attention group. Result: `FIRST_MATERIAL_OPERATOR_FOUND` with operator status
**NARROWED**, not confirmed.

## First Divergent Operator

The first non-zero checkpoint is input RMSNorm. The first material operator
found by this diagnostic is the attention softmax output; the causal defect
remains unconfirmed.

## Numerical Semantics Audit

The diagnostic graph uses the portable reference operations exported to ONNX:
matmul for QK/PV and linear projections, elementwise causal masking,
softmax, RMSNorm subgraph, RoPE multiply/add, SiLU and residual adds. Inputs
and outputs are FP16 at the TensorRT boundary. TensorRT emitted warnings for
FP16 layer normalization and default-stream synchronization; the runtime may
force some Reduce/Pow normalization work to FP32. Exact per-node accumulation
precision and tactic selection are not directly exposed by this artifact and
remain `UNKNOWN`. No claim is made that the softmax discrepancy is caused by a
specific precision mode.

## Optional Precision A/B

Not performed. The probe narrowed the first amplification but did not confirm
a single precision semantic defect; no diagnostic variable was changed and no
repair was attempted.

## Root Cause Assessment

**NARROWED / NOT CONFIRMED.** The evidence directly localizes the first material
amplification to the attention softmax checkpoint in this valid probe path.
It does not distinguish softmax implementation, upstream FP16 representation,
or tactic/accumulation semantics as the causal defect. MLP gate×up/down adds
later amplification.

## Root Cause Ranking

1. Softmax implementation or accumulation precision semantics: narrowed, not
   confirmed.
2. Upstream FP16 representation entering softmax: possible; QK score error is
   only `1.95e-7`.
3. TensorRT tactic-specific behavior: possible; no tactic-level evidence was
   collected.

## Next Step

An explicitly authorized C1H single-variable softmax precision/implementation
A/B is the next diagnostic. Do not modify B4.2, repair the path, or start C2.

## B4.2 Regression Control

The original B4.2 engine remained read-only. Its canonical Layer 0 control was
finite, shape-correct and unchanged: relative-L2 `0.0037876803`, cosine
`0.9999921322`. No historical artifact was overwritten.

## Memory

All memory evidence is diagnostic-only. The three engine builds and executions
completed without OOM or exit 137. Jetson traces are retained in the raw JSON
artifacts; no production capacity claim is made.

## C1 Status

Phase 2.2-C1 remains **BLOCKED**. C2 must not start. No fix was applied.

## Recommended Next Experiment

An explicitly authorized C1H experiment may run one single-variable softmax
precision/implementation A/B with the same canonical input and a fresh
timestamped engine, after first preserving this C1G evidence. Do not modify
B4.2 and do not start C2.

Evidence files:

- `artifacts/c1g_layer0_group_a_20260902T200000Z.json`
- `artifacts/c1g_layer0_group_b_20260902T200000Z.json`
- `artifacts/c1g_layer0_group_c_20260902T200000Z.json`
