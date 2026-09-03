# Phase 2.3-D Mixed Precision Policy

Status: `PASS / BOUNDED` (2026-09-04)

## Objective And Boundary

This experiment derives a 196-Linear mixed-precision assignment for the frozen
Qwen3-0.6B decoder. It reuses Phase 2.3-C sensitivity evidence and performs
bounded, exact-input component prevalidation for the selected INT8 assignments.
It does not build a TensorRT quantized decoder, export ONNX, create an engine,
run a 28-layer quantized runtime, benchmark, profile, or investigate C1.

Reference A remains the HF BF16 semantic baseline. Reference B, the current
TensorRT FP16 runtime, remains the direct quantization baseline. This D-stage
portable component evidence is a policy prevalidation and does not attribute
the existing HF-to-TRT FP16 C1 drift to INT8.

## Recovered C Evidence

Phase 2.3-C supplied all 196 static Linear reconstructions, 34 exact-input
portable dynamic targets, and eight deterministic TensorRT confirmations. Its
PT-W8A8-vs-F P95 robust outlier analysis has Q1 `0.0512919657`, Q3
`0.0973420833`, IQR `0.0460501176`, and high boundary `0.1664172597`.
The six C outliers are `L27:up_proj`, `L6:down_proj`, `L27:gate_proj`,
`L0:down_proj`, `L18:down_proj`, and `L9:down_proj`.

Family P95 ranking is `up_proj` (`0.3665403904`), `down_proj`
(`0.2204458933`), `gate_proj` (`0.2116336206`), `o_proj`
(`0.1526101978`), `v_proj` (`0.0976623753`), `k_proj`
(`0.0850183609`), then `q_proj` (`0.0538613725`). C established
`NO_CLEAR_MONOTONIC_LAYER_TREND`. Its TensorRT capability boundary is retained:
`PER_CHANNEL_QDQ_CAPABILITY_EXISTS_IN_PROBES`, while the real Qwen3 Linear
per-channel QDQ parser path is `BLOCKED`.

## Candidate Policies

| Policy | FP16 Linear | PT-W8A8 Linear | INT8 parameter coverage |
| --- | ---: | ---: | ---: |
| P0 all PT-W8A8 | 0 | 196 | 100.00% |
| P1 robust-outlier guard | 6 | 190 | 95.71% |
| P2 family guard, prevalidation | 57 | 139 | 59.29% |
| P2 family guard, final | 63 | 133 | 56.90% |

P2 is selected because the top two C-sensitive families are `up_proj` and
`down_proj`: all 56 are preserved FP16. The C outlier `L27:gate_proj` is also
preserved, producing 57 prevalidation FP16 assignments. P1 protects only the
six exact C outliers and is retained as a comparison policy, not selected.
Coverage is quantized parameter coverage, not measured memory saving.

## Streaming Calibration And Validation

The frozen BF16 HF model used real forward-pre-hooks on all 139 P2 quantized
Linear modules. For every prompt, hooks were consumed immediately and no raw
activation file was written. Two calibration-only passes processed 24 prompts:
the first recorded online absmax/count sufficient statistics; the second
accumulated each predetermined candidate's elementwise SSE and clipping count.
The disjoint 12-prompt evaluation then ran one prompt at a time.

Activation quantization is symmetric signed per-tensor INT8, range `[-127,127]`,
zero-point `0`, with clip factors `0.90`, `0.925`, `0.95`, `0.975`, `1.00`.
The chosen candidate minimizes aggregate calibration-only reconstruction MSE
with tie-break `(mse, factor)`. Recorded scales are float32; this does not make
a claim about the eventual TensorRT scale representation.

Prevalidation component outputs were finite for all 139 targets x 12 held-out
samples. Aggregate PT-W8A8-vs-FP16 relative-L2 is median `0.0540820572`, P95
`0.1008826614`, and maximum `0.2792529699`; aggregate cosine median is
`0.9985402524`. These are component-level policy measurements, not full-model
accuracy or TensorRT execution results.

## One Bounded Refinement

One fixed robust P95 pass over the 139 prevalidation assignments found six
additional outliers: `L2:v_proj`, `L14:o_proj`, `L2:q_proj`, `L27:o_proj`,
`L8:o_proj`, and `L2:k_proj`. They were changed once from INT8 to FP16. No
second refinement was performed. The final P2 policy therefore contains 63
FP16 and 133 PT-W8A8 Linear assignments.

## Safety And Limitations

Memory evidence records minimum available memory `2,911,473,664` bytes after
the evaluation and `OOM=false`, `exit137=false`. The committed artifact set is
JSON-only; it contains no raw activation `.pt`, ONNX, TensorRT engine, or
profiler result.

During setup, two newly-created Jetson `/tmp` D temporary directories were
removed with `rm -rf`. No repository file, historical artifact, model, engine,
or diagnostic was deleted, but this violated the project's absolute process
restriction. It is recorded here as a process safety deviation; no additional
cleanup was performed.

The policy has not been compiled into a TensorRT Q/DQ graph. Thus there is no
claim of full decoder execution, end-to-end accuracy, generation quality,
latency, throughput, power, memory saving, per-channel parser repair, INT4, or
Nsight evidence. C1 remains `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`.

## Gate And Next Boundary

`Phase 2.3-D = PASS / BOUNDED`. C-derived P0/P1/P2 policies are complete,
the final 196-entry policy and 139 prevalidation scales are recorded, exact
input provenance is complete, and the one permitted refinement is complete.

Phase 2.3-E is only policy-ready; it has not started and requires explicit
owner authorization.

## Evidence

- Script: `src/phase2_3d/derive_mixed_precision_policy.py`
- Artifacts: `artifacts/phase2_3d_20260904T001100Z/`
- Phase 2.3-C source evidence: `artifacts/phase2_3c_20260903T220600Z/`
