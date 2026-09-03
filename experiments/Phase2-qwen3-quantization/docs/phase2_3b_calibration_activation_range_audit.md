# Phase 2.3-B Calibration / Activation Range Audit

Date: 2026-09-03

## Objective

Audit the activation distribution entering the already proven real Layer 0
`model.layers.0.self_attn.q_proj` path, derive fixed per-tensor INT8 activation
scales from a calibration split only, and measure held-out activation-only
error. This is a bounded component calibration experiment; it is not a
full-model calibration, 28-layer INT8 runtime, semantic generation, benchmark,
or production precision-policy result.

## Frozen State

- Branch at start: `phase/02-qwen3-quantization`; Windows starting HEAD:
  `9a577fe16328a26e55adc4c7dad6b59dbea3c3f8`.
- Frozen model: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, checkpoint SHA256
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Frozen weight policy: Layer 0 `q_proj`, symmetric per-tensor INT8 W8,
  zero-point 0, scale `0.00507354736328125`, unchanged from Phase 2.3-A.
- Phase 2.2 remains `CLOSED / PASS / BOUNDED`; C1 remains
  `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED` and was not reopened.

## Phase 2.3-A Baseline

Phase 2.3-A established FP16, W8-QDQ and W8A8-QDQ execution for this component.
Its canonical tensor was the B2 handoff `x`; B2 did not prove that `x` was the
actual argument to `q_proj`. This experiment therefore preserves A unchanged
and establishes a new exact q_proj-input corpus.

## Exact q_proj Input Provenance

In the frozen `phase1-hf` environment, the real embedding and Layer 0 weights
were loaded without loading the 28-layer model. A forward-pre-hook on
`model.layers.0.self_attn.q_proj` captured the actual input for every sample.
The capture matched a direct `model.layers.0.input_layernorm` computation exactly
for all 36 samples.

Classification: `EXACT_QPROJ_INPUT_PROVEN`.

| Field | Evidence |
| --- | --- |
| Producer | `model.layers.0.input_layernorm` |
| Consumer | `model.layers.0.self_attn.q_proj` |
| Relationship | post-input-layernorm, pre-q_proj |
| Shape | `[1, token_count, 1024]` |
| Capture dtype | BF16; cast once to FP16 for TensorRT variants |
| Raw handoff | Jetson-local `qproj_inputs_bf16.pt` (not committed) |
| Handoff SHA256 | recorded in both manifests and `qproj_input_provenance.json` |

## Calibration Corpus

The deterministic calibration split contains 24 prompts, six per length group:
short, medium, long and very-long. Categories are English, Chinese, numerical,
structured, code-like and mixed-language content. Real local tokenizer encoding
was used with `add_special_tokens=False`; token IDs and token-ID SHA256 values
are committed in `calibration_manifest.json`. Actual token counts range from 6
to 169.

## Evaluation Corpus

The held-out evaluation split contains 12 prompts, three per length group, with
English, Chinese, numerical, structured, code-like and mixed-language coverage
across the groups, plus a distinct `heldout=evaluation` source suffix.
All token-ID SHA256 values are disjoint from calibration, and sample IDs are
disjoint by construction. Actual token counts range from 10 to 140. No random
tensor was used as formal calibration or evaluation input.

## Activation Range Distribution

Calibration exact q_proj inputs were finite and BF16. Global calibration range
was min `-3.4375`, max `2.8125`, absmax `3.4375`. Per-sample absmax was:

| Statistic | Value |
| --- | ---: |
| Minimum | 2.3125 |
| Median | 2.8203125 |
| Mean | 2.80859375 |
| P90 | 3.0921875 |
| P95 | 3.19140625 |
| Maximum | 3.4375 |
| Global absolute P99 | 0.6484375 |
| Global absolute P99.9 | 1.734375 |
| Global absolute P99.99 | 2.546875 |

`cal_023` supplied the largest calibration absmax. Evaluation global absmax
was `3.203125`, below the calibration global maximum.

## Scale Stability

The calibration max/median absmax ratio was `1.2188366`; P95/median was
`1.1315789`. These are descriptive measurements, not threshold-based Gate
criteria. The largest range was not spread across many samples: `cal_023` was
the sole maximum-range sample.

## Calibration Candidates

All candidates were derived from calibration activations only, with symmetric
zero-point 0 and scale equal to range/127.

| Policy | Calibration range | Scale |
| --- | ---: | ---: |
| GLOBAL_ABSMAX | 3.4375 | 0.0270669291 |
| P99_9 | 1.734375 | 0.0136564961 |
| P99_99 | 2.546875 | 0.0200541331 |
| BOUNDED_MSE_CLIP | 3.09375 (`0.90 * global absmax`) | 0.0243602362 |

The MSE policy used only the predetermined factors `0.90, 0.925, 0.95,
0.975, 1.00`; `0.90` minimized calibration reconstruction MSE. This is a
bounded grid, not open-ended tuning.

## Activation Reconstruction

Held-out reconstruction is exact q_proj input versus fixed-scale INT8 Q/DQ.
Values below are median / P95 / maximum across 12 evaluation samples.

| Policy | Relative-L2 | RMSE | Clipping percentage |
| --- | --- | --- | --- |
| GLOBAL_ABSMAX | 0.037914 / 0.038571 / 0.038716 | 0.007805 / 0.007837 / 0.007845 | 0 / 0 / 0 |
| P99_9 | 0.073877 / 0.093839 / 0.094995 | 0.015271 / 0.019485 / 0.019838 | 0.09320 / 0.11270 / 0.11489 |
| P99_99 | 0.029487 / 0.035400 / 0.036610 | 0.006036 / 0.007326 / 0.007566 | 0.00835 / 0.01898 / 0.01953 |
| BOUNDED_MSE_CLIP | 0.034320 / 0.034721 / 0.034757 | 0.007040 / 0.007089 / 0.007094 | 0 / 0.00110 / 0.00244 |

## FP16 / W8 / W8A8 Evaluation

Each evaluation sample used the same captured input, cast once to FP16, for all
three TensorRT variants. FP16 uses real FP16 weight, W8 uses the frozen A W8
weight, and W8A8 adds one candidate fixed activation scale. All 48 component
executions (4 policies x 12 samples) produced finite outputs.

## Activation-Only Incremental Delta

The primary metric is `W8A8 vs W8`, isolating activation quantization from the
frozen weight error. Relative-L2 is mean / median / P95 / maximum; cosine is
mean / median / minimum.

| Policy | Relative-L2 | Cosine |
| --- | --- | --- |
| GLOBAL_ABSMAX | 0.021658 / 0.021891 / 0.022442 / 0.022601 | 0.999766 / 0.999766 / 0.999745 |
| P99_9 | 0.051617 / 0.051703 / 0.071147 / 0.071661 | 0.998819 / 0.998874 / 0.997721 |
| P99_99 | 0.017943 / 0.016737 / 0.022595 / 0.023394 | 0.999838 / 0.999860 / 0.999734 |
| BOUNDED_MSE_CLIP | 0.019453 / 0.019526 / 0.020111 / 0.020117 | 0.999811 / 0.999810 / 0.999798 |

## Total Quantization Delta

Secondary `W8A8 vs FP16` relative-L2 mean / median / P95 / maximum and cosine
mean / median / minimum:

| Policy | Relative-L2 | Cosine |
| --- | --- | --- |
| GLOBAL_ABSMAX | 0.034191 / 0.034316 / 0.035477 / 0.035942 | 0.999416 / 0.999411 / 0.999354 |
| P99_9 | 0.058107 / 0.058039 / 0.076132 / 0.076441 | 0.998462 / 0.998516 / 0.997352 |
| P99_99 | 0.032014 / 0.031466 / 0.035528 / 0.035783 | 0.999487 / 0.999505 / 0.999365 |
| BOUNDED_MSE_CLIP | 0.032856 / 0.033015 / 0.033980 / 0.034272 | 0.999460 / 0.999456 / 0.999413 |

## Weight Quantization Context

The frozen W8-versus-FP16 context was identical across policies: relative-L2
mean / median / P95 / maximum `0.026571 / 0.026780 / 0.027547 / 0.027934`;
cosine mean / median / minimum `0.999635 / 0.999637 / 0.999605`.

## Calibration Policy Selection

`BOUNDED_MSE_CLIP` is selected for the next sensitivity experiment. Selection
uses the lowest held-out activation-only P95 relative-L2, then maximum
relative-L2, then the predetermined simplicity order. Its held-out activation
only P95/max are `0.020111 / 0.020117`, with median clipping `0`, P95 clipping
`0.001099%` and maximum `0.002441%`. P99.99 has a lower median but a higher
P95/max under this corpus; GLOBAL_ABSMAX has no clipping but larger error. The
selection is target/corpus-specific and not a production accuracy threshold.

## Weight-Dominance Hypothesis

`WEIGHT_QUANTIZATION_DOMINANCE_SUPPORTED` for this target/corpus: selected-policy
W8-vs-FP16 median/P95 relative-L2 are `0.026780/0.027547`, while the A8-only
W8A8-vs-W8 values are `0.019526/0.020111`. This supports, but does not
generalize beyond this component and corpus, the hypothesis that frozen W8
weight error is a major contributor. Weight granularity was not changed.

## Engine Evidence

The selected policy scale is `0.0243602362`. Detailed TensorRT EngineInspector
shows an Int8 activation output from the Q node, an Int8 weight input to the
fusion, and tactic
`sm80_xmma_gemm_i8i8_i8i32_f32_tn_n_tilesize64x64x64_stage4_warpsize2x2x1_tensor16x8x32_by_fusion_tactic`.
Therefore `INT8_COMPUTE_PROVEN = YES` for the selected graph/profile. No
Nsight or timing benchmark was run.

## Memory Safety

Capture maximum RSS was 1,957,011,456 bytes; evaluation maximum RSS was
3,264,868,352 bytes. Minimum recorded `MemAvailable` was 3,139,993,600 bytes
during evaluation. `OOM=false`, `exit137=false`. These are safety observations
only and are not interpreted as a leak or performance result.

## Gate

`Phase 2.3-B = PASS`.

Exact provenance, deterministic disjoint splits, real activation collection,
calibration-only fixed scales, held-out reconstruction, W8 and W8A8 component
evaluation, activation-only and total deltas, clipping evidence, selected fixed
policy, selected-policy INT8 engine evidence, finite outputs and safety checks
all passed. MSE clipping was bounded and optional per-channel activation
execution was not performed.

## Limitations

This remains one Layer 0 q_proj, one checkpoint, one bounded local corpus and
one TensorRT dynamic profile. The selected MSE policy is not a final model-wide
calibration policy. No full-model calibration, 28-layer INT8 runtime, semantic
generation, per-channel weight experiment, mixed precision, INT4, benchmark or
Nsight work was performed. Phase 2.2/C1 historical drift remains unchanged.

## Next Authorized Boundary

`Phase 2.3-C — Layer / Operator Sensitivity`. It was not started.

## Artifacts

Committed evidence is under
`experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T205610Z/`.
Raw BF16 inputs remain Jetson-local under
`/tmp/phase2_3b_20260903T205610/capture2/qproj_inputs_bf16.pt` and are not
committed.
