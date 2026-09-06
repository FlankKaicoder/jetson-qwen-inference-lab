# Phase 6-D Plan: Evidence-Corrected Optimization Target Re-Ranking

## Objective

Phase 6-D performs selection and evidence synthesis only. It re-ranks the
remaining candidates using the frozen Phase 5 and Phase 6 evidence chains. It
does not run, build, deserialize, execute, profile, benchmark, or modify any
engine, ONNX model, precision setting, tactic, runtime, or kernel.

The only question is:

> After correcting the Phase 4/5 target ranking using all Phase 5 and Phase 6
> evidence, which remaining candidate deserves the next bounded investigation?

## Starting State

| Field | Value |
| --- | --- |
| Starting branch | `phase/06d-evidence-corrected-target-ranking` |
| Starting HEAD | `4c43da2b5bed5f189ca67d5d64b9814e17cbdb09` |
| Starting tracked working tree | Clean |
| New Jetson execution | None |
| New benchmark or profiling | None |
| Engine, ONNX, precision, tactic, or runtime change | None |

Protected untracked directories remain untouched and unstaged. No historical
artifact may be overwritten or deleted.

## Evidence Hierarchy And Boundary Rule

Evidence conflicts are resolved in this order:

1. Git commit / branch.
2. Raw result artifacts.
3. Experiment reports.
4. `docs/PROJECT_STATE.md`.
5. Conversation.

Every performance conclusion must state its measurement boundary. An all-trace
NSYS aggregate is not a steady-state share, CUDA-event latency, IProfiler
latency, or NCU duration. Theoretical occupancy is not achieved occupancy.
Device utilization is not occupancy. Effective bandwidth is not a direct DRAM
counter.

## Candidate Universe

Candidates must come from existing artifacts. The required core set is:

1. Layer-0 decode QK^T h16816 path.
2. QK^T layers 1-27 static xmma path, not runtime-isolated.
3. Attention x V across 28 layers.
4. `gate_proj` bounded shared-h16816 path.
5. Fused `q_proj;k_proj;v_proj` as one shared candidate, never three candidates.
6. TensorRT internal non-GEMM `__myl_*` family.
7. `down_proj` / `o_proj`.
8. `up_proj`, explicitly retained as `CLOSED_FOR_NOW`.
9. RMSNorm / RoPE as `NO_CURRENT_ACTION`.

The Phase 3-B persistent ExecutionContext is already a proven runtime
optimization and is not treated as a remaining candidate.

## Frozen Scoring Model

This model is frozen before any score is written. It uses five positive
dimensions and one uncertainty penalty, each with integer values 0 through 3.
No weights are used.

| Dimension | Meaning | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- |
| R | Runtime significance within a stated boundary | Insignificant / no evidence | Bounded | Meaningful | Dominant or significant |
| A | Attribution confidence across semantic, engine, runtime, and kernel evidence | UNKNOWN | LOW | MEDIUM | HIGH |
| S | Optimization-surface clarity and isolatability | No identifiable surface | Highly entangled / fused / unknown | Bounded but isolatable | Clean surface |
| B | Baseline quality / defect evidence | Strong evidence current implementation is competitive | No defect evidence | Unresolved suspicious behavior | Strong or proven inefficiency signal |
| F | Feasibility evidence for a bounded investigation | Contradicted / replacement already lost | Unknown | Plausible | Strong bounded feasibility evidence |
| U | Uncertainty penalty | Few critical unknowns | Moderate | Major | Fundamental unresolved identity or surface issue |

The score is:

```text
positive = R + A + S + B + F
final    = positive - U
```

If active candidates tie on `final`, use this frozen tie-break order: higher
`R`, then higher `A`, then lower `U`, then lower candidate ID. Ties do not
receive invented precision.

Scores rank active candidates only. `up_proj` is scored for transparency but is
excluded from active ranking because it remains `CLOSED_FOR_NOW`. RMSNorm /
RoPE are also excluded from active ranking. The score is a bounded evidence
score, not a predicted speedup or implementation readiness.

## Decision Rules

1. Runtime size alone cannot establish an optimization target.
2. HIGH semantic identity alone cannot establish an optimization target.
3. A clean replacement surface requires evidence of an isolatable operation or
   runtime boundary; it is not inferred from a kernel name.
4. `ATTRIBUTION_ONLY` is a valid final classification.
5. Negative evidence must be recorded for every candidate.
6. If the strongest candidate has meaningful runtime evidence but incomplete
   identity, kernel, surface, or workload evidence, its next action is a minimal
   attribution study.
7. The final gate must be exactly one of `NEXT_FEASIBILITY_TARGET_RECOVERED`,
   `NEXT_ATTRIBUTION_TARGET_RECOVERED`,
   `NO_CLEAN_NEXT_OPTIMIZATION_TARGET`, or `TARGET_RANKING_INCONCLUSIVE`.

## Frozen Authorization State

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
CUTLASS implementation: NOT AUTHORIZED
Engine rebuild: NOT AUTHORIZED
Precision change: NOT AUTHORIZED
Tactic forcing: NOT AUTHORIZED
```

## Stop Condition

Stop after the corrected ranking, gate update, project-state updates, commit,
and ordinary push. Do not start the selected follow-up experiment.
