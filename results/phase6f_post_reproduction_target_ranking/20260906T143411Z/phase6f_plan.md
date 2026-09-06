# Phase 6-F Plan: Post-Reproduction Evidence-Corrected Target Re-Ranking

## Objective

Re-rank the remaining optimization candidates after Phase 6-E. The only
question is:

> Given all evidence through Phase 6-E, what is the highest-value next bounded
> investigation target?

This is offline evidence synthesis and target selection only. No inference,
benchmark, Nsight Systems, Nsight Compute, engine execution or build, ONNX
export, or new experiment is performed.

## Starting State

| Field | Value |
| --- | --- |
| Starting branch | `phase/06e-qk-runtime-shape-invocation-attribution` |
| Starting HEAD | `826a5ae1735e1fe2f48ee583ddb86ed3f1c199f5` |
| New branch | `phase/06f-post-reproduction-target-ranking` |
| Analysis date | `2026-09-06` |
| New Jetson execution | None |
| New profiling or benchmark | None |
| Engine, ONNX, precision, or tactic change | None |

The four protected untracked directories remain untouched and unstaged.
Historical results are not overwritten.

## Evidence Boundary

The repository hierarchy remains:

1. Git commit and branch.
2. Raw result artifacts.
3. Experiment reports.
4. `docs/PROJECT_STATE.md`.
5. Conversation.

All reported times remain at their original measurement boundaries. All-trace
NSYS aggregates are not steady-state shares, CUDA Event times, IProfiler times,
or NCU durations. Derived row sums are bounded aggregates and do not create
per-launch performance conclusions.

## Candidate Universe

The required Phase 6-D candidates are retained:

1. Layer-0 decode QK^T historical h16816 path.
2. Attention x V across 28 layers.
3. QK^T layers 1-27 static xmma path.
4. Fused `q_proj;k_proj;v_proj` as one shared candidate.
5. `gate_proj` shared-h16816 path.
6. `down_proj` / `o_proj`.
7. TensorRT internal non-GEMM `__myl_*`.
8. `up_proj`, retained only as `EXCLUDED / CLOSED_FOR_NOW`.
9. RMSNorm / RoPE.

One additional historical candidate is included for completeness:

10. PyTorch/TensorRT CUDA Graph capture/replay path from Phase 3-D0. It is not
promoted because its captured graph omitted TensorRT kernels and failed
validation.

No new candidate category is invented.

## Frozen Scoring Model

Phase 6-F reuses the Phase 6-D model exactly:

```text
positive = R + A + S + B + F
final    = positive - U
```

All dimensions remain integer `0-3`. No weight, multiplier, or new
reproducibility score is added. Phase 6-E evidence is incorporated through the
existing dimensions:

| Dimension | Phase 6-E use |
| --- | --- |
| `R` | Existing historical and newly derived all-trace runtime totals only. |
| `A` | Existing semantic, runtime, engine, and kernel attribution. |
| `S` | Isolatability, replacement boundary, and operation clarity. |
| `B` | Suspicious-but-unresolved behavior; no defect is claimed. |
| `F` | Feasibility of a sharply bounded next attribution query or study. |
| `U` | Non-reproducibility, workload identity, shape, trigger, and surface uncertainty. |

For layer-0 QK, `F` is reduced from `1` to `0` and `U` is increased from `2`
to `3` because the bounded frozen-engine attempt did not reproduce h16816 and
did not establish workload comparability. `B` remains `2`: the historical
behavior is real and unresolved, but no tactic defect is proven.

Tie-break remains higher `R`, then higher `A`, then lower `U`, then lower
candidate ID. Classification overrides raw score for eligibility.

## Decision Rules

1. Historical runtime magnitude alone cannot make a target actionable.
2. HIGH semantic identity alone cannot make a target actionable.
3. A non-reproducible historical path is not erased, but is not promoted.
4. `NO_CURRENT_ACTION`, `ATTRIBUTION_ONLY`, and `CLOSED_FOR_NOW` are valid
   outcomes.
5. Gate C is acceptable if no candidate is sufficiently actionable.
6. A next experiment, if recommended, must be one sharply bounded study and
   must not begin during Phase 6-F.

## Required Outputs

Exactly these artifacts are required in this directory:

1. `phase6f_plan.md`
2. `environment.md`
3. `evidence_manifest.csv`
4. `candidate_master_inventory.csv`
5. `candidate_evidence_matrix.csv`
6. `candidate_scoring.csv`
7. `candidate_ranking.csv`
8. `phase6f_target_ranking_report.md`
9. `gate_update.md`

## Stop Condition

Stop after validation, state/index/registry updates, commit, and ordinary push.
Do not begin the recommended follow-up.
