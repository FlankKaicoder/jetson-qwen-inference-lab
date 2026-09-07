# Phase 7 Global Optimization Target Reassessment Plan

## Objective

Answer one repository-driven question after Phase 3 through Phase 6:

```text
Does any remaining candidate satisfy runtime significance, semantic identity,
kernel identity, clean optimization surface, and evidence that the current
implementation may be insufficient?
```

If yes, the gate is `NEXT_FEASIBILITY_TARGET_RECOVERED`. If no, the gate is
`NO_PROVEN_CUSTOM_KERNEL_OPTIMIZATION_TARGET`.

## Scope

This is an offline, read-only evidence synthesis. No Jetson execution,
benchmark, NSYS run, NCU run, engine build, engine modification, ONNX change,
precision change, tactic forcing, kernel implementation, plugin, or
FlashAttention work is authorized.

## Evidence Hierarchy

1. Git commits and raw result artifacts.
2. Frozen experiment reports.
3. `docs/PROJECT_STATE.md`.
4. Conversation only as task context.

The authoritative scoring model remains the Phase 6-D model:

```text
positive = R + A + S + B + F
final    = positive - U
```

All dimensions are integers `0-3`. Weights and dimensions are not changed.
Phase 7 only incorporates frozen Phase 6-G and Phase 6-H evidence into the
existing matrix. Classification and prior closure gates override raw scores.

## Frozen Candidate Policy

The candidate freeze from the owner is preserved. Existing closed candidates
are not reopened:

| Candidate | Status |
| --- | --- |
| Persistent ExecutionContext | `PROVEN_OPTIMIZATION` |
| `up_proj` GEMM | `CLOSED_FOR_NOW` |
| Layer-0 QK h16816 | `NO_CURRENT_ACTION` |
| Attention x V | `NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY` |
| QK layers 1-27 | `ATTRIBUTION_ONLY` |
| Fused QKV | `ATTRIBUTION_ONLY` |
| `gate_proj` shared h16816 | `ATTRIBUTION_ONLY` |
| `down_proj` / `o_proj` | `NO_CURRENT_ACTION` |
| TensorRT `__myl_*` | `ATTRIBUTION_ONLY` |
| RMSNorm / RoPE | `NO_CURRENT_ACTION` |

The invalid current CUDA Graph prototype remains `NO_CURRENT_ACTION`.

## Phase 7-A/B/C/D Workflow

### A. Candidate Master Inventory

Create one complete inventory that includes the proven runtime optimization
and every active, blocked, or closed optimization candidate. Preserve the
Phase 6-F identities and evidence boundaries.

### B. Evidence Matrix And Scoring

Reconcile each candidate against the five positive dimensions and uncertainty
penalty. Only Attention x V changes from Phase 6-F because Phase 6-G/H added
direct representative-boundary runtime attribution and an explicit feasibility
boundary.

The AV update is:

| Dimension | Phase 6-F | Phase 7 | Reason |
| --- | ---: | ---: | --- |
| R | 1 | 2 | Steady AV is `4,237,568 ns / 2.866503%` with direct boundary attribution. |
| A | 3 | 3 | Attention x V semantic identity remains HIGH. |
| S | 1 | 2 | 112/112 instances and 28/28 layers map to one exact xmma kernel, but replacement shape and tactic remain UNKNOWN. |
| B | 1 | 1 | No defect, inefficiency, or headroom evidence. |
| F | 2 | 1 | The existing-SQLite attribution question is consumed; no directly owned AV NCU evidence exists. |
| U | 2 | 3 | Kernel arguments, workload shape, tactic identity, backend identity, and headroom remain UNKNOWN. |

The resulting score remains `6`, but its composition is now stronger on
attribution and weaker on implementation feasibility.

### C. Ranking And Gate Decision

Use the score only after classification and closure overrides. The highest
active evidence surface is Attention x V, but its Phase 6-H gate explicitly
blocks an optimization claim.

### D. Closeout

Write the evidence manifest, reassessment report, and gate update. Update the
project state, experiment index, and experiment registry. Commit the offline
synthesis; do not delete or overwrite historical artifacts.

## Success Criteria

Phase 7 passes only when every conclusion has a repository artifact path and
when missing evidence is explicitly `UNKNOWN`. The valid result may be
`NO_PROVEN_CUSTOM_KERNEL_OPTIMIZATION_TARGET`; that is not a failure of the
method.
