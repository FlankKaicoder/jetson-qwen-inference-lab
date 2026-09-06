# Phase 6-H Plan: Attention x V Feasibility Boundary Study

## Scope

Phase 6-H is a repository-only feasibility-boundary synthesis. It answers one
question:

```text
Does the Phase 6-G Attention x V runtime surface have a provable
optimization opportunity?
```

No CUDA implementation, FlashAttention implementation, TensorRT Plugin,
CUTLASS tuning, kernel replacement, engine rebuild, ONNX change, precision
change, tactic forcing, runtime redesign, benchmark, inference, Nsight Systems
run, or Nsight Compute run is authorized or performed.

## Input Boundary

Phase 6-H reads only repository evidence frozen before this phase. The
authoritative runtime chain is the Phase 6-G direct
semantic-to-NVTX-to-runtime-to-correlationId-to-kernel attribution. Existing
NCU evidence is reviewed only to determine whether it can be attributed to the
same AV instances; name similarity alone is not ownership evidence.

The user's Phase 6-H prompt was truncated after the heading text `Final Gate:`.
The exact user-supplied gate specification is therefore
`UNAVAILABLE_TRUNCATED_PROMPT`. This plan freezes conservative tentative gates
without inventing omitted thresholds.

## Frozen Evidence Rules

1. `inefficient enough` requires a measured efficiency, utilization, duration,
   or headroom signal directly attributable to the Phase 6-G AV instances.
2. `clean enough` requires direct AV attribution and a bounded replacement
   surface. Kernel arguments, exact runtime shapes, backend identity, and
   tactic identity are allowed to remain uncertain, but their uncertainty must
   remain explicit.
3. Existing NCU metrics count as AV evidence only when the sample can be tied
   to at least one Phase 6-G AV `correlationId`, the same engine invocation,
   or an equivalent direct instance-ownership record. Kernel-name or family
   similarity is insufficient.
4. Do not compare different workload boundaries, different kernels, or NCU
   durations with NSYS event durations.
5. A TensorRT tactic defect cannot be claimed without matched workload and
   backend identity.
6. All missing evidence remains `UNKNOWN`; evidence that exists but cannot
   support a conclusion remains `INCONCLUSIVE` or `NOT PROVEN`.

## Frozen Tentative Gates

- `AV_OPTIMIZATION_OPPORTUNITY_PROVEN`: direct AV evidence proves a material
  inefficiency or headroom and the replacement surface is clean enough to
  justify a future feasibility experiment.
- `AV_OPTIMIZATION_OPPORTUNITY_PARTIALLY_SUPPORTED`: direct AV evidence gives
  a bounded positive signal, but at least one required condition is missing.
- `NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY`: either no direct efficiency
  evidence is available or the evidence is insufficient to prove that the
  surface is both inefficient enough and clean enough.

The final gate must be selected from these three gates. If the owner later
supplies the missing exact gate specification, it must be reconciled in a new
frozen plan before further work.

## Method

1. Verify the Phase 6-G totals and per-instance attribution surface.
2. Separate representative-steady and all-trace denominators.
3. Inventory existing NCU artifacts that contain the exact AV kernel name,
   recorded AV kernel family, or a similar TensorRT GEMM name.
4. Check each relevant NCU artifact for direct correlationId, invocation,
   TensorRT-layer, or workload ownership evidence.
5. Classify the current surface as inefficient enough, clean enough, directly
   attributable, NCU-supported, or feasible for a future study only according
   to the frozen rules.
6. Record `UNKNOWN` where evidence does not exist and stop after the gate.

## Stop Condition

Stop after the final gate, reports, registry updates, and commit. Do not start
another phase, profiling run, benchmark, or implementation.
