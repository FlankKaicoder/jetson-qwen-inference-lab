# Phase 6-F Gate Update

## Final Gate

```text
PASS / BOUNDED
NEXT_ATTRIBUTION_TARGET_RECOVERED
```

## Selected Next Candidate

Attention x V across 28 layers is the highest eligible candidate. It is
`ATTRIBUTION_ONLY`.

The numerical rank-1 layer-0 decode QK^T h16816 candidate has the same final
score, 6, but is `NO_CURRENT_ACTION`. Its classification prevents selection.
Phase 6-E did not reproduce the historical h16816 path and left workload
equivalence, runtime shape, operation identity, and trigger `UNKNOWN`.

## Bounded Next Experiment

One read-only representative-boundary AV runtime/kernel attribution query
against the existing Phase 3-C raw Nsys SQLite. It must correlate odd
`/MatMul_*` AV ranges through runtime APIs to kernels, separate warmup from
representative steady boundaries, and save only derived compact evidence.

No new profiling, engine execution, implementation, or benchmark is authorized
as part of that follow-up decision.

## Authorization

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
CUTLASS implementation: NOT AUTHORIZED
Engine rebuild: NOT AUTHORIZED
Precision change: NOT AUTHORIZED
Tactic forcing: NOT AUTHORIZED
```
