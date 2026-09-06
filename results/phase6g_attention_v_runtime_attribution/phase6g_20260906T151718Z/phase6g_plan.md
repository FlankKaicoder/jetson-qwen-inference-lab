# Phase 6-G Plan: Representative Boundary Attention x V Runtime/Kernel Attribution

## Scope

Phase 6-G is attribution only. It answers one question:

```text
Can Attention x V be isolated as a meaningful runtime/kernel surface
under the representative steady boundary?
```

No CUDA implementation, FlashAttention, TensorRT Plugin, engine rebuild,
precision change, tactic forcing, Nsight Compute, broad Nsight Systems sweep,
new benchmark, or runtime redesign is authorized.

## Inputs

- Phase 6-F selected Attention x V across 28 decoder layers.
- Phase 6-A establishes HIGH semantic identity for `P6A_002`, `P6A_004`, ...,
  `P6A_056`, corresponding to `/MatMul_1`, `/MatMul_3`, ..., `/MatMul_55`.
- Phase 4-A establishes MEDIUM-confidence all-trace runtime mapping from those
  NVTX ranges to TensorRT layers and one xmma GEMM kernel family.
- The Phase 3-C Mixed persistent raw Nsys SQLite is the only raw trace source.

## Frozen Source Artifact

```text
Remote path: /tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite
bytes: 3477504
sha256: ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8
```

The source is queried through a read-only SQLite URI. It is not copied into
Git. Its remote existence, size, and SHA-256 must be verified before use.

## Method

1. Read the 28 AV NVTX ranges and all runtime API rows contained in each range.
2. Correlate launch runtime APIs to CUDA kernel rows by `correlationId`.
3. Join candidate identity only from Phase 6-A semantic artifacts.
4. Assign kernel instances to warmup, representative prefill, or one of four
   representative decode boundaries by NVTX containment.
5. Preserve all-trace totals and separately calculate representative-steady
   totals. Do not call all-trace values steady contribution.
6. Aggregate kernel identity, launch count, durations, and boundary split.
7. Use `UNKNOWN` when a field is not directly available.

## Boundaries

Representative steady Mixed persistent GPU kernel denominator:

```text
147,830,560 ns
```

Composed of:

```text
PHASE3B_STEADY_PREFILL_S8:      20,975,776 ns
PHASE3B_STEADY_DECODE_STEP_0:   36,161,760 ns
PHASE3B_STEADY_DECODE_STEP_1:   42,863,744 ns
PHASE3B_STEADY_DECODE_STEP_2:   21,190,464 ns
PHASE3B_STEADY_DECODE_STEP_3:   26,638,816 ns
```

The historical all-trace AV aggregate is `4,237,568 ns` over 112 calls and
must remain separately labeled.

## Gate Rules

- `AV_RUNTIME_SURFACE_RECOVERED`: semantic identity, runtime layer identity,
  kernel mapping, and representative steady contribution are all recovered.
- `AV_ATTRIBUTION_PARTIALLY_RECOVERED`: semantic/runtime mapping is partially
  recovered but kernel or steady surface is incomplete.
- `NO_ACTIONABLE_AV_SURFACE`: AV cannot be isolated or has semantic identity
  only.

Recovering an attribution surface does not prove optimization headroom. Q5 is
`NOT PROVEN` unless explicit feasibility and headroom evidence exists.

## Stop Condition

Stop after the gate decision and required artifacts are committed. Do not start
the next experiment.
