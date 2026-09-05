# Phase 5-A TensorRT GEMM Boundary Reconciliation

## Gate

**Phase 5-A Step 3: `CASE_2_SUPPORTED_BOUNDED`.**

The historical NSYS evidence supports a bounded reconciliation: the correlated
TensorRT `up_proj` GEMM kernel time is materially slower than the Phase 5-A
direct cuBLASLt median. This does **not** prove a tactic-selection defect,
does not identify the TensorRT backend, and does **not** authorize a CUDA
kernel, TensorRT Plugin, engine change, or Phase 5-B work.

The updated Phase 5-A decision is therefore:

```text
CASE_2_SUPPORTED_BOUNDED
NO_CUDA_IMPLEMENTATION_AUTHORIZED
NO_PROVEN_TACTIC_DEFECT
TENSORRT_BACKEND_IDENTITY_UNKNOWN
```

## Question

Why does the frozen Phase 4-E TensorRT `up_proj` layer timing report
approximately `153.28-159.76 us`, while the Phase 5-A direct cuBLASLt median
is `80.077961 us`?

The requested distinction is:

1. GEMM kernel compute time;
2. other kernel time;
3. launch/runtime boundary overhead.

## Method

No new profiling, engine execution, benchmark, kernel implementation, engine
modification, rebuild, plugin, tactic forcing, or runtime change was performed.

This step re-read the existing Phase 4-F Mixed persistent NSYS SQLite trace in
read-only mode and correlated:

```text
NVTX_EVENTS
  -> CUPTI_ACTIVITY_KIND_RUNTIME
  -> CUPTI_ACTIVITY_KIND_KERNEL
```

The source trace was `/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite`.
The Phase 4-F manifest hash remains:

```text
ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8
```

The exact script is
[phase5a_boundary_reconciliation.py](../../../experiments/Phase5-cuda-feasibility/scripts/phase5a_boundary_reconciliation.py)
and the exact remote-executed copy is
[phase5a_boundary_reconciliation_remote_copy.py](phase5a_boundary_reconciliation_remote_copy.py).
Both have SHA-256
`1801f699c0e697e38f416bbb1fb25573bb81133d34ad8ca887fa75c7b1f6d1fd`.

## Evidence

Primary raw artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `trt_boundary_reconciliation_raw.json` | `4715a583926831c8c0332dfbe7558e9c8eb24f40ef4b99bb11d5d86f4d5b24ed` |
| `kernel_breakdown.csv` | `2d101d5069b4240259a7599bbe7f32f3639175e176aeafd5265bc9cce68584bb` |

The requery found `28` unique `up_proj` layer names, `7` observed trace
invocations, `196` NVTX range instances and `196` correlated kernel events.
Every one of the 196 observed NVTX instances had exactly one correlated
runtime API and one correlated CUDA kernel.

This is an important boundary clarification. The Phase 4-F report uses "28
ranges" at the range-name level and then reports 196 correlated launches. The
new compact evidence resolves those rows as 196 NVTX range instances. It is
not evidence that one logical layer invocation serially launched seven CUDA
kernels.

All durations below are CUPTI event durations in nanoseconds unless marked
otherwise. The first observed trace invocation is treated separately as a
non-steady boundary.

### Family totals

| Kernel family | Count | Total duration | Median per launch | Min | Max | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `h16816` | `84` | `18,252,672` | `253,680` | `110,880` | `278,528` | `270,976` |
| `sm80_xmma_gemm` | `112` | `19,301,440` | `118,800` | `110,048` | `248,416` | `245,408` |
| other kernels | `0` | `0` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` |

No non-GEMM kernel was attributed to an `up_proj` NVTX instance in this trace.

### Invocation-level family transition

The family assignment changes by trace invocation, not by decoder layer:

| Trace invocation | Family | Count | Median duration |
| ---: | --- | ---: | ---: |
| 1 | `h16816` | `28` | `259.920 us` |
| 2 | `h16816` | `28` | `264.752 us` |
| 3 | `h16816` | `28` | `133.648 us` |
| 4 | `sm80_xmma_gemm` | `28` | `240.288 us` |
| 5 | `sm80_xmma_gemm` | `28` | `242.992 us` |
| 6 | `sm80_xmma_gemm` | `28` | `113.920 us` |
| 7 | `sm80_xmma_gemm` | `28` | `114.432 us` |

This bimodal pattern is an observed timeline result. The reason for the
family and speed transition remains `UNKNOWN`.

## Kernel Breakdown

For the 168 NVTX instances excluding the first invocation:

| Metric | Value |
| --- | ---: |
| Correlated GEMM kernel duration median | `147,424 ns` |
| Correlated GEMM kernel duration mean per range | `180,727.238 ns` |
| Correlated GEMM kernel duration minimum | `110,048 ns` |
| Correlated GEMM kernel duration maximum | `278,528 ns` |
| Correlated GEMM kernel duration P95 | `268,032 ns` |

The aggregate 196-launch median was `236,544 ns`. The lower
steady-state median reflects the four faster invocations in the historical
timeline.

The direct cuBLASLt Phase 5-A median remains `80,077.961 ns`. Ratios are
informational because the historical NSYS context and direct standalone
benchmark are not controlled-clock paired runs:

| Comparison | Ratio versus cuBLASLt median |
| --- | ---: |
| Steady-state 168-launch median | `1.841x` |
| All 196-launch median | `2.954x` |
| `h16816` median | `3.168x` |
| `sm80_xmma_gemm` median | `1.484x` |

Even the faster observed invocation medians, `113.920 us` and
`114.432 us`, remain above the direct cuBLASLt median by about `42-43%`.

## TensorRT Overhead Analysis

The evidence does **not** support the hypothesis that the `153-160 us` versus
`80 us` difference is mostly host launch or TensorRT layer overhead.

For the 168 instances excluding the first invocation:

| Metric | Value |
| --- | ---: |
| NVTX host range duration median | `42,496 ns` |
| Correlated kernel launch API duration median | `16,240 ns` |
| All observed runtime API duration median | `16,240 ns` |
| Correlated GEMM kernel duration median | `147,424 ns` |

The arithmetic `NVTX duration - kernel duration` residual has median
`-121,888 ns`. A negative value must not be interpreted as negative overhead or
as a disjoint runtime cost. The NVTX ranges are host-side boundaries and all
196 correlated kernels end after their NVTX range ends. Asynchronous GPU work
therefore extends beyond the host range.

The host launch API median is only about `16.24 us`. It is also a host API
duration, not an additive GPU execution cost. Thus it cannot account for the
approximately `67-80 us` difference between the direct cuBLASLt median and the
Phase 4-E TensorRT layer range.

Because exactly one GEMM kernel is correlated to each observed `up_proj`
instance and no other kernel is attributed, the reconciled dominant component
is the GEMM kernel itself. `IProfiler` remains a different measurement
boundary, so an exact additive decomposition of Phase 4-E layer time remains
`INCONCLUSIVE`.

## Updated Phase 5-A Gate Judgement

The historical trace distinguishes two bounded kernel modes:

* `h16816`: 84/196 launches, median `253.680 us`;
* `sm80_xmma_gemm`: 112/196 launches, median `118.800 us`.

Both are slower than the direct cuBLASLt median of `80.078 us`. The steady-state
correlated kernel median of `147.424 us` is also slower. This satisfies the
direction of Case 2: TensorRT GEMM kernel time appears materially slower than
the direct cuBLASLt baseline.

It does **not** establish the stronger claim that TensorRT selected a defective
tactic. The run is historical and uncontrolled for clocks, the observed family
switch cause is `UNKNOWN`, the exact accumulator/layout/alpha/beta semantics
remain `UNKNOWN`, and TensorRT backend identity remains `UNKNOWN`. The result
does not authorize a CUDA kernel, CUTLASS optimization kernel, TensorRT Plugin,
engine modification or rebuild, tactic forcing, or runtime redesign.

## Limitations

- Device power, clocks, p-state and utilization were not controlled and were
  mostly unreadable on the integrated platform.
- The historical trace is synthetic and has only seven observed invocations.
- The mapping from NVTX instance to a production decode invocation remains
  `UNKNOWN`.
- The cause of the `h16816` / `sm80_xmma_gemm` transition is `UNKNOWN`.
- `IProfiler`, NVTX, CUDA runtime API and CUDA kernel event timing are not
  identical measurement boundaries.
- No Phase 4 historical artifact or conclusion was modified.

## Exact Next Action

Stop after this reconciliation. Owner review is required before any further
Phase 5-B discussion or implementation work.
