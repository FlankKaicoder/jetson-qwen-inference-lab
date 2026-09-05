# Phase 4-E up_proj Narrow Baseline Probe Report

## Gate

**Phase 4-E: `PASS / BOUNDED`**

The `PASS` component is supported by complete TensorRT `IProfiler` layer timing
for all 28 existing-engine `up_proj` layers. The `BOUNDED` component is required
because the execution context is synthetic, there is no power or clock control,
`IProfiler` does not expose CUDA kernel identity, and therefore h16816 coverage
remains `UNKNOWN`.

## Environment

| Field | Value |
| --- | --- |
| Host | `nvidia-desktop`, Jetson Orin Nano Engineering Reference Developer Kit Super |
| TensorRT | `10.3.0` |
| CUDA | `12.6` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| Local branch | `phase/04a-tensorrt-operator-attribution-recovery` |
| Local HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |
| Remote Jetson branch | `phase/03e-tensorrt-kernel-attribution` |
| Remote Jetson HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |

## Methodology

The existing Mixed Decode engine was deserialized and executed without being
modified or rebuilt. TensorRT `IExecutionContext.profiler` was attached to a
Python `IProfiler` implementation. `enqueue_emits_profile` was set to `false`,
and `report_to_profiler()` was called after every execution.

The probe used batch 1, cache length 8, position 8, and deterministic synthetic
Half tensors filled with `0.125` for `hidden_states`, `past_k*`, and `past_v*`.
It ran on a non-default `torch.cuda.Stream`, with 5 warmup and 30 measured
iterations. Every measured execution reported 699 TensorRT layers. The
representative logical target remained:

```text
C[1,3072] = A[1,1024] * B[1024,3072]
```

This was a narrow baseline probe, not a production benchmark. It did not use a
real token, real activation corpus, controlled clocks, or controlled power.

## Measurement source

The measured value is TensorRT `IProfiler` per-layer time, obtained through:

```text
execute_async_v3
  -> stream synchronization
  -> IExecutionContext.report_to_profiler()
  -> IProfiler.report_layer_time(layer_name, latency_ms)
```

`latency_us` in `up_proj_timing.csv` is the median of 30 per-layer measurements.
The CSV also records mean, sample standard deviation, coefficient of variation,
minimum, maximum, and nearest-rank-style P95.

`IProfiler` did not provide a CUDA kernel name, kernel-to-layer mapping, or
tactic name. Tactic strings in the CSV are inherited from Phase 4-A EngineInspector
evidence, not measured in this phase.

## up_proj execution evidence

All 28 expected `up_proj` layers were reported by `IProfiler`:

```text
22, 51, 74, 99, 124, 149, 174, 199, 223, 248, 273, 298, 323, 348, 372, 397,
422, 447, 472, 497, 522, 547, 572, 597, 622, 647, 672, 696
```

Coverage was therefore `28/28`. The exact layer-name suffixes matched the
EngineInspector layer IDs. This provides an existing-engine Layer -> Operator
timing evidence chain for `up_proj`, but not a runtime kernel identity.

The aggregate `up_proj` time over all 28 layers had mean `4682.253873 us` and
median `4339.215986 us` per measured decode execution. Because the run had no
power or clock control and used synthetic inputs, these values are a bounded
probe result, not a production baseline.

## Per-layer timing

The complete table is in [up_proj_timing.csv](up_proj_timing.csv). Key observations:

| Statistic over 28 layer medians | Value |
| --- | ---: |
| Minimum median latency | `153.280005 us` |
| Approximate range | `153.280005-159.759998 us` |
| Highest single-layer mean | `169.997867 us` |
| Lowest single-layer mean | `163.995733 us` |
| Typical per-layer CV | about `0.18-0.21` |

The individual per-layer CVs are high enough that the result must be treated as
bounded. The per-iteration aggregate also drifts from `3309.983991 us` minimum
to `7047.104046 us` maximum, with CV `0.192909`.

## Tactic distribution

The existing EngineInspector evidence identifies:

| Tactic family | Layers | Decoder layers |
| --- | ---: | --- |
| `f16f16_f16` | 25 | all except 8, 14, 27 |
| `f16f32_f32` | 3 | 8, 14, 27 |

In TensorRT layer IDs, the exceptions are `223`, `372`, and `696`. The
accumulator dtype, physical layout semantics, `tn_n` semantics, and alpha/beta
semantics remain `UNKNOWN`.

## h16816 coverage

**`UNKNOWN`.**

`IProfiler` reports TensorRT layer execution time but not CUDA kernel names.
Therefore this probe cannot prove that any measured `up_proj` layer used
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1`, the h16816 family, or any specific
tactic-kernel implementation.

The Phase 3-E h16816 evidence is also aggregate/shared evidence with
`operator_match=gate_proj;up_proj`. It cannot be split between those operators
and cannot be directly compared with this synthetic IProfiler probe.

## Comparison with Phase 3-E

Phase 3-E established that GEMM is the main GPU workload and that the rank-1
h16816 kernel is memory/L2-heavy, with memory/L2 at `97.06%`, HMMA active at
`39.673148%`, and achieved occupancy at `24.78%`. However, that evidence does
not identify a pure `up_proj` latency.

Phase 4-E now provides complete, bounded per-layer `up_proj` timing for one
existing engine execution context. It does not establish h16816 coverage, real
workload latency, or an optimization margin. The Phase 3-E aggregate and Phase
4-E IProfiler measurements differ in boundary, profiler mechanism, input
context, and operator sharing, so no direct latency ratio or efficiency ratio is
valid from the current evidence.

## Remaining unknowns

- Whether each `up_proj` execution maps to h16816 or another kernel family.
- Real-token `up_proj` latency under a production-like decode workload.
- Accumulator dtype.
- Physical operand/output layout and exact tactic semantics.
- Alpha/beta values and semantics.
- Whether the synthetic probe preserves the performance-relevant context of real
  activation and cache data.
- Whether a controlled-clock run would materially change the medians or CV.

## CUDA implementation readiness

**NOT READY.**

The missing piece was not merely a per-layer number: CUDA implementation would
require a clear improvement hypothesis and evidence that the existing TensorRT
path is not already near the relevant hardware limit. This probe adds a bounded
baseline, but it does not prove an optimization margin.

Specifically, h16816 coverage remains `UNKNOWN`, the runtime context is
synthetic, there is no controlled-clock comparison, and the Phase 3-E NCU
profile is memory/L2-heavy rather than an obvious compute-bound target. No CUDA
kernel or TensorRT Plugin work is justified from this report alone.

The next evidence step, if separately authorized, should correlate these exact
`up_proj` layer timings with existing or newly authorized NSYS/NCU kernel
evidence before any implementation decision.
