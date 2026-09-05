# Phase 4-F up_proj Kernel Attribution Refinement Report

## Gate

**Phase 4-F: `PASS / BOUNDED`**

The `PASS` component is supported by direct runtime evidence from the existing
Phase 3-C Mixed persistent SQLite timeline: all 28 observed `up_proj` NVTX
ranges correlate through the NSYS NVTX -> ONNX -> EngineInspector -> TensorRT
layer chain to observed CUDA kernel launches.

The `BOUNDED` component is required because the result is limited to the
observed historical trace, every observed range contains more than one kernel
name, the SQLite kernel events do not directly carry a TensorRT layer ID, and
future invocation behavior is `UNKNOWN`.

## Environment

| Field | Value |
| --- | --- |
| Host | `nvidia-desktop`, Jetson Orin Nano Engineering Reference Developer Kit Super |
| TensorRT | `10.3.0` |
| CUDA | `12.6` |
| Nsight Systems | `2024.5.4.34-245434855735v0` |
| SQLite | `3.37.2` |
| Local branch | `phase/04a-tensorrt-operator-attribution-recovery` |
| Local HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |
| Remote branch | `phase/03e-tensorrt-kernel-attribution` |
| Remote HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |

No new profiling, benchmark, engine rebuild, engine modification, ONNX export,
CUDA kernel work, TensorRT Plugin work, quantization change, runtime architecture
change, or tactic replacement was performed.

## Evidence sources

The study reused existing artifacts:

| Artifact | Role |
| --- | --- |
| `results/phase4a_operator_attribution/20260904T131524Z/mixed_decode_engine_inspector.json` | TensorRT layer metadata |
| `results/phase4a_operator_attribution/20260904T132820Z/mapping.csv` | TensorRT layer -> ONNX node -> `up_proj` mapping |
| `results/phase4a_operator_attribution/20260904T134556Z/runtime_nvtx_kernel_mapping.csv` | Prior NVTX aggregate mapping |
| `results/phase4e_up_proj_baseline/20260904T161320Z/` | Existing `IProfiler` layer timing capability evidence |
| `/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite` | Existing raw NSYS SQLite timeline |
| `/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.nsys-rep` | Existing raw NSYS report |

The raw artifacts were hashed in `preflight.json` and remain unchanged on Jetson.

## Existing correlation capability

The existing SQLite trace supports a stronger form of correlation than the
Phase 4-A.2 aggregate summary:

```text
NVTX_EVENTS.textId -> StringIds.value
NVTX_EVENTS.globalTid + time containment -> CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_RUNTIME.correlationId -> CUPTI_ACTIVITY_KIND_KERNEL.correlationId
```

The raw kernel event does not directly contain a TensorRT layer ID. Therefore,
kernel identity becomes a TensorRT-layer identity only after joining the observed
NVTX range to the exact ONNX node and then to the Phase 4-A EngineInspector
metadata.

`IProfiler` remains useful for layer timing but is not a kernel attribution
source. Phase 4-E already demonstrated that it reports layer names and times but
does not expose CUDA kernel identity.

## Kernel-layer mapping result

The direct raw SQLite query found:

| Metric | Value |
| --- | ---: |
| Unique `up_proj` NVTX ranges | `28` |
| Unique `up_proj` TensorRT layers | `28` |
| Correlated kernel launch rows | `196` |
| Kernel launches per range | `7` |
| Ranges with more than one kernel name | `28` |
| Ranges with h16816-family launch | `28` |
| Ranges with `sm80_xmma_gemm_*` launch | `28` |

The observed kernel-name counts are:

| Kernel name | Launch rows |
| --- | ---: |
| `trt_ampere_h16816gemm_128x64_ldg8_nn_v1` | `78` |
| `trt_ampere_h16816gemm_256x64_ldg8_tn_v1` | `6` |
| `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt` | `100` |
| `sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt` | `12` |

Layer IDs are `22, 51, 74, 99, 124, 149, 174, 199, 223, 248, 273, 298, 323,
348, 372, 397, 422, 447, 472, 497, 522, 547, 572, 597, 622, 647, 672, 696`.
The per-range mapping and durations are recorded in
[correlation_analysis.json](correlation_analysis.json).

## Coverage

The h16816 family is present in all `28/28` observed `up_proj` NVTX ranges, but
it does **not** cover all observed `up_proj`-linked kernel launches:

```text
h16816-family launch rows:      84 / 196
sm80_xmma_gemm_* launch rows:  112 / 196
```

This is a launch-level result from one bounded trace. It must not be rewritten as
`28/28 h16816 invocation coverage`. Whether one NVTX range corresponds exactly
to one logical `up_proj` invocation remains `UNKNOWN`.

Phase 4-E's `h16816_coverage=UNKNOWN` is therefore refined to a bounded result:
h16816 is observed in every observed range, but non-h16816
`sm80_xmma_gemm_*` kernels are also observed in every observed range.

## Unknown cases

- Whether future invocations use the same kernel distribution as this trace.
- Whether each NVTX range maps one-to-one to one logical `up_proj` invocation.
- Why each observed range contains both h16816 and `sm80_xmma_gemm_*` kernels.
- Accumulator dtype and exact accumulator semantics.
- Physical operand/output layout and `tn_n` semantics.
- Alpha/beta values and semantics.
- FP16 persistent-trace correlation, real-token production distribution, and
  non-Mixed runtime behavior.
- Whether debug-listener, debug-state, or callback APIs can improve
  kernel-to-layer attribution; their presence was observed, but they were not
  exercised.

## Toolchain limitation

The raw CUPTI kernel events expose kernel identity and launch metadata but no
direct TensorRT layer ID or operator identity. TensorRT `IProfiler` exposes layer
timing but no kernel identity. The current reliable bridge is therefore a
multi-source correlation, not a single-tool mapping.

The correlation depends on the existing NVTX source-path range names. Internal
`__myl_*` kernels without an exact ONNX node path remain `UNKNOWN`, and no
operator identity was inferred from a kernel or tactic name alone.

## CUDA readiness assessment

**NOT READY.**

Phase 4-F establishes a bounded observed-launch attribution chain for `up_proj`,
but it still lacks:

1. a reliable per-logical-invocation kernel distribution;
2. a proven optimization margin against the existing Tensor Core path;
3. controlled-clock and production-workload evidence;
4. exact GEMM split/layout/accumulator semantics;
5. proof that replacing the existing path would preserve numerics and beat
   TensorRT's current kernels.

The fact that h16816 is not the only observed kernel is attribution evidence, not
by itself a CUDA optimization opportunity.

## Recommendation

Do not start CUDA kernel or TensorRT Plugin implementation from this result.

If separately authorized, the next evidence step should be a narrowly bounded
per-invocation feasibility study that resolves the NVTX-range-to-logical-invocation
boundary and validates whether the mixed h16816/`sm80_xmma_gemm_*` pattern is
stable for real decode workloads. Only after that should target selection consider
whether an implementation hypothesis exists.
