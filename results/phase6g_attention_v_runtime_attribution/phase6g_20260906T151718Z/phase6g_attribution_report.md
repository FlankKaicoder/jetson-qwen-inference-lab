# Phase 6-G Attention x V Runtime/Kernel Attribution Report

## Executive Summary

Phase 6-G recovered a complete, bounded runtime/kernel attribution surface for
decode Attention x V across all 28 decoder layers using the existing Phase 3-C
Mixed persistent Nsys SQLite. No new profiling, benchmark, inference, engine
build, implementation, or precision change occurred.

The final gate is:

```text
AV_RUNTIME_SURFACE_RECOVERED
```

This establishes an attribution surface only. It does **not** prove optimization
headroom, feasibility, or benefit. The optimization-surface answer is
`NOT PROVEN`.

## Source And Method

The raw source was verified before querying:

```text
/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite
bytes = 3477504
sha256 = ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8
```

A remote Python process opened the file with
`file:<path>?mode=ro` and SQLite URI mode. The script read Phase 3-C NVTX
boundaries, AV NVTX ranges, contained runtime APIs, and kernels joined by
`correlationId`. Semantic identity came only from Phase 6-A artifacts. The
TensorRT decode-layer identity came from Phase 6-A runtime-context evidence.

The trace stores phase boundaries such as `PHASE3B_STEADY_DECODE_STEP_0`
directly in `NVTX_EVENTS.text`, while decode AV ranges such as `/MatMul_1` are
in StringIds. The query read both storage paths with `COALESCE`.

## Semantic And Runtime Evidence

Phase 6-A provides HIGH semantic identity for the 28 odd decode MatMul nodes:

```text
P6A_002, P6A_004, ..., P6A_056
/MatMul_1, /MatMul_3, ..., /MatMul_55
Attention_x_V
```

The Phase 6-G query recovered exactly 112 AV NVTX instances:

| Check | Result |
| --- | ---: |
| AV NVTX instances | 112 |
| Unique AV candidates/layers | 28 |
| Instances per candidate | 4 |
| Runtime rows contained in each AV NVTX | 112 total |
| Correlated kernels | 112 total |
| Unique correlation IDs | 112 |
| Launch APIs | 112 `cuLaunchKernelEx` |
| Distinct TensorRT layers | 28 |
| Distinct kernels | 1 |

Every instance has exactly one correlated kernel:

```text
sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize64x128x32_stage5_warpsize2x2x1_tensor16x8x16_aligna2_alignc2_execute_kernel_trt
```

The recorded kernel family is:

```text
sm80_xmma_gemm_f16f16_f16f32_f32
```

The per-instance chain is therefore:

```text
Phase 6-A semantic Attention x V
-> Phase 3-C AV NVTX range
-> contained cuLaunchKernelEx
-> correlationId
-> xmma GEMM kernel
```

Representative examples of TensorRT runtime-layer identity are
`/MatMul_1_myl0_18`, `/MatMul_3_myl0_47`, and `/MatMul_5_myl0_70`. All 28
layers are enumerated in `attention_v_kernel_mapping.csv`.

## Representative Boundary Separation

Boundary assignment required both the kernel and its originating NVTX range to
be fully contained in the same observed boundary. No instance was ambiguous.

| Boundary | AV instances | AV kernels | AV duration |
| --- | ---: | ---: | ---: |
| `PHASE3B_INIT` | 0 | 0 | `0 ns` |
| `PHASE3B_WARMUP` | 0 | 0 | `0 ns` |
| `PHASE3B_STEADY_PREFILL_S8` | 0 | 0 | `0 ns` |
| `PHASE3B_STEADY_DECODE_STEP_0` | 28 | 28 | `1,202,560 ns` |
| `PHASE3B_STEADY_DECODE_STEP_1` | 28 | 28 | `1,448,480 ns` |
| `PHASE3B_STEADY_DECODE_STEP_2` | 28 | 28 | `750,496 ns` |
| `PHASE3B_STEADY_DECODE_STEP_3` | 28 | 28 | `836,032 ns` |

The five representative steady ranges sum to:

```text
112 instances
112 kernels
4,237,568 ns
2.866503% of 147,830,560 ns
```

There are zero AV instances in `PHASE3B_WARMUP` and
`PHASE3B_STEADY_PREFILL_S8`. This is consistent with the frozen Phase 6-A
finding that prefill Attention x V is fused into `_gemm_mha_v2_*`, while decode
exposes standalone `/MatMul_*` layers. Thus the historical AV aggregate is not
warmup dominated.

## Historical All-Trace Reconciliation

The same 112 recovered instances also reproduce the historical all-trace
aggregate:

```text
4,237,568 ns
1.834842% of 230,950,048 ns
```

This is a different denominator from the representative steady result above:

```text
4,237,568 ns / 147,830,560 ns = 2.866503%
```

The raw duration is identical only because every observed AV instance already
falls inside one of the four representative decode boundaries. The percentages
are not interchangeable and are reported with their own denominators.

## Required Answers

| Question | Answer |
| --- | --- |
| Q1: Is there an independent AV runtime surface? | `PROVEN` for the decode standalone runtime chain |
| Q2: Do all 28 layers have runtime evidence? | `YES`, 28/28 candidates have a TensorRT layer, NVTX range, and kernel |
| Q3: Is the contribution steady significant or historical aggregate only? | Representative steady contribution is `4,237,568 ns / 2.866503%`; the all-trace share is `1.834842%` |
| Q4: Is kernel identity known? | `KNOWN`: one FP16 xmma GEMM kernel name and one recorded family |
| Q5: Is there a clean optimization surface? | `NOT PROVEN` |

## Limits

- Kernel arguments, exact GEMM workload shape at execution time, tactic identity
  as a numeric identifier, and CUDA backend identity remain `UNKNOWN`.
- The recovered surface does not prove a bottleneck, headroom, replacement
  benefit, or TensorRT tactic defect.
- The 4-step decode window is the existing representative boundary. Extrapolation
  to other sequence lengths or longer decoding is not authorized by this
  evidence.
- `HIGH` confidence applies to the direct semantic-to-NVTX-to-runtime-to-kernel
  attribution chain, not to optimization feasibility.

## Evidence Files

- `attention_v_runtime_attribution.csv`: 112 per-instance rows.
- `attention_v_kernel_mapping.csv`: 28 layer/runtime/kernel aggregate rows.
- `attention_v_boundary_analysis.csv`: boundary split and denominator-bounded shares.
- `evidence_manifest.csv`: source and method artifacts with hashes.
- `raw_attention_v_events.jsonl`: derived raw-event export from read-only SQLite.
