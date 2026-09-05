# Phase 5-B Step 2: TensorRT xmma GEMM vs cuBLASLt Microarchitectural Comparison

## Gate

**Phase 5-B Step 2: `CASE_A_SUPPORTED_BOUNDED` with the event-time gap
remaining `INCONCLUSIVE`.** The NCU comparison does not show the TensorRT
`f16f16_execute_kernel_trt` kernel taking longer than the cuBLASLt algorithm-21
kernel. The two kernels have distinct resource shapes, but near-equal profiled
duration. This is not proof that the earlier event-time gap is absent, and it is
not proof of a TensorRT kernel defect. No CUDA implementation is authorized.

## Question

For the frozen decode-only `up_proj` problem:

```text
M=1, K=1024, N=3072
FP16 input, FP16 output
Tensor Core path
```

what microarchitectural differences exist between:

1. the Mixed Decode TensorRT kernel
   `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt`; and
2. the direct cuBLASLt algorithm-21 baseline?

## Method

Both profiles used Nsight Compute `2024.3.1.0` with `--clock-control none`,
`--target-processes all`, and one captured launch. The following sections were
collected:

- `SpeedOfLight`
- `ComputeWorkloadAnalysis`
- `MemoryWorkloadAnalysis`
- `Occupancy`
- `WarpStateStats`
- `SchedulerStats`
- `LaunchStats`
- `SourceCounters`

The cuBLASLt harness was newly added as
[phase5b_step2_cublaslt_algo21_profile.cu](../../../experiments/Phase5-cuda-feasibility/src/phase5b_step2_cublaslt_algo21_profile.cu).
It was built and run only under Jetson `/tmp`. It fixed heuristic index `4`,
algorithm ID `21`, FP32 accumulate, workspace `0`, row-major layout, and the
frozen input distribution. It performed `100` warmup calls before the profiled
post-warmup launch. The smoke-run correctness was `PASS`.

The TensorRT side reused the persistent-context runtime and profiled one decode
step after the required prefill setup. Prefill kernels were not selected by the
NCU filter. No engine was rebuilt, no tactic was forced, and no runtime or
precision policy was changed.

## Captured Kernels

| Backend | NCU kernel symbol | Filter / capture |
| --- | --- | --- |
| Direct cuBLASLt algorithm 21 | `void Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_nn_align8>(Params)` | post-warmup launch after 100 warmups |
| Mixed Decode TensorRT | `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt` | exact-kernel filter, one launch |

The cuBLASLt symbol is a vendor kernel-name observation only. It does not prove
that TensorRT uses cuBLASLt. TensorRT backend identity remains `UNKNOWN`.

## NCU Metric Comparison

| Metric | cuBLASLt algorithm 21 | TensorRT f16f16 xmma | Difference |
| --- | ---: | ---: | ---: |
| Duration | `242.912 us` | `244.160 us` | `-1.248 us` |
| SM frequency | `305.83 MHz` | `305.75 MHz` | `+0.08 MHz` |
| Compute (SM) throughput | `34.147087%` | `25.515291%` | `+8.631796%` |
| Memory/L2 throughput | `76.06%` | `76.92%` | `-0.86%` |
| L1/TEX throughput | `21.97%` | `57.68%` | `-35.71%` |
| Tensor cycles active | `35.537047%` | `17.886626%` | `+17.650421%` |
| HMMA instruction active | `8.884262%` | `4.471656%` | `+4.412606%` |
| L2 hit rate | `0.546464%` | `3.161875%` | `-2.615411%` |
| Theoretical occupancy | `16.67%` | `25.00%` | `-8.33%` |
| Achieved occupancy | `12.95%` | `23.70%` | `-10.75%` |
| Registers per thread | `218` | `50` | `+168` |
| Shared memory per block | `82.944 KB` | `50.176 KB` | `+32.768 KB` |
| Grid size | `16` | `96` | `-80` |
| Waves per SM | `1` | `4` | `-3` |
| Long-scoreboard stall ratio | `7.140229` | `6.022843` | `+1.117386` |
| Wait stall ratio | `2.503743` | `0.871092` | `+1.632651` |
| Not-selected stall ratio | `0.050872` | `0.196999` | `-0.146127` |

Direct DRAM throughput remains `N/A` on this integrated platform and was not
estimated.

## Interpretation

Under NCU, both kernels ran at approximately `306 MHz` and had nearly equal
durations. The cuBLASLt algorithm-21 kernel used a smaller grid, much higher
register/shared-memory cost, lower occupancy, and higher tensor-cycle/HMMA
activity. The TensorRT kernel used more blocks, higher achieved occupancy, much
lower register/shared-memory cost, and lower tensor-cycle/HMMA activity. Both
were memory/L2-dominated and had long-scoreboard as the leading reported stall.

This is a resource-shape difference, not direct evidence that one kernel is
"worse". The TensorRT kernel's lower HMMA activity is offset by higher measured
occupancy and much lower per-thread resource pressure in this sample. Duration
and memory/L2 throughput are close.

## Event-Timing Boundary

The frozen standalone cuBLASLt CUDA-event median is `80.077961 us`. The
historical Phase 5-A correlated TensorRT xmma median is `118.800 us`, and the
steady-state correlated kernel median is `147.424 us`. These boundaries are not
directly comparable to NCU durations.

In this NCU run, both kernels were approximately `243-244 us` at about
`306 MHz`. Therefore NCU does not reproduce the event-timing environment. It
also does not prove that the earlier event-time gap comes from kernel execution
inefficiency. The event-time gap remains `INCONCLUSIVE`.

## Root-Cause Hypothesis

A bounded hypothesis is that the earlier `80 us` versus `147 us` observation
reflects an environment, clock-state, or integration/measurement boundary
difference rather than a proven TensorRT kernel deficiency. NCU supports the
narrower statement that the two kernels can exhibit nearly equal profiled
duration with different resource configurations. It does not prove the root
cause.

## Gate Judgement

```text
CASE_A_SUPPORTED_BOUNDED
EVENT_TIME_GAP_INCONCLUSIVE
NO_PROVEN_OPTIMIZATION_TARGET
TENSORRT_BACKEND_IDENTITY_UNKNOWN
NO_CUDA_IMPLEMENTATION_AUTHORIZED
```

## Evidence

- `ncu/cublaslt_algo21_postwarmup_details.csv`
- `ncu/cublaslt_algo21_postwarmup_raw.csv`
- `ncu/trt_f16f16_execute_kernel_trt_details.csv`
- `ncu/trt_f16f16_execute_kernel_trt_raw.csv`
- `cublaslt_algo21_ncu_stdout.log`
- `trt_f16f16_ncu_stdout.log`
- `environment.md`
- `gate_update.md`

Raw `.ncu-rep` files remain Jetson-local under
`/tmp/phase5b_step2_20260905T112513Z/ncu/` and are intentionally not committed.
