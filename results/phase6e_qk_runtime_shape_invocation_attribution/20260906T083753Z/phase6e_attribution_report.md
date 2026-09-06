# Phase 6-E QK Runtime Shape And Invocation Attribution Report

## Executive Summary

Phase 6-E executed one read-only, frozen-engine controlled run on Jetson. The
run used the historical Mixed runtime stack, persistent execution contexts,
sample `eva_025`, one warmup prefill at sequence length 8, one steady prefill
at sequence length 8, and decode steps 0-3. It then correlated direct
TensorRT I/O shapes with the Nsight Systems `NVTX -> runtime API -> kernel`
chain.

The final gate is:

```text
QK_H16816_PATH_NOT_REPRODUCED
```

Phase 6-E status is `PASS / BOUNDED`. The controlled decode invocations
recovered direct cache-shape progression and one `/MatMul` range per decode
engine invocation, but all four correlated kernels were the xmma family.
Zero `trt_ampere_h16816gemm_128x64_ldg8_nn_v1` launches were observed.
Therefore the historical h16816 path was not reproduced, workload comparison
remains `UNKNOWN`, and no performance ratio or implementation is authorized.

## Run Boundary

The run was a `NEW_CONTROLLED_RUN`, not a byte-for-byte reproduction. The
remote raw directory was:

```text
/tmp/phase6e_qk_20260906T092107Z/
```

The three executions were:

1. One no-Nsys validation run under
   `/tmp/phase6e_qk_20260906T092107Z/harness_no_nsys_retry/`.
2. One Nsys run under `/tmp/phase6e_qk_20260906T092107Z/harness_nsys/`.
3. One exported read-only Nsys SQLite database under
   `/tmp/phase6e_qk_20260906T092107Z/phase6e_qk.sqlite`.

The committed evidence is derived and does not include the raw `.nsys-rep` or
SQLite database. `historical_run_configuration.md` records the frozen engine
hashes, input selection, forced tokens, runtime lifetime, and historical trace
boundary. `environment.md` records Orin SM 8.7, TensorRT 10.3.0, CUDA 12.6,
PyTorch 2.5.0a0+872d972e41.nv24.08, Nsys 2024.5.4.34, and 25 W power mode.

The direct runtime log contains 776 I/O shape rows: 92 warmup prefill, 92
steady prefill, and 592 decode rows. This is direct
`IExecutionContext` evidence sampled after execution and stream
synchronization.

## Runtime Shape Evidence

The direct Mixed decode cache progression is:

| Step | Past K0 shape | Present K0 shape | Past KV length | Key length |
| ---: | --- | --- | ---: | ---: |
| 0 | `[1,8,8,128]` | `[1,8,9,128]` | 8 | 9 |
| 1 | `[1,8,9,128]` | `[1,8,10,128]` | 9 | 10 |
| 2 | `[1,8,10,128]` | `[1,8,11,128]` | 10 | 11 |
| 3 | `[1,8,11,128]` | `[1,8,12,128]` | 11 | 12 |

These cache and hidden shapes are `DIRECT_RUNTIME_EVIDENCE`. Semantic Q, K,
K-transpose, and QK output tensors are not direct engine I/O. They are derived
for layer 0 from the proven static QK graph identity plus the direct cache
progression, and are therefore `DERIVED_FROM_PROVEN_STATE`:

| Step | Derived Q | Derived K | Derived K^T | Derived output | Per-head GEMM |
| ---: | --- | --- | --- | --- | --- |
| 0 | `[16,1,128]` | `[16,9,128]` | `[16,128,9]` | `[16,1,9]` | `1 x 9 x 128` |
| 1 | `[16,1,128]` | `[16,10,128]` | `[16,128,10]` | `[16,1,10]` | `1 x 10 x 128` |
| 2 | `[16,1,128]` | `[16,11,128]` | `[16,128,11]` | `[16,1,11]` | `1 x 11 x 128` |
| 3 | `[16,1,128]` | `[16,12,128]` | `[16,128,12]` | `[16,1,12]` | `1 x 12 x 128` |

The effective per-head work is `M=1`, `N=key_length`, `K=128`, repeated across
16 Q-heads with GQA repeat factor 2. The recorded FP16 total MAC counts are
18,432 / 20,480 / 22,528 / 24,576 for steps 0-3. Kernel argument identity
remains `UNKNOWN`; derived shapes do not become direct kernel arguments.

## Kernel Correlation

Every decode invocation exposed exactly one `/MatMul` NVTX range. Each range
correlated through one runtime API record to exactly one CUDA kernel. The
observed full kernel name was:

```text
sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_aligna2_alignc2_execute_kernel_trt
```

The four launches used grid `1x1x16`, block `128x1x1`, and stream 7. Their
NSYS durations were:

| Step | Engine invocation | Correlation ID | Duration ns |
| ---: | --- | ---: | ---: |
| 0 | `PHASE6E_mixed_decode_0001` | 11293 | 31,648 |
| 1 | `PHASE6E_mixed_decode_0002` | 15157 | 31,488 |
| 2 | `PHASE6E_mixed_decode_0003` | 18665 | 31,424 |
| 3 | `PHASE6E_mixed_decode_0004` | 22173 | 32,384 |

The count by family across the four controlled layer-0 QK launches is:

| Kernel family | Count |
| --- | ---: |
| `sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_..._execute_kernel_trt` | 4 |
| `trt_ampere_h16816gemm_128x64_ldg8_nn_v1` | 0 |
| Other | 0 |

These durations are observational NSYS kernel durations for the new xmma
path. They are not a benchmark mean, are not correctness-normalized, and must
not be compared to historical h16816 durations.

## Same-Workload Decision

The historical h16816-versus-xmma pair remains `UNKNOWN` because historical
NSYS did not expose runtime tensor shapes, kernel arguments, or sufficient
invocation identity. The historical h16816 path versus the new controlled
xmma path also remains `UNKNOWN`: one side has only historical invocation
evidence while the other has direct invocation and derived-shape evidence.

There is no `EXACT_SAME_WORKLOAD` or `COMPARABLE_WORKLOAD` classification.
Same static layer-0 QK semantic identity, the same forced-token sequence, and
similar cache progression are insufficient because the historical h16816
runtime shape and operation identity remain unknown.

## Path Trigger

The controlled run did not observe the historical h16816 path at decode steps
0-3 with past KV lengths 8-11 and key lengths 9-12. All four eligible `/MatMul`
invocations selected xmma. This is direct runtime evidence that the historical
h16816 path was not reproduced under the controlled boundary, but it is not
sufficient to identify a transition trigger. The trigger remains
`UNKNOWN`.

This result does not invalidate the Phase 6-B/6-C historical evidence. It
downgrades optimization readiness because the target path cannot be reproduced
or isolated in the current frozen controlled setup.

## Performance Comparison

Normalized performance is `NOT_CALCULATED`.

| Comparison | Classification | Allowed? |
| --- | --- | --- |
| Historical h16816 versus historical xmma | `UNKNOWN` | No raw duration ratio |
| Historical h16816 versus Phase 6-E xmma | `UNKNOWN` | No raw duration ratio |
| New controlled xmma step durations | Observational NSYS only | No optimization claim |

No tactic defect is claimed. No effective bandwidth, achieved occupancy, or
DRAM throughput is inferred. The four new durations are reported only as
kernel timestamps under the stated profiling boundary.

## Optimization Surface

The result provides a clean negative attribution boundary: the selected
historical h16816 path is absent from the controlled workload that otherwise
matches the historical engine, runtime, sample, forced tokens, persistent
context lifetime, prefill length, and decode sequence as closely as the
frozen setup allows.

There is still no proven replacement surface. Derived QK GEMM shapes are not
kernel-argument evidence. The workload comparison is `UNKNOWN`. The trigger is
`UNKNOWN`. Consequently, the prior Phase 6-D Rank-1 candidate must be
downgraded from an active attribution target to a bounded, unresolved path
attribution.

## Authorization State

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
Engine rebuild: NOT AUTHORIZED
ONNX change: NOT AUTHORIZED
Precision change: NOT AUTHORIZED
Tactic forcing: NOT AUTHORIZED
NCU: NOT AUTHORIZED
```

No historical result was modified or overwritten. Raw Nsys and SQLite evidence
remained outside Git.
