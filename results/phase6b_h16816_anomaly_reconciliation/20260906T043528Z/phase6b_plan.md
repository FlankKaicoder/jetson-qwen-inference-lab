# Phase 6-B Plan: h16816 Historical Anomaly Reconciliation

## Objective

Explain the Phase 6-A suspicious row:

```text
myelin-exec:/MatMul
trt_ampere_h16816gemm_128x64_ldg8_nn_v1
7 instances
54,984,352 ns
```

This is an attribution reconciliation phase. It is not a CUDA optimization,
Attention feasibility, or FlashAttention phase.

## Starting State

| Field | Value |
| --- | --- |
| Branch at work start | `phase/06b-h16816-anomaly-reconciliation` |
| HEAD at work start | `dfdb0de64f26198caf13433f9448f476c4fe3691` |
| Tracked working tree | Clean |
| New Jetson execution | None |
| New profiling or benchmark | None |
| Engine, ONNX, precision, or tactic change | None |

Protected untracked directories are retained untouched:

```text
experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T203103Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040200Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040300Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040400Z/
```

## Method

1. Re-read frozen Phase 6-A and Phase 4-A.2 derived artifacts.
2. Re-read the frozen Phase 3-C Mixed persistent NSYS SQLite in read-only URI
   mode over SSH.
3. Join exact `/MatMul` NVTX ranges to contained CUDA runtime launches and then
   to exactly one kernel by `correlationId`.
4. Map the NVTX source path and launch to the static decode EngineInspector
   layer `/MatMul_myl0_15`.
5. Compare launch shape and context against historical h16816 evidence.
6. Evaluate real-attention, misattribution, and measurement-artifact
   hypotheses without changing the experiment definition.

No new profiling is needed. Historical evidence distinguishes a real seven-launch
aggregate from duplicate counting or an aggregation artifact. It also establishes
direct runtime ownership. It does not establish a clean optimization surface.

## Hypotheses

| ID | Hypothesis |
| --- | --- |
| H1 | The seven launches are real directly attributed attention MatMul runtime. |
| H2 | The kernel time is real, but `/MatMul` ownership is invalid. |
| H3 | The aggregate is duplicate, overlapping, badly grouped, or otherwise a measurement artifact. |
| H4 | Existing evidence is insufficient to distinguish H1 through H3. |

## Required Gate

The only allowed gate choices are:

```text
H16816_REAL_ATTENTION_HOTSPOT_RECOVERED
H16816_MISATTRIBUTION_RECOVERED
H16816_REAL_BUT_OPTIMIZATION_SURFACE_UNRESOLVED
H16816_IDENTITY_UNRESOLVED
```

No implementation may start in Phase 6-B. Custom CUDA, FlashAttention, TensorRT
Plugin, engine rebuild, ONNX change, precision change, and tactic forcing remain
forbidden.
