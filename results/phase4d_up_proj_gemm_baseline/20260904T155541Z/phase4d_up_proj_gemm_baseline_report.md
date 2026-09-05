# Phase 4-D up_proj GEMM Baseline Report

## Gate

**Phase 4-D: `PASS / BOUNDED`**

The exact logical GEMM, tensor source mapping, fusion status, standalone
library reference, and measurement protocol are now established. It is
`BOUNDED` because the standalone probe uses synthetic operands, the exact
backend kernel is `UNKNOWN`, and there is still no pure TensorRT `up_proj`
per-call baseline.

## Repository state

| Field | Verified value |
| --- | --- |
| Local branch | `phase/04a-tensorrt-operator-attribution-recovery` |
| Local HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |
| Tracked diff | none |
| Deleted files | none |
| Remote Jetson branch | `phase/03e-tensorrt-kernel-attribution` |
| Remote Jetson HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |
| Remote repository changed by probe | no |

The Jetson repository was not assumed to be synced with the local Phase 4-A
branch. The probe used an ephemeral Python script through SSH and did not modify
the remote repository.

## Exact GEMM definition

The target is decode-only `up_proj` FP16 GEMM:

```text
C[1,3072] = A[1,1024] * B[1024,3072]
```

Activation, weight, and output are Half. The ONNX graph has no transpose for
this MatMul and logical weight layout is `B[K,N]`. The accumulator dtype, tactic
layout semantics, physical memory layout, and alpha/beta values remain
`UNKNOWN`.

## Tensor source mapping

Layer 22 is the representative first-decoder-layer path:

| Tensor | Source / role |
| --- | --- |
| `__myln_k_arg__bb1_2691` | Half activation input `[1,1,1024]`, post-attention layernorm path |
| `__mye614249dconst` | Half TensorRT expanded constant `[1,1024,3072]`, from ONNX initializer shape `[1024,3072]` |
| `__myln_k_arg__bb1_2693` | Half output `[1,1,3072]` |

There is no bias input. Layer 22 is a pure GEMM candidate. Its output is
consumed by layer 23, a fusion layer containing `gate_proj` and a sigmoid path.
The same shape pattern applies to all 28 decoder layers; L8/L14/L27 use a
different tactic string.

## Fusion analysis

`up_proj` itself is not fused with activation, residual, or bias in the
representative EngineInspector GEMM layer. However, it is embedded in a fused
runtime graph: its output feeds the next `gate_proj` fusion layer. Therefore,
standalone extraction can isolate the logical GEMM, but it does not reproduce the
full runtime fusion context.

## Standalone baseline

Primary trusted-library reference, `torch.matmul` with contiguous
`A[1,1024]` and `B[1024,3072]` in Half:

| Metric | Value |
| --- | ---: |
| Median | `0.086364701 ms` |
| Mean | `0.086365059 ms` |
| Sample std | `0.000039214 ms` |
| CV | `0.045405%` |
| Trials | 7 |

Secondary `torch.nn.functional.linear` reference gives median
`0.083020223 ms`, CV `0.035288%`. Its physical layout differs from `B[K,N]`, so
it is secondary. Both used deterministic synthetic operands and neither loaded
real `up_proj` weights.

## TensorRT baseline

No pure TensorRT `up_proj` per-call baseline is available. The Phase 3-E
h16816 kernel is aggregate/shared evidence:

- Mixed steady-state time `39.590976 ms`.
- Share `26.781320%`.
- `operator_match=gate_proj;up_proj`.
- NCU mean duration `7601.557 us`.
- Memory/L2 `97.06%`, HMMA active `39.673148%`, achieved occupancy `24.78%`.

These values must not be relabeled as pure `up_proj` latency.

## Benchmark methodology

The standalone probe used CUDA Event timing around repeated trusted PyTorch
library calls, deterministic Half operands, calibration, warmup, 500 ms-target
measurement windows, and 7 trials per case. Allocation, checks, and
synchronization were outside the timed region. The rerun reached
`704.950 ms` warmup, slightly below the 1000 ms target, but produced low CV.
Full details are in [benchmark_protocol.md](benchmark_protocol.md).

## Comparison

The standalone reference and TensorRT baseline are not directly comparable as
latency numbers. They differ in implementation, operand provenance, launch
context, layout handling, fusion, and measurement boundary. The standalone value
is a reference point, not proof that TensorRT is inefficient.

## Optimization hypothesis

Possible future directions remain memory access, tiling, fusion, and launch
overhead, but Phase 4-D does not prove that any of them will improve the real
runtime. The NCU memory/L2-heavy profile is suggestive, not an optimization
target proof. No custom kernel should be started from this result alone.

## Implementation readiness

**NOT READY** for CUDA implementation authorization.

The exact logical GEMM, correctness oracle, benchmark protocol, and standalone
reference exist, but the expected improvement hypothesis is not established and
the pure TensorRT per-call baseline is still missing.

## Remaining unknowns

- Pure TensorRT `up_proj` per-call latency.
- Whether every `up_proj` invocation uses h16816.
- Accumulator dtype.
- Physical TensorRT operand/output layout.
- Tactic `tn_n` semantics.
- Alpha/beta values and semantics.
- Exact PyTorch backend kernel identity in the standalone probe.
- Real-weight correctness and latency behavior.

## Recommendation

The next authorized step should be a narrow TensorRT-side baseline probe that
isolates one exact `up_proj` GEMM execution without modifying the engine or
runtime. Only after that probe and an explicit expected-improvement hypothesis
should CUDA implementation be considered.
