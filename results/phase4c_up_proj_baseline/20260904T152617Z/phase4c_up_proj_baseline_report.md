# Phase 4-C up_proj Optimization Baseline Report

## Gate

**Phase 4-C: `PASS / BOUNDED`**

The exact target, logical GEMM definition, 28-layer shape audit, correctness
oracle, benchmark protocol, and success criteria are frozen. The gate is
`BOUNDED`, not `PASS / PROVEN_BASELINE`, because no pure `up_proj` per-call
microbenchmark baseline exists yet. The observed h16816 NCU latency is
shared-kernel/profiler-context evidence, not a pure target baseline.

## Repository State

| Field | Verified value |
| --- | --- |
| Branch | `phase/04a-tensorrt-operator-attribution-recovery` |
| Starting HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |
| Last commit | `phase3e: close compute efficiency audit` |
| Tracked diff | none |
| Deleted files | none |
| Existing untracked dirs | preserved |
| Commit / push in Phase 4-C | none |

This phase was read-only evidence recovery and protocol design on Windows. No
Jetson execution was performed in Phase 4-C.

## Target Definition

The frozen target is the **decode-only `up_proj` FP16 GEMM path** in the
existing Mixed Decode TensorRT engine. The 28 decoder layers are selected by
Phase 4-A.1 HIGH evidence and Phase 4-A.2 HIGH runtime mapping. The target is a
research and measurement target only; no implementation is authorized by this
phase alone.

TensorRT layer IDs are:

```text
22, 51, 74, 99, 124, 149, 174, 199, 223, 248, 273, 298, 323, 348, 372,
397, 422, 447, 472, 497, 522, 547, 572, 597, 622, 647, 672, 696
```

## Exact GEMM Problem

The logical computation is:

```text
C[M,N] = A[M,K] * B[K,N]
M = 1
K = 1024
N = 3072
C shape = [1,1,3072]
```

| Field | Value | Evidence |
| --- | --- | --- |
| Activation logical shape | `[1,1,1024]` | Mixed Decode EngineInspector |
| Weight logical shape | `[1024,3072]` | ONNX initializer / Phase 4-A evidence |
| Inspector expanded constant shape | `[1,1024,3072]` | EngineInspector |
| Output logical shape | `[1,1,3072]` | EngineInspector |
| Activation dtype | Half | EngineInspector |
| Weight dtype | Half | EngineInspector |
| Output dtype | Half | EngineInspector |
| Accumulator dtype | `UNKNOWN` | No existing metadata proves it |
| ONNX transpose | none | ONNX graph |
| Logical weight layout | `B[K,N]` | ONNX MatMul shape semantics |
| Inspector alpha/beta inputs | Float, shape `[1]` | EngineInspector |
| Alpha/beta values and semantics | `UNKNOWN` | Not recoverable from committed metadata |
| Tactic layout label | `tn_n` | Inspector tactic string |
| NN/NT/TN/TT semantics | `UNKNOWN` | Tactic label alone is not layout proof |
| Physical memory layout | `UNKNOWN` | Not captured by committed evidence |
| Decode dynamic dimensions | none for decode input | Mixed Decode static shape |

## 28-Layer Shape Audit

Result: **BOUNDED UNIFORM**.

All 28 layers have the same logical activation, weight, expanded constant, and
output shapes, and all are `Half/Half/Half`. All 28 are HIGH attributed to
`up_proj`. Tactic strings are not fully uniform:

- 25 layers use `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_...`.
- Layers 8, 14, and 27 use `sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_...`.

The tactic difference must not be used to infer accumulator dtype or actual
execution semantics. The complete row-level audit is in
[shape_audit.csv](shape_audit.csv).

## TensorRT Baseline

Baseline A is the existing TensorRT GEMM path for the frozen decode `up_proj`
problem. The committed evidence identifies the Phase 3-E top GEMM family:

- Kernel: `trt_ampere_h16816gemm_128x64_ldg8_nn_v1`
- Mixed steady-state kernel time: `39.590976 ms`
- Share of Mixed steady-state kernel time: `26.781320%`
- `operator_match`: `gate_proj;up_proj`

This time cannot be split between `gate_proj` and `up_proj`. Phase 4-A.2 also
records multiple kernel families for `up_proj`, so the probe must not assume
that every `up_proj` invocation uses h16816. The exact pure `up_proj`
per-call baseline is therefore `UNKNOWN` and requires a future narrowly
authorized baseline probe.

## Runtime Baseline

Baseline B is the Phase 3-C persistent Mixed runtime. It is an end-to-end
runtime baseline, not a pure `up_proj` baseline.

| Field | Value |
| --- | --- |
| Runtime | `mixed_persistent` |
| Batch / decode | batch 1, 8-step decode window evidence |
| Warmup / repeats | 5 / 10 |
| Timing | `time.perf_counter` with CUDA synchronization |
| S=8 decode TPOT median | `43.411762 ms` |
| S=8 decode TPOT CV | `9.1712839%` |
| S=8 8-step decode window median | `365.059806 ms` |
| S=8 window CV | `0.3611726%` |
| S=16 decode TPOT median | `43.173353 ms` |
| S=16 decode TPOT CV | `0.7028%` |
| Persistent context counters | 5 creations, 448 reuses |

## NCU Evidence

Phase 3-E rank-1 h16816 evidence, mean of three launches:

| Metric | Value |
| --- | ---: |
| Duration mean | `7601.557 us` |
| Duration range | `7600.128-7603.488 us` |
| Memory/L2 throughput | `97.06%` |
| SM throughput | `39.57%` |
| HMMA pipe active | `39.673148%` |
| Achieved occupancy | `24.78%` |
| Theoretical occupancy | `25.00%` |
| Direct DRAM throughput | `N/A` |

The evidence indicates a memory/L2-heavy profile for the shared kernel under
NCU. It does not prove that a custom `up_proj` kernel will be faster, and it
does not provide a pure per-call baseline.

## Correctness Oracle

The future implementation must compare `C_custom` against an FP32 accumulation
reference using Half-quantized operands, analogous to Exp04 Track A. The
inherited gate is:

```text
max_abs_error <= 1e-3 + 1e-4 * max_abs_reference
```

Required metrics include finite checks, shape equality, guard/CUDA status,
`max_abs_error`, and `max_rel_error`. This gate is inherited for standalone
GEMM implementation correctness and is `PROPOSED` for the real `up_proj` tensors
until measured. Details are in
[correctness_protocol.md](correctness_protocol.md).

## Benchmark Protocol

The microbenchmark must inherit Exp04 CUDA Event methodology: about 1000 ms
warmup, adaptive about 500 ms measurement window with 2x safety factor, at
least seven trials, mean/median/sample std/CV/min/max, and exclusion of all
non-kernel work from timing. The future baseline probe must first measure the
existing path because no pure per-call baseline is committed.

Runtime integration is separate and must use the persistent runtime with
same-session paired comparison. A microbenchmark win must never be reported as
a runtime win. Details are in [benchmark_protocol.md](benchmark_protocol.md).

## Success Criteria

Future optimization success requires all five gates:

1. Target validity: exact decode-only `M=1,K=1024,N=3072` `up_proj` FP16 path.
2. Correctness: all inherited correctness checks PASS.
3. Microbenchmark performance: controlled median improvement with confidence bounds excluding zero.
4. Measurement stability: sufficient trials and reported variability.
5. Runtime integration: no end-to-end decode regression after explicit authorization.

A microbenchmark-only result cannot produce a `PROVEN` optimization claim.
Machine-readable conditions are in
[success_criteria.json](success_criteria.json).

## Remaining Unknowns

- Pure `up_proj` per-call latency.
- Accumulator dtype.
- Physical operand/output layout.
- TensorRT internal semantics of the `tn_n` tactic label.
- Alpha/beta values and semantics.
- Whether every `up_proj` invocation uses h16816.
- Real-`up_proj` measured suitability of the inherited tolerance.

Each is recorded as `UNKNOWN` or `PROPOSED`; no value is inferred.

## Implementation Readiness

The project is ready to design a narrowly scoped exact-baseline and correctness
probe. It is **not** ready to claim an optimization target is proven, and
CUDA/CUTLASS/Plugin implementation is not authorized by Phase 4-C alone.

## Next Recommendation

If the owner explicitly authorizes the next step, run a narrow baseline probe
under the frozen correctness and benchmark protocols: measure the existing
path for the exact GEMM problem, capture the executed kernel subset, and freeze
the pure per-call baseline before considering implementation. Do not start
CUDA implementation, CUTLASS work, or Plugin development from this report.
