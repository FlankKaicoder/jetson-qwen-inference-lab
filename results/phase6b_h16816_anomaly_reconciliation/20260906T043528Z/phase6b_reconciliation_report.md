# Phase 6-B h16816 Anomaly Reconciliation Report

## Executive Summary

The suspicious Phase 6-A row is real, exactly reconstructable, and is not a
duplicate aggregate or timing artifact. It consists of seven distinct
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1` launches summing exactly to
`54,984,352 ns`. Each launch has direct runtime correlation to an exact
`myelin-exec:/MatMul` NVTX range. The static decode layer is
`/MatMul_myl0_15`, decoder layer 0, and Phase 6-A establishes its ONNX semantic
identity as QK^T with HIGH confidence.

The final gate is therefore:

```text
H16816_REAL_BUT_OPTIMIZATION_SURFACE_UNRESOLVED
PASS / BOUNDED
```

H1 is supported in a bounded sense: the time is real and directly owned by
`/MatMul`. H2 and H3 are rejected. Gate C rather than Gate A is selected because
the runtime tactic differs from the static inspector tactic, the exact CUDA
implementation and layout semantics remain UNKNOWN, and no clean replacement
surface is proven.

## Source Recovery

The committed historical source is
`results/phase4a_operator_attribution/20260904T134556Z/runtime_nvtx_kernel_mapping.csv`,
line 2. It records exact NVTX `/MatMul`, TRT layer 15
`/MatMul_myl0_15`, seven instances, `54,984,352 ns`, and historical confidence
`MEDIUM`.

The companion tactic-consistent row records four
`sm80_xmma_gemm_f16f16_f16f32_f32...` launches totaling `75,840 ns`.

## Seven-Instance Reconstruction

The raw SQLite is
`/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite`, read in
read-only URI mode. Its SHA-256 is
`ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8`.

| Instance | Correlation | Start ns | Duration ns | Boundary |
| ---: | ---: | ---: | ---: | --- |
| 1 | 9095 | 10824947040 | 10106112 | Warmup |
| 2 | 10478 | 10880442016 | 8926944 | Warmup |
| 3 | 11865 | 10918609824 | 4742752 | Steady prefill S8 |
| 4 | 13709 | 10980720704 | 10102688 | Steady decode step 0 |
| 5 | 15187 | 11040526912 | 8816768 | Steady decode step 1 |
| 6 | 16665 | 11081646112 | 4690080 | Steady decode step 2 |
| 7 | 18143 | 11120935584 | 7599008 | Steady decode step 3 |

All seven use stream 7, grid `1187x1x1`, block `64x1x1`, 149 registers per
thread, and 24,576 B static shared memory.

Statistics:

```text
sum   = 54,984,352 ns
min   =  4,690,080 ns
median=  8,816,768 ns
max   = 10,106,112 ns
mean  =  7,854,907.428571 ns
```

There is no single outlier. All seven launches are multi-millisecond.

The warmup instances are correlations 9095 and 10478, totaling
`19,033,056 ns`. They are excluded from steady-state contribution. The five
representative-boundary instances total `35,951,296 ns`.

## Attribution Mechanism

For all 11 exact `/MatMul` NVTX ranges:

```text
exact /MatMul NVTX range
  -> contained CUDA launch API
  -> runtime correlationId
  -> exactly one CUPTI kernel
```

There is one runtime event per correlation and one kernel per correlation. Each
launch API start is contained by exactly one `/MatMul` NVTX range. Therefore,
runtime ownership is HIGH. Seven correlations use h16816; four use the expected
xmma tactic.

The kernels start after their NVTX ranges end because CUDA execution is
asynchronous. Ownership comes from launch correlation, not from time-overlap
inference. This leaves the historical generic mapping confidence at MEDIUM for
the aggregate chain, while Phase 6-A supplies HIGH ONNX semantic identity for
`/MatMul` as QK^T in decoder layer 0.

## Boundary Correction

The Phase 3-C steady-state Mixed GPU denominator is `147,830,560 ns` for one
representative S8 prefill plus decode steps 0 through 3. Warmup and
initialization are excluded. The all-trace denominator is `230,950,048 ns`.

| Contribution | Boundary | Time ns | Share |
| --- | --- | ---: | ---: |
| Seven h16816 `/MatMul` launches | All trace | 54,984,352 | 23.807898% |
| Five h16816 `/MatMul` launches | Representative steady | 35,951,296 | 24.319258% |
| All nine `/MatMul` steady launches | Representative steady | 36,027,136 | 24.370560% |
| All 56 attention MatMul candidates minus warmup h16816 | Representative steady | 42,782,720 | 28.940376% |
| Two warmup h16816 `/MatMul` launches | Warmup only | 19,033,056 | 22.898428% of warmup kernels |

The all-trace `23.807898%` value must not be presented as steady-state
contribution. The Phase 6-A tactic-consistent family value remains
`6.831424 ms`, or `2.957966%`, on the all-trace denominator.

## Cross-Phase Comparison

Phase 3-E NCU profiled the same full kernel name and launch configuration:

```text
trt_ampere_h16816gemm_128x64_ldg8_nn_v1
grid 1187x1x1
block 64x1x1
registers/thread 149
static shared memory 24,576 B
duration mean 7,601.557 us
memory/L2 97.06%
SM 39.57%
HMMA active 39.673148%
achieved occupancy 24.78%
direct DRAM N/A
```

That NCU context was FP16 persistent, S=8, `decode_steps=0`, not the Phase 3-C
Mixed raw trace. It supports the microarchitectural identity of the h16816
configuration but does not replace the direct `/MatMul` runtime correlation.

Phase 4-F's up_proj and gate_proj h16816 launches use the same full kernel name
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1` but grid `24x1x1`, not `1187x1x1`.
Their per-launch durations are on the order of hundreds of microseconds, not
multi-milliseconds. Therefore the Phase 4-F up_proj semantic context must not be
merged into the Phase 6-B `/MatMul` context. They are same family but different
launch configurations.

## Hypothesis Decision

| Hypothesis | Decision |
| --- | --- |
| H1 Real attention hotspot | `SUPPORTED / BOUNDED` |
| H2 Real kernel but misattributed | `REJECT` |
| H3 Measurement or aggregation artifact | `REJECT` |
| H4 Insufficient evidence | `REJECT` |

H1 remains bounded because the observed tactic is not the static layer tactic
and the exact CUDA implementation semantics, physical layout, accumulator
semantics, and replacement surface are unresolved.

## Gate

```text
H16816_REAL_BUT_OPTIMIZATION_SURFACE_UNRESOLVED
PASS / BOUNDED
NO IMPLEMENTATION
```

## Authorization State

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
```

No engine rebuild, ONNX modification, precision change, tactic forcing, or
runtime redesign is authorized by Phase 6-B.
