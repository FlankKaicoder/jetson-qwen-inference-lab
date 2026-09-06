# Phase 6-G Gate Update

## Final Gate

```text
AV_RUNTIME_SURFACE_RECOVERED
```

## Decision

Gate A is selected because all required links were recovered:

1. Phase 6-A semantic identity is HIGH for 28/28 Attention x V decode nodes.
2. Phase 6-G recovered 28/28 standalone TensorRT runtime layers.
3. Phase 6-G recovered 112/112 direct NVTX-to-runtime-to-kernel correlations.
4. All 112 kernels have one known xmma GEMM kernel name and family.
5. The representative steady contribution is `4,237,568 ns`, or `2.866503%`
   of the `147,830,560 ns` steady GPU kernel denominator.

Gate B is not selected because the kernel and steady surfaces are complete for
the observed decode boundary. Gate C is not selected because AV can be isolated
beyond semantic identity.

## Important Boundaries

The all-trace aggregate is `4,237,568 ns / 1.834842%` of
`230,950,048 ns`. The representative steady share is `4,237,568 ns /
2.866503%` of `147,830,560 ns`. The raw durations are identical only because
all observed AV instances occur in decode steps 0-3. The denominators and
conclusions are not interchangeable.

No AV instance is observed in `PHASE3B_WARMUP` or
`PHASE3B_STEADY_PREFILL_S8`.

## Optimization Surface

Q5 remains:

```text
NOT PROVEN
```

This gate does not prove headroom, bottleneck cause, replacement feasibility,
expected benefit, or a TensorRT tactic defect. Kernel arguments and exact
runtime workload identity remain `UNKNOWN`.

## Authorization

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
Engine rebuild: NOT AUTHORIZED
Precision change: NOT AUTHORIZED
Tactic forcing: NOT AUTHORIZED
```

The only authorized next decision is an owner-reviewed controlled AV
feasibility study, corrected target re-ranking, or closing the Attention
branch. Do not automatically start another phase.
