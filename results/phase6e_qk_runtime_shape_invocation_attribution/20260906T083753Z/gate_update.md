# Phase 6-E Gate Update

## Final Gate

```text
QK_H16816_PATH_NOT_REPRODUCED
```

Phase 6-E status is `PASS / BOUNDED`.

## Decision

The controlled frozen-engine run did not reproduce
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1`. All four controlled decode
layer-0 `/MatMul` invocations correlated to the xmma kernel family. The
historical path is therefore not reproducible under the current attribution
boundary.

This is a bounded pass, not a disproven historical record. Phase 6-B/6-C
remains valid evidence that historical h16816 launches were real and directly
owned by `/MatMul`. It also does not establish that xmma and h16816 executed
the same workload. Historical and new-path workload classification remains
`UNKNOWN`.

## Consequences

1. Direct I/O cache and hidden shapes are `DIRECT_RUNTIME_EVIDENCE`.
2. Layer-0 Q/K/K^T/output GEMM shapes are `DERIVED_FROM_PROVEN_STATE`.
3. Kernel argument identity is `UNKNOWN`.
4. Historical h16816 runtime shape and operation identity are `UNKNOWN`.
5. Historical-versus-controlled workload classification is `UNKNOWN`.
6. The path trigger is `UNKNOWN`.
7. Raw h16816-versus-xmma duration ratios are not valid.
8. Optimization readiness is downgraded.
9. No implementation is authorized.

## Next Action Only

Proposed owner decision: `corrected target re-ranking`. The Phase 6-E result
must be reviewed before any new experiment, implementation, profiling, engine
change, or precision change. Phase 6-E itself is complete and execution must
stop.

## Authorization

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
