# Phase 6-D Gate Update

## Final Gate

```text
NEXT_ATTRIBUTION_TARGET_RECOVERED
```

Phase 6-D status is `PASS / BOUNDED`.

## Decision

The selected attribution target is `P6D_001`, the layer-0 decode QK^T h16816
path. It is not an optimization or implementation target.

The decision is bounded because:

1. Seven h16816 launches are real and directly owned by `/MatMul`, with
   54,984,352 ns all-trace and 35,951,296 ns on the representative steady
   boundary.
2. Phase 6-A gives HIGH ONNX semantic identity as layer-0 QK^T.
3. Phase 6-C leaves runtime Q/K/output shapes as `UNKNOWN`.
4. Same mathematical workload, same graph region, same engine invocation, and
   transition trigger are `NOT_PROVEN` or `UNKNOWN`.
5. Normalized h16816-versus-xmma performance is `NOT_CALCULATED`.
6. No clean replacement surface is proven, so `NEXT_FEASIBILITY_TARGET_RECOVERED`
   is not justified.

## Next Action Only

Proposed owner decision: authorize or reject a minimal Phase 6-E frozen-engine
attribution study for layer-0 QK runtime shapes and invocation/workload
identity. Phase 6-D does not execute that study.

## Authorization

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
CUTLASS implementation: NOT AUTHORIZED
Engine rebuild: NOT AUTHORIZED
Precision change: NOT AUTHORIZED
Tactic forcing: NOT AUTHORIZED
```
