# Phase 2.2 Gate Plan

This plan separates tonight's CPU-side preparation from future runtime execution. A gate cannot pass merely because an interface exists; it needs the evidence named below.

| Gate | Objective | Required future evidence | Current preparation state |
| --- | --- | --- | --- |
| A - runtime design | Ownership, sequence and prefill/decode contracts are explicit | Reviewed design docs and precheck | `PREPARED` |
| B - KV cache prototype | CPU-side layout and position semantics behave as specified | Deterministic unit checks for allocation, append, visibility and reset | `PREPARED` |
| C - single-layer TensorRT integration | Bind a bounded single-layer graph to explicit cache input/output buffers | Authorized TensorRT execution, output/canary checks and saved artifacts | `NOT_STARTED` |
| D - full Qwen3 FP16 runtime | Execute prefill and decode through a full model runtime | Explicit authorization, full export/engine design, correctness against Phase 1 reference, cache lifecycle and stability evidence | `NOT_STARTED` |

Gate C and Gate D require separate authorization because they exceed this preparation scope. They must not be inferred from Phase 2.1.8's synthetic stateless decoder-block result.
