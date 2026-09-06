# Phase 6-A Plan

## Objective

Phase 6-A is attribution-only recovery of the historical
`unknown_attention_matmul / /MatMul_*` candidate. The allowed question is what
these nodes semantically and operationally represent, not whether to implement
an attention kernel.

## Scope

The plan reused frozen Phase 2.3-E EngineInspector data, Phase 4-A ONNX and
runtime attribution artifacts, and the Phase 3-C Mixed persistent GPU kernel
summary. No Jetson execution, TensorRT execution, benchmark, profiling, engine
rebuild, ONNX modification, precision change, tactic forcing, CUDA kernel,
FlashAttention, or TensorRT plugin was authorized or performed.

## Recovery Steps

1. Audit branch, HEAD, working tree, project state, and protected artifacts.
2. Hash and read the five frozen evidence inputs.
3. Recover every decode EngineInspector layer whose metadata contains
   `/MatMul` or `/MatMul_0..55`.
4. Join the decode candidates to the ONNX node inventory, producer/consumer
   graph edges, prefill EngineInspector metadata, and runtime NVTX mapping.
5. Assign QK^T or Attention x V identity only when operand provenance, graph
   anchors, shape compatibility, and downstream structure all agree.
6. Reconcile timing only within the Phase 3-C all-trace NVTX-to-kernel
   aggregate boundary and preserve all boundary caveats.
7. Decide a bounded gate and stop before any implementation phase.

## Success Criteria

- Exactly 56 decode MatMul candidates are recovered.
- Each candidate is mapped to one of decoder layers 0 through 27.
- QK^T and Attention x V identities are accepted only at HIGH confidence with
  explicit anchors; otherwise they remain UNKNOWN.
- Prefill, decode, fusion/rewrite, and timing boundaries are separated.
- Historical `61.815776 ms` is not relabeled as a single-node latency.
- A `PASS`, `BOUNDED`, `BLOCKED`, or negative result is acceptable.

## Stop Condition

Stop after attribution, contribution reconciliation, report, gate, repository
state updates, commit, push if permitted, and audit. Do not start any
implementation or new profiling phase.
