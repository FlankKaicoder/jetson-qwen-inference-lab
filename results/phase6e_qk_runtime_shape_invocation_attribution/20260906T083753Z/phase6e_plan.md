# Phase 6-E Plan

## Objective

Recover, for the frozen Mixed decoder engines, the runtime invocation identity
and direct I/O tensor-shape evidence needed to interpret the historical
layer-0 decode QK^T `/MatMul` h16816 and xmma launches. This is an
observation-only audit.

## Scope

The study uses one minimal controlled workload:

1. One warmup prefill with sequence length 8.
2. One steady prefill with sequence length 8.
3. Decode steps 0, 1, 2, and 3.

The historical Phase 3-C Mixed persistent profile used two prefill warmups,
one S=8 prefill, and decode steps 0-3. The new run is therefore a
NEW_CONTROLLED_RUN with the same engine, runtime, sample, forced-token, and
workload sequence, but not a byte-for-byte reproduction.

## Evidence Method

1. Deserialize five frozen engines and create one persistent execution context
   per engine.
2. Wrap each unique engine enqueue in an NVTX child range carrying its engine
   invocation ID.
3. Wrap each complete prefill or decode step in an NVTX parent range carrying
   the runtime invocation ID.
4. After each execution and stream synchronization, query every TensorRT I/O
   tensor with `IExecutionContext::get_tensor_shape`,
   `IExecutionContext::get_tensor_strides`, and engine tensor metadata.
5. Run once without Nsys to validate correctness and logging.
6. Run the identical minimal workload under Nsys with CUDA and NVTX tracing.
7. Export a read-only SQLite database and join:
   `NVTX -> CUDA runtime API -> CUDA kernel`.
8. Correlate layer-0 candidate kernels with invocation ID and direct I/O
   shapes.

## Shape Evidence Rules

- Shapes sampled from TensorRT I/O tensors are
  `DIRECT_RUNTIME_EVIDENCE`.
- Semantic Q, K, and QK output shapes are likely engine-internal and may not be
  exposed as I/O tensors. If absent, they remain `UNKNOWN`.
- If derivable from direct runtime cache shapes and frozen static graph shape
  metadata, they are labeled `DERIVED_FROM_PROVEN_STATE`; they are not direct
  runtime evidence.
- Static EngineInspector shapes remain `STATIC_GRAPH_EVIDENCE`.
- Historical launches remain `HISTORICAL_EVIDENCE`.

## Workload Comparison Rules

`EXACT_SAME_WORKLOAD` requires strong evidence for the same semantic QK
operation, compatible runtime shape/work, the same engine invocation context,
and observable dtype/layout constraints. Same decode step, same parent enqueue,
or the same static ONNX node alone is insufficient.

Allowed classifications are:

```text
EXACT_SAME_WORKLOAD
COMPARABLE_WORKLOAD
DIFFERENT_WORKLOAD
UNKNOWN
```

## Gate Options

Exactly one final gate will be selected:

```text
QK_SAME_WORKLOAD_PATH_DIVERGENCE_RECOVERED
QK_DYNAMIC_PATH_TRIGGER_RECOVERED
QK_RUNTIME_OPERATION_SEPARATION_RECOVERED
QK_H16816_PATH_NOT_REPRODUCED
QK_RUNTIME_ATTRIBUTION_INCONCLUSIVE
```

## Safety Limits

No historical artifact modification, engine rebuild, ONNX modification,
precision change, tactic forcing, plugin, FlashAttention, custom CUDA, NCU, or
runtime redesign is authorized. No clocks or power mode will be changed. The
four protected untracked directories will remain untouched.

## Required Artifacts

Exactly these primary artifacts are required in this directory:

1. `phase6e_plan.md`
2. `environment.md`
3. `evidence_manifest.csv`
4. `historical_run_configuration.md`
5. `runtime_shape_log.csv`
6. `qk_runtime_kernel_correlation.csv`
7. `qk_path_trigger_analysis.csv`
8. `qk_workload_comparison.csv`
9. `phase6e_attribution_report.md`
10. `gate_update.md`
