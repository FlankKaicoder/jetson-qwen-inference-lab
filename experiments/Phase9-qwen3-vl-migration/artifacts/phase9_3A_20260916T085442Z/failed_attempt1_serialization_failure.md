# Phase 9.3-A Failed Attempt 1: Result Serialization

Date: 2026-09-16 (Asia/Shanghai)

The first harness execution loaded the pinned model, reached the unchanged C1
TensorRT engine, created an execution context, and returned an execute success
at the integration boundary. It then failed during `json.dumps` while writing
the final result:

```text
TypeError: Object of type Tensor is not JSON serializable
```

The cause was that the harness stored raw measured-trial tensors inside the
serializable result object under a temporary `raw_measured_trials` field. This
was a harness serialization defect, not a model, TensorRT, CUDA, or inference
failure. No final result JSON was produced by this attempt. The protocol,
model, engine, and environment were unchanged.

The corrected harness kept measured tensors in process-local state for
cross-backend comparison but did not place them in the serialized result. The
frozen protocol hash remained
`4f1f83c68e0d6d1edcf389e586e13dd7243989396157a3dcfc5caf7f78dc94b8`.
The corrected script SHA-256 is
`9d775fcef2f60e814e84a748540b7a0dc25f636d47915b6d7e9a4c033e813887`.
