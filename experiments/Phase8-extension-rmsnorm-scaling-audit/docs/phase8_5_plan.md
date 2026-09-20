# Phase 8.5 RMSNorm Scaling and Numerical Stability Audit Plan

Date: 2026-09-20

## Objective

Audit the unchanged Phase 8.4-A RMSNorm TensorRT plugin when the replacement
scope grows from one `input_layernorm` node to multiple Transformer nodes.
This experiment measures error propagation and latency attribution only. It
does not implement fusion, optimize the plugin, or change any existing Phase 8
file or artifact.

## Frozen scales

The source graph is the same portable 28-layer FP16 Qwen3 decoder graph used by
Phase 8.4-A. The historical Jetson `/tmp` graph was missing at session start,
so a new isolated source graph is reconstructed from the pinned Qwen3 0.6B
checkpoint. The checkpoint is verified before use.

| Scale | Replaced nodes |
| ---: | --- |
| 0 | None; deterministic baseline self-check |
| 1 | Layer 0 `input_layernorm` |
| 4 | Layers 0-3 `input_layernorm` |
| 8 | Layers 0-7 `input_layernorm` |
| 28 | Layers 0-27 `input_layernorm` |
| 56 | Layers 0-27 `input_layernorm` plus layers 0-27 `post_attention_layernorm` |

Q/K RMSNorm nodes and the final RMSNorm are not replaced.

## Predeclared gate

For every nonzero scale:

1. all compared hidden and replaced-norm outputs are finite;
2. every replaced RMSNorm output has relative L2 <= `1e-3`;
3. `hidden_l27` has relative L2 <= `0.05`;
4. all 8 deterministic greedy generated tokens match the baseline.

If any condition fails, the result is
`NO_PROVEN_SCALABLE_PLUGIN_PATH`. Passing the gate is still bounded: the
experiment does not claim full-model accuracy, deployment readiness, or
optimization.

## Measurements

For prefill and decode, each scale records all 28 hidden-state relative L2 and
max absolute errors, final hidden error, replaced-norm output error, CUDA-event
prefill/decode latency, and deterministic generation consistency. Nsight
Systems records CUDA API launch counts and GPU kernel count/time for the same
fixed profile workload. Raw `.nsys-rep` files stay Jetson-local; only compact
summaries enter the repository.

## Evidence policy

Engine and ONNX binaries are not committed. Failed attempts and logs are
preserved in a timestamped artifact directory. Missing evidence is recorded as
`UNKNOWN`; insufficient evidence is recorded as `INCONCLUSIVE`.
