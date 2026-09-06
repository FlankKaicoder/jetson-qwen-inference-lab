# Phase 6-C Plan: QK^T Dynamic Path Attribution

## Objective

Attribution-only reconciliation of the decode QK^T `/MatMul` path evidence. The
phase determines what the historical Mixed persistent NSYS trace can and cannot
prove about the eleven exact `/MatMul` launches, including the apparent h16816
and xmma paths.

No new Jetson execution, benchmark, profiling, engine build, ONNX rewrite,
precision change, tactic forcing, or runtime redesign is authorized.

## Starting State

| Field | Value |
| --- | --- |
| Starting branch | `phase/06b-h16816-anomaly-reconciliation` |
| Working branch | `phase/06c-qk-dynamic-path-attribution` |
| Starting HEAD | `abb1cfd5bb8a652e042fcaf864aa8eff4da95fa5` |
| Tracked working tree at start | Clean |
| New Jetson execution | None |
| New profiling or benchmark | None |
| Engine, ONNX, precision, or tactic change | None |

Protected untracked directories remain untouched and unstaged:

```text
experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T203103Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040200Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040300Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040400Z/
```

## Method

1. Re-read Phase 6-A and Phase 6-B reports and derived evidence.
2. Reopen the Jetson-local Phase 3-C SQLite in read-only URI mode over SSH.
3. Reconstruct the eleven exact `/MatMul` NVTX ranges, contained runtime calls,
   correlated kernels, and immediate NVTX parent ranges.
4. Record runtime tensor shapes as `UNKNOWN` when absent from the trace.
5. Use code-derived workload lengths only as a sequence hint, never as direct
   per-launch tensor-shape proof.
6. Compare h16816 and xmma parent stacks and refuse normalized performance when
   workload identity cannot be established.
7. Use Phase 6-A static cross-layer evidence only for static path context.

## Decision Rules

The final gate must be `QK_PATH_TRANSITION_UNRESOLVED` with Gate D unless a
bounded frozen-engine audit supplies stronger evidence. Gate A/B/C would require
a proven same-workload dynamic transition. Gate D documents insufficient
historical evidence and stops without implementation.

## Hard Constraints

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
```
