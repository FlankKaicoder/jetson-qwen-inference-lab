# Phase 6-F Environment

## Analysis Host

| Field | Value |
| --- | --- |
| Role | Offline repository evidence synthesis and target re-ranking |
| Windows repository | `E:\nvidia-qwen` |
| Starting branch | `phase/06e-qk-runtime-shape-invocation-attribution` |
| Starting HEAD | `826a5ae1735e1fe2f48ee583ddb86ed3f1c199f5` |
| New branch | `phase/06f-post-reproduction-target-ranking` |
| Analysis date | `2026-09-06` |
| New inference run | None |
| New benchmark | None |
| New Nsight Systems or Nsight Compute run | None |
| Engine deserialization or build | None |
| ONNX export or modification | None |
| Precision or tactic change | None |

## Historical Evidence Runtime

The relevant runtime evidence was produced on Jetson Orin Nano Super, SM 8.7,
batch 1, TensorRT `10.3.0`, CUDA `12.6`, PyTorch
`2.5.0a0+872d972e41.nv24.08`, Nsight Systems `2024.5.4.34`, and power mode
`NV Power Mode: 25W`. Clocks and power state were not changed.

## Artifact Handling

Only committed repository artifacts and reports are read. Raw Nsys and SQLite
files remain Jetson-local. No historical result is moved, deleted, or
overwritten. The following protected untracked directories remain untouched and
unstaged:

```text
experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T203103Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040200Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040300Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040400Z/
```
