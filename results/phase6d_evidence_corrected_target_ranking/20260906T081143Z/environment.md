# Phase 6-D Environment

## Analysis Host

| Field | Value |
| --- | --- |
| Role | Repository-side offline evidence synthesis and re-ranking |
| Windows repository | `E:\nvidia-qwen` |
| Branch | `phase/06d-evidence-corrected-target-ranking` |
| Starting HEAD | `4c43da2b5bed5f189ca67d5d64b9814e17cbdb09` |
| Analysis date | 2026-09-06 Asia/Shanghai |
| New Jetson execution | None |
| New benchmark | None |
| New profiling | None |
| Engine or ONNX modification | None |

## Frozen Evidence Device And Runtime

The relevant historical runtime evidence was produced in Phase 3-C on Jetson
Orin Nano Super, SM 8.7, batch 1, S=8 prefill plus four decode steps,
persistent runtime, Mixed engines, TensorRT 10.3.0, CUDA 12.6, and Nsight
Systems 2024.5.4.34. Power mode was `NV Power Mode: 25W`; clocks were not
modified.

## Artifact Handling

All inputs are committed, read-only repository artifacts or reports. This phase
does not reopen the Jetson-local SQLite, copy a raw profiler binary, alter a
historical result, stage a protected directory, or create a large raw artifact.

Protected untracked directories remain untouched and unstaged:

```text
experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T203103Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040200Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040300Z/
results/phase6a_unknown_attention_matmul_attribution/20260906T040400Z/
```
