# Phase 6-A Environment

## Analysis Environment

| Field | Value |
| --- | --- |
| Role | Repository-side offline attribution recovery |
| Host path | `E:\nvidia-qwen` |
| OS | Microsoft Windows NT 10.0.26200.0 |
| Shell | PowerShell 7.6.5 |
| Python launcher | `py` |
| Python version | 3.14.0 |
| Analysis date | 2026-09-06 Asia/Shanghai |

## Evidence Environment

The evidence is offline and read-only. The primary runtime evidence comes from
the already-collected Phase 3-C Mixed persistent NSYS artifacts on Jetson Orin
Nano Super. This phase did not run a Jetson process, TensorRT engine, profiler,
benchmark, build, or data transfer.

## Engine And Runtime Artifact State

- The Phase 2.3-E prefill and decode EngineInspector summary was read only.
- The standalone Phase 4-A Mixed Decode EngineInspector copy was byte-hashed
  against the embedded Phase 2.3-E copy before use.
- No engine was deserialized, rebuilt, executed, or modified.
- No ONNX model was exported, changed, or rebuilt.
- No TensorRT tactic was forced.
- The protected untracked Phase 2 artifact directory
  `experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T203103Z/`
  was not moved, renamed, overwritten, deleted, or staged.
