# Phase 6-C Environment

## Analysis Host

| Field | Value |
| --- | --- |
| Windows repository | `E:\nvidia-qwen` |
| Starting branch | `phase/06b-h16816-anomaly-reconciliation` |
| Working branch | `phase/06c-qk-dynamic-path-attribution` |
| Starting HEAD | `abb1cfd5bb8a652e042fcaf864aa8eff4da95fa5` |
| Jetson access | Read-only SSH alias `jetson` |
| New execution or profiling | None |

## Frozen Evidence Device And Runtime

The raw trace was produced in Phase 3-C on Jetson Orin Nano Super, SM 8.7,
batch 1, S=8 prefill plus four decode steps, persistent runtime, Mixed engines,
TensorRT 10.3.0, CUDA 12.6, and Nsight Systems 2024.5.4.34. Power mode was
`NV Power Mode: 25W`; clocks were not modified.

## Raw Trace Identity

| Artifact | Value |
| --- | --- |
| Remote SQLite | `/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite` |
| SQLite SHA-256 | `ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8` |
| SQLite bytes | `3,477,504` |
| Remote Nsys report | `/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.nsys-rep` |
| Nsys report SHA-256 | `14a579d7b15f5f936879fd13964021adf58b4cff6d9dc3e727177cfb17a9cce6` |
| Nsys report bytes | `782,415` |

The raw SQLite remains Jetson-local. It was opened with
`file:<path>?mode=ro` and `uri=True`. No remote trace, engine, artifact, or
repository file was modified, and no large profiler binary was copied into Git.
