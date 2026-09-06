# Phase 6-B Environment

## Analysis Host

| Field | Value |
| --- | --- |
| Windows repository | `E:\nvidia-qwen` |
| Branch | `phase/06b-h16816-anomaly-reconciliation` |
| HEAD | `dfdb0de64f26198caf13433f9448f476c4fe3691` |
| Local Python launcher | `py` |
| Jetson access | Read-only SSH alias `jetson` |

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

SQLite was opened with `file:<path>?mode=ro` and `uri=True`. No remote trace,
engine, artifact, or repository file was modified.

## Relevant Static Engine Metadata

The Mixed Decode engine is recorded by Phase 4-A.0 with SHA-256
`b9305150544221c601861aa7b7a86232bbc62d851ac45619fe6166758cc5fe71` for the
committed standalone EngineInspector copy. The runtime-side engine copy
remaining at `/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine`
was not deserialized or executed in Phase 6-B.
