# Phase 5-A Environment Freeze Report

## Gate

**Stage: `PASS / READ_ONLY_ENVIRONMENT_FREEZE`**

No benchmark, CUDA kernel, CUTLASS candidate, TensorRT Plugin, engine rebuild,
ONNX change, package install, clock change, or power change was performed.

## Frozen Environment

| Field | Value |
| --- | --- |
| Host | `nvidia-desktop`, Jetson Orin Nano Engineering Reference Developer Kit Super |
| JetPack / Tegra release | `R36 (release), REVISION: 4.3` |
| GPU / capability | `Orin (nvgpu)`, SM `8.7` |
| Driver | `540.4.0` |
| CUDA nvcc | `12.6.68` at `/usr/local/cuda/bin/nvcc` |
| C++ compiler | `g++ 11.4.0` |
| cuBLAS / cuBLASLt package | `12.6.1.4-1`; `cublasLtGetVersion() = 120601` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| Python | `3.10.12` |
| TensorRT | `10.3.0` |
| Nsight Compute | `2024.3.1.0` at `/usr/local/cuda/bin/ncu` |
| Nsight Systems | `2024.5.4.34-245434855735v0` |
| CUTLASS | `NOT_FOUND_IN_STANDARD_SEARCH_PATHS` |

`nvidia-smi` did not expose useful power, clock, memory, p-state, or utilization
fields on this integrated platform. They are recorded as `N/A` or
`UNKNOWN_NO_READABLE_NVIDIA_SMI_FIELDS`; no value is inferred.

## Frozen Baseline Asset

The existing Mixed Decode TensorRT engine remains read-only:

```text
/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine
650,285,868 bytes
SHA-256 445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c
```

The TensorRT Baseline 1 evidence remains the already frozen Phase 4-E
`IProfiler` result. It will not be rerun in Phase 5-A.

## Repository State

- Windows branch: `phase/05a-cuda-feasibility-baseline-study`
- Windows HEAD: `4979469d82e39910fe54de8275de442054e85b04`
- Remote Jetson branch: `phase/03e-tensorrt-kernel-attribution`
- Remote Jetson HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Remote working tree has pre-existing untracked paths; none were modified.
- The pre-existing local untracked Phase 2.3-B artifact remains preserved.

## Risks And Unknowns

- CUTLASS is not installed. Acquisition, if approved, must be a fixed-version
  checkout under Jetson `/tmp`, with its commit SHA recorded; it must not be
  committed to Git.
- Exact cuBLASLt algorithm and workspace behavior remain
  `UNKNOWN_UNTIL_DIRECT_HARNESS_RUN`.
- Device power and clock state are not readable through the attempted
  `nvidia-smi` fields. Benchmark conclusions must account for this as an
  uncontrolled-environment limitation.
- The local cuBLASLt harness is design-only until separately reviewed.
