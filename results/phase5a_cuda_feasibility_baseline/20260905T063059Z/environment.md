# Phase 5-A Benchmark Environment

## Device

| Field | Value |
| --- | --- |
| Host | `nvidia-desktop` |
| Board | Jetson Orin Nano Engineering Reference Developer Kit Super |
| GPU | Orin (nvgpu), SM 8.7 |
| CUDA driver | 12060 |
| CUDA runtime | 12060 |
| nvcc | CUDA 12.6.68 |
| cuBLASLt runtime | 120601 |

Power, clocks, p-state and utilization remain
`UNKNOWN_NO_READABLE_NVIDIA_SMI_FIELDS`. No clock or power state was changed.

## Benchmark Software

| Backend | Version / identity |
| --- | --- |
| cuBLASLt | Direct `cublasLtMatmul`, runtime `120601` |
| CUTLASS | `v3.5.1`, commit `f7b19de32c5d1f3cedfc735c2849f12b537522ee` |
| CUTLASS acquisition | Jetson `/tmp/cutlass-v3.5.1`, not committed |

## Common Protocol

```text
shape: C[1,3072] = A[1,1024] * B[1024,3072]
layout: row-major
dtype: FP16 operands and FP16 output
alpha / beta: 1.0 / 0.0
input: std::mt19937_64 uniform [-0.125, 0.125]
warmup: 200 calls plus >=1000 ms CUDA Event window
measurement: 7 trials, >=500 calls and approximately >=500 ms each
timing: CUDA events only
```

The same seeds and operands produced the same checksums in both harnesses.
Refer to the result JSON files for checksum values and source SHA-256 values.
