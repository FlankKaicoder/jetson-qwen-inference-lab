# Phase 5-A Direct cuBLASLt Result

## Gate

**Stage: `PASS / MEASURED`; FP32-accumulate comparison primary.**

The harness used direct `cublasLtMatmul` only. It did not use `torch.matmul`
as the measured backend. No TensorRT engine or runtime path was changed.

## Backend Identity

| Field | Value |
| --- | --- |
| Backend | cuBLASLt |
| API | `cublasLtMatmul` |
| CUDA runtime | 12060 |
| cuBLASLt runtime | 120601 |
| Matrix layout | A `1x1024`, B `1024x3072`, C `1x3072`, row-major |
| Primary compute type | FP16 input/output, FP32 accumulate |
| Scale type | FP32 |

Eight heuristic candidates were requested for each compute type. The primary
selected result is heuristic index 4:

| Field | Value |
| --- | ---: |
| Algorithm ID | 21 |
| Workspace size | 0 bytes |
| Heuristic waves count | 0.75 |
| Heuristic state | `CUBLAS_STATUS_SUCCESS` (0) |

The exact vendor kernel name remains `UNKNOWN`; cuBLASLt exposes an algorithm
identity, not a kernel-name contract.

## Primary Timing

| Metric | Value |
| --- | ---: |
| Mean | `0.080076141 ms` |
| Median | `0.080077961 ms` |
| Sample std | `0.000005899 ms` |
| CV | `0.000073666` (0.0073666%) |
| Min | `0.080066331 ms` |
| Max | `0.080081552 ms` |
| P95 | `0.080081552 ms` |
| Iterations per trial | 6243 |
| TFLOPS | `0.078568422` |

All eight FP32-accumulate variants were stable with CV below `0.43%`. The
fastest was heuristic 4; see `cublaslt_results.json` for all records.

## FP16 Accumulate

FP16-accumulate variants are recorded separately. All eight failed the frozen
max-absolute-error gate and are excluded from performance ranking. The fastest
FP16 record had mean `0.078483504 ms`, but `max_abs_error` was
`0.002110384` and correctness was `FAIL`.

## Limitations

- Power and clock state were unreadable through attempted `nvidia-smi` fields
  and remained uncontrolled.
- The backend algorithm identity is cuBLASLt algorithm 21; the underlying
  kernel name is `UNKNOWN`.
- This is a standalone synthetic GEMM result and is not a TensorRT-layer
  comparison without a boundary caveat.
