# Phase 5-A CUDA Feasibility Baseline Report

## Gate

**Phase 5-A Step 2: `INCONCLUSIVE / BOUNDED`.**

No CUDA optimization is proven. A Phase 5-B CUDA kernel discussion is **not**
authorized by this result. This is a feasibility baseline study, not a CUDA
kernel implementation phase.

## Executive Summary

The frozen decode-only `up_proj` shape
`C[1,3072] = A[1,1024] * B[1024,3072]` was measured with direct cuBLASLt and a
CUTLASS `v3.5.1` library-generated candidate. The TensorRT evidence was not
rerun and remains the frozen Phase 4-E result.

Direct cuBLASLt was the fastest correctness-passing standalone library result.
Its primary FP32-accumulate configuration had median `0.080077961 ms` and CV
`0.0073666%`. The best CUTLASS candidate had median `0.083619133 ms` and CV
`0.0104699%`, approximately `4.42%` slower than cuBLASLt.

The Phase 4-E TensorRT per-layer median range is
`0.153280005-0.159759998 ms`, but it was measured through `IProfiler` inside
the existing Mixed Decode engine with a different execution boundary. Therefore
the lower standalone-library values are informational and do **not** prove that
TensorRT can be replaced or improved by that margin.

## Benchmark Table

| Backend | Selection | Median | Mean | CV | TFLOPS | Correctness |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cuBLASLt | FP32 accumulate, algorithm 21, heuristic 4 | `0.080077961 ms` | `0.080076141 ms` | `0.000073666` | `0.078568422` | PASS |
| CUTLASS | `v3.5.1`, candidate 1, `tb32x32x64`, split-K 1 | `0.083619133 ms` | `0.083616242 ms` | `0.000104699` | `0.075242033` | PASS |
| TensorRT | Phase 4-E `IProfiler`, 28-layer per-layer median range | `0.153280005-0.159759998 ms` | `UNKNOWN` for directly comparable boundary | `0.18-0.21` per layer | `UNKNOWN` | PASS / BOUNDED |

The full 26-record table is in [benchmark.csv](benchmark.csv). Raw result
JSON files preserve all measured candidates.

## Measurement Protocol

Both standalone harnesses used the same shape, row-major layout, FP16 operands
and output, deterministic seeds, input distribution, alpha `1.0`, beta `0.0`,
200-call warmup plus a `1000 ms` CUDA Event warmup window, seven measured
trials, at least 500 calls and approximately at least `500 ms` per trial.
Timing boundaries used CUDA events around repeated GEMM calls only.

The cuBLASLt harness requested eight heuristics for both FP32 and FP16
accumulate. It recorded algorithm ID, workspace size, heuristic state and waves
count. CUTLASS was acquired as a fixed release under Jetson `/tmp`; no CUTLASS
source or binary was committed.

## Correctness

The FP64 CPU gate was:

```text
max_abs_error <= 1e-3 + 1e-4 * max_abs_reference
```

Eight of eight cuBLASLt FP32-accumulate records passed. Eight of eight cuBLASLt
FP16-accumulate records failed and were excluded from ranking. All ten CUTLASS
candidates passed. Details are in
[correctness_report.md](correctness_report.md).

## Gate Mapping

| Case | Decision | Reason |
| --- | --- | --- |
| Case A: TensorRT fastest | `NOT_APPLICABLE` | Phase 4-E `IProfiler` and standalone GEMM use different timing boundaries. |
| Case B: TensorRT approximately cuBLASLt | `NOT_APPLICABLE` | The same boundary mismatch prevents a direct equality claim. |
| Case C: CUTLASS clearly wins | `NOT_SATISFIED` | CUTLASS is slower than cuBLASLt, so no >=20% CUTLASS improvement exists. |
| Overall | `INCONCLUSIVE / BOUNDED` | Direct standalone comparison is internally valid, but TensorRT comparison is boundary-mismatched. |

Because Case C is not satisfied, the pre-registered bootstrap CI criterion is
not invoked to justify a CUDA target. The run does not establish a proven CUDA
optimization target, and no CUDA kernel implementation may begin.

## Evidence

| Artifact | SHA-256 |
| --- | --- |
| cuBLASLt harness source | `ed85cb348c54e441152607cd318d466a9d056629778f495f3fa55089d6051d1b` |
| CUTLASS harness source (executed) | `bd858537f6d9940d3003f9e2843449192610b464d8cc3faece42e25057bae45f` |
| CUTLASS harness source (serialization normalization) | `5497d59c60f547132c69f75ad14b133452aa6810e022efac6caa431fbfda2067` |
| `cublaslt_results.json` | `665edd7add4cb3cea2beea0bfcd845d21fbbc51b59f17e0e9c2df5db6f7d4256` |
| `cutlass_results.json` (normalized) | `b84ead7a32ea0a738495538edde86da6b4a318159f7242d04a126e84a329320b` |
| Jetson original `cutlass_results.json` | `52780c21a8f385cffd556531083832dbd609b8d2aef7a810f3fd3256d8a639ad` |

The direct result JSONs are the primary raw evidence. `summary.csv`,
`summary.json` and `benchmark.csv` are compact derived artifacts.

The original CUTLASS stdout was retained on Jetson. Its `threadblock` and
`warp` fields were emitted as unquoted shape strings, so the committed JSON was
mechanically normalized to quote those fields. No timing, correctness, device,
algorithm or checksum value was changed; original and normalized SHA-256 values
are both recorded.

## Limitations

- Device power, clocks, p-state and utilization remained uncontrolled and were
  mostly unreadable on the integrated platform.
- TensorRT backend identity remains `UNKNOWN`; it is not assumed to use
  cuBLASLt.
- The exact cuBLASLt vendor kernel name remains `UNKNOWN`.
- CUTLASS used a bounded candidate set, not exhaustive tuning.
- Phase 4-E inputs and measurement boundary differ from the standalone
  synthetic GEMM harnesses.
