# Exp04 - CUDA GEMM: Tiling and WMMA

## Scope

Exp04 establishes a CPU FP32 reference (V0), a one-thread-per-output CUDA FP32 baseline (V1), shared-memory tiled FP32 variants (V2), and an explicitly mixed-precision WMMA path (V3).

Status: `PASS / CLOSED`

## Reproduction

On Jetson, use the scripts in `scripts/`:

```bash
bash experiments/Exp04-gemm/scripts/build.sh
bash experiments/Exp04-gemm/scripts/run_correctness.sh
bash experiments/Exp04-gemm/scripts/run_benchmark.sh
bash experiments/Exp04-gemm/scripts/run_wmma_dual_correctness.sh
```

The binary is kept outside Git under `/tmp/jetson-qwen-exp04-build`. Timestamped correctness, environment, calibration, raw trial, and summary files are written under `benchmark/` and are never overwritten.

## Versions

- V0: CPU row-major FP32 reference, `C[m,n] = sum_k A[m,k] * B[k,n]`. Correctness-only.
- V1: naive CUDA FP32, one thread computes one `C[row,col]`, with bounds checks for arbitrary dimensions.
- V2-T16: shared-memory tiled FP32 GEMM; selected by the bounded T8/T16/T32 survey.
- V3-WMMA-FP16: FP16 inputs, FP32 accumulation/output using WMMA; performance and SASS evidence are retained from the formal run.

Double buffering and `cp.async` are outside the Exp04 scope and were not started.

## Current gates

- Gate A1: `PASS` — V0/V1 correctness matrix.
- Gate A2: `PASS` — V2-T8/T16/T32 correctness matrix.
- Gate A3: `PASS` — V3 aligned correctness plus dual FP16-reference/original-FP32-reference evidence.
- Gate B: `PASS` — retained adaptive formal cross-version benchmark.
- Gate C: `PASS` — retained NCU and HMMA SASS evidence.
- Overall: `PASS / CLOSED`.

## Initial evidence (2026-08-31)

Jetson Orin (CC 8.7, CUDA 12.6) built V1 successfully. Gate A1 passed 13/13 cases with intact output canaries and no CUDA failures; max absolute/relative error and RMSE were zero for the deterministic inputs. The raw file is `benchmark/raw/correctness_20260831T141605Z.csv`.

The initial V1 adaptive benchmark used seven trials for `256^3`, `512^3`, `1024^3` and `512x384x640`. Mean latency/GFLOPS were `0.243670 ms / 137.704 GFLOPS`, `1.826164 ms / 146.997 GFLOPS`, `20.862597 ms / 102.935 GFLOPS`, and `1.755451 ms / 143.358 GFLOPS`, respectively. CV was `0.0349%`, `0.5045%`, `0.1473%`, and `0.1161%`. The `256^3` actual event window was 341 ms due to calibration drift; this is retained as a methodology observation, not a final stability gate.

V2-T8/T16/T32 correctness passed 13/13 each; the bounded survey selected V2-T16. V3-WMMA-FP16 correctness passed 8/8 aligned cases. Formal cross-version benchmark and Gate C are documented in `docs/exp04_summary.md` and `docs/nsight_analysis.md`.

The dual-reference WMMA artifact is `benchmark/raw/wmma_correctness_dual_reference_20260831T165518Z.csv`. All eight aligned shapes pass implementation correctness against the FP16-quantized/FP32-accumulation reference, with intact canaries and no CUDA failures. The comparison against the original FP32 reference characterizes the end-to-end numerical impact of the mixed-precision WMMA path, including FP16 input quantization; it is not an assertion of FP32-equivalent correctness or a Tensor Core-only error.

The precision-impact closure patch adds host-side reference evidence only. Existing V1/V2/V3 performance, benchmark methodology, NCU reports and SASS evidence were not rerun or modified. Double buffering is not required by the current scope; `cp.async` and Exp05 execution remain not started.
