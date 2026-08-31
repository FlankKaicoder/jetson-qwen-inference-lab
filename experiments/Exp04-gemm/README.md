# Exp04 - CUDA GEMM: Naive FP32 Foundation

## Scope

This first stage establishes a CPU FP32 reference (V0) and the simplest one-thread-per-output CUDA FP32 kernel (V1). It intentionally stops before shared-memory tiling, WMMA/Tensor Cores, vectorization, register tiling, double buffering, or `cp.async`.

Status: `INCONCLUSIVE` (A3 evidence completion pending)

## Reproduction

On Jetson, use the scripts in `scripts/`:

```bash
bash experiments/Exp04-gemm/scripts/build.sh
bash experiments/Exp04-gemm/scripts/run_correctness.sh
bash experiments/Exp04-gemm/scripts/run_benchmark.sh
```

The binary is kept outside Git under `/tmp/jetson-qwen-exp04-build`. Timestamped correctness, environment, calibration, raw trial, and summary files are written under `benchmark/` and are never overwritten.

## Versions

- V0: CPU row-major FP32 reference, `C[m,n] = sum_k A[m,k] * B[k,n]`. Correctness-only.
- V1: naive CUDA FP32, one thread computes one `C[row,col]`, with bounds checks for arbitrary dimensions.

V2 Shared Memory Tiling, V3 WMMA, double buffering and `cp.async` are explicitly not started in this stage.

## Current gates

- Gate A1: CPU/V1 correctness across tiny, boundary-like, rectangular, non-power-of-two and large cases, with FP32 error tolerances and output canaries.
- Gate B/C and the final Exp04 Gate A/B/C remain `NOT_STARTED` until the corresponding evidence exists.

## Initial evidence (2026-08-31)

Jetson Orin (CC 8.7, CUDA 12.6) built V1 successfully. Gate A1 passed 13/13 cases with intact output canaries and no CUDA failures; max absolute/relative error and RMSE were zero for the deterministic inputs. The raw file is `benchmark/raw/correctness_20260831T141605Z.csv`.

The initial V1 adaptive benchmark used seven trials for `256^3`, `512^3`, `1024^3` and `512x384x640`. Mean latency/GFLOPS were `0.243670 ms / 137.704 GFLOPS`, `1.826164 ms / 146.997 GFLOPS`, `20.862597 ms / 102.935 GFLOPS`, and `1.755451 ms / 143.358 GFLOPS`, respectively. CV was `0.0349%`, `0.5045%`, `0.1473%`, and `0.1161%`. The `256^3` actual event window was 341 ms due to calibration drift; this is retained as a methodology observation, not a final stability gate.

V2-T8/T16/T32 correctness passed 13/13 each; the bounded survey selected V2-T16. V3-WMMA-FP16 correctness passed 8/8 aligned cases. Formal cross-version benchmark and Gate C are documented in `docs/exp04_summary.md` and `docs/nsight_analysis.md`.

Exp04 closure is currently `INCONCLUSIVE`: the required original-FP32-reference WMMA precision-impact comparison is not yet present in raw evidence. Double buffering is not required by the current scope; `cp.async` and Exp05 execution remain not started.
