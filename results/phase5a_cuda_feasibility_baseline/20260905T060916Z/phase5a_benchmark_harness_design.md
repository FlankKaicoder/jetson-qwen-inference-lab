# Phase 5-A Benchmark Harness Design

## Scope

This document freezes the comparison design. It is not an implementation and
does not authorize a CUDA kernel. A future direct harness must use vendor or
CUTLASS library entry points only; it must not write a custom GEMM kernel.

## Frozen Target

```text
C[1,3072] = A[1,1024] * B[1024,3072]
dtype: FP16 activation, FP16 weight, FP16 output
mode: decode-only, batch 1
```

The logical weight layout is `B[K,N]`. Physical layout and TensorRT tactic
semantics remain bounded by Phase 4 evidence and must not be inferred.

## Baseline 1: TensorRT

Use the existing Phase 4-E evidence without rerunning it:

```text
results/phase4e_up_proj_baseline/20260904T161320Z/
```

Record mean, median, sample standard deviation, CV, minimum, maximum, and P95
for both single-layer and 28-layer aggregate views. Keep these explicit
limitations:

- `IProfiler` measures TensorRT layer time and does not expose CUDA kernels.
- Inputs were deterministic synthetic tensors, not real-token activations.
- There was no clock or power control.
- It is not directly comparable to a standalone GEMM harness without a caveat.

No TensorRT engine may be rebuilt or modified in Phase 5-A.

## Baseline 2: cuBLAS / cuBLASLt

Build a direct cuBLASLt harness later, not a PyTorch-only proxy. It must:

1. use deterministic synthetic operands and record checksums;
2. validate the row-major-to-column-major mapping with a tiny known matrix
   before timing;
3. use `cublasLtMatmul` with FP16 A/B/C;
4. benchmark `CUBLAS_COMPUTE_32F` and `CUBLAS_COMPUTE_16F` as separate variants;
5. use heuristic results, record every evaluated algorithm and workspace size,
   and name the selected algorithm;
6. use alpha `1.0` and beta `0.0`;
7. exclude allocation, reference generation, synchronization, and correctness
   checks from CUDA Event timing.

The exact backend kernel name is not guaranteed. Record the cuBLASLt API
algorithm rather than claiming a kernel name.

## Baseline 3: CUTLASS Candidate

The environment freeze found no CUTLASS installation. If approved, acquire a
fixed CUTLASS release under Jetson-local `/tmp`, record its tag and commit SHA,
and build only a library-generated GEMM instantiation or use the CUTLASS
profiler. This is not a custom CUDA kernel.

Use a small pre-registered candidate matrix suitable for skinny decode GEMMs,
for example:

```text
TensorOp 16x8x16
tile shapes: 32x32x64, 64x64x64, 128x64x64
stages: 4, 5, 6
split-K: 1, 2, 4, 8
FP32 accumulator primary; FP16 accumulator secondary if supported
```

Do not explore an unbounded configuration space. Record the exact selected
template, kernel name, CUTLASS commit, and compile command.

## Correctness Gate

Use a deterministic FP64 or FP32 CPU/GPU reference over the same operands and
apply the inherited Phase 4-C gate:

```text
max_abs_error <= 1e-3 + 1e-4 * max_abs_reference
```

For every candidate record shape equality, finite checks, CUDA status, guard
status, `max_abs_error`, and `max_rel_error`. A candidate that fails
correctness is excluded from performance ranking and marked
`FAIL_CORRECTNESS`. Do not weaken the gate to make a result pass.

## Timing Protocol

Use CUDA Event timing around repeated GEMM calls only.

- Calibration determines iterations for an approximately 500 ms window.
- Use at least 1000 ms warmup.
- Use at least 7 measured trials per candidate and variant.
- Record total window, iterations, per-call mean/median/std/CV/min/max/P95.
- Randomize or rotate candidate ordering to reduce ordering bias.
- Keep workspaces resident during measurements.
- Record device name, clocks/power where readable, allocator state, stream, and
  synchronization boundary.

Achieved TFLOPS is informational:

```text
TFLOPS = 2 * M * K * N / elapsed_seconds / 1e12
```

It must not be treated as hardware peak utilization.

## Stability And Comparison Gate

For each backend/variant, require aggregate CV `<= 5%`. If this is not met, the
corresponding comparison is `INCONCLUSIVE`; do not salvage it with a different
metric.

For each candidate comparison, compute the median delta relative to TensorRT
Baseline 1 and a 10,000-sample bootstrap 95% confidence interval over trial
aggregates. Pre-register:

- `TIE` if the interval includes zero or absolute median delta is below 10%.
- `CLEAR_CANDIDATE_WIN` if median improvement is at least 20% and the 95% CI
  lower bound remains above 10%.
- otherwise `NO_PROVEN_ADVANTAGE`.

Correctness failure or boundary mismatch overrides any speed comparison.

## Phase 5-A Gate Mapping

```text
Case A: TensorRT fastest and stable -> NO CUDA TARGET
Case B: TensorRT approximately cuBLASLt and CUTLASS no clear win
        -> NO PROVEN OPTIMIZATION
Case C: CUTLASS correct, stable, and clear winner
        -> Phase 5-B discussion only
Other: boundary mismatch, CV > 5%, or correctness failure
        -> INCONCLUSIVE / BLOCKED as appropriate
```

A Phase 5-A `Case C` result authorizes a Phase 5-B discussion, not an
implementation.

## Required Artifacts

The future benchmark stage must create, without overwriting this design:

- `preflight.json`
- `cublaslt_results.json` and `cublaslt_results.csv`
- `cutlass_results.json` and `cutlass_results.csv`
- `correctness.json`
- `summary.csv` and `summary.json`
- `phase5a_cuda_feasibility_report.md`
- optional compact NCU summaries; raw `.ncu-rep` remains Jetson-local

## Non-Goals

- No CUDA kernel authoring.
- No TensorRT Plugin.
- No engine rebuild, ONNX modification, or tactic forcing.
- No runtime integration.
- No claim from microbenchmark to end-to-end runtime improvement.
