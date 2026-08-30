# Exp02 - CUDA Reduction

## Status

Exp02.0 initialization and Exp02.1 correctness are the current scope. Gate B (Benchmark/Stability) and Gate C (Nsight/Microarchitecture) remain `NOT_STARTED`; Overall remains `IN_PROGRESS`.

## Learning objective

Reduction makes thread cooperation explicit: global loads feed per-thread values, shared memory holds block-local partial sums, barriers establish visibility, warp execution reduces synchronization, and warp shuffle exchanges register values within a warp.

## Implemented mechanisms

V1 global multi-pass, V2 modulo-interleaved shared reduction, V3 indexed interleaved reduction, V4 sequential addressing, V5 first-add-during-load, V6 explicit warp-tail synchronization, and V7 `__shfl_down_sync` are implemented in `src/`.

## Correctness

The runner covers six supported block sizes, eight block-relative boundary sizes, plus N=2^20+13 at B=256, three deterministic input patterns, seven versions and three independent executions per case: 3,087 executions total. It records timestamped raw and summary CSV files under `benchmark/raw/`. The frozen tolerance and complete matrix are in `notes/experiment_design.md`.

## Reproduction

On Jetson:

```bash
experiments/Exp02-reduction/scripts/build.sh
experiments/Exp02-reduction/scripts/run_correctness.sh
```

Build output stays under `/tmp/jetson-qwen-exp02-build`. This phase does not run benchmark or Nsight profiling.

