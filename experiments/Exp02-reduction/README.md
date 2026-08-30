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


## Exp02.2 Benchmark Gate

Gate B is PASS. B1/B2/B3 were completed with CUDA Event timing over the complete GPU reduction pipeline. All measured configurations passed the frozen correctness criterion. The final stability winner is V5/B128 at 1.626439 ms mean for N=16,777,229; V7/B128 is 2.321449 ms and V6/B128 is 2.655211 ms. V7 vs V6 paired 95% CI is [0.316896, 0.350628] ms for delta=V6-V7. Detailed raw evidence and limitations are in notes/benchmark_analysis.md.

Gate A = PASS; Gate B = PASS; Gate C = NOT_STARTED; Overall = IN_PROGRESS. Exp03 has not started.
