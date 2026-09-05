# Phase 4-C up_proj Benchmark Protocol

## Scope

This protocol separates a standalone microbenchmark from runtime integration.
A microbenchmark win alone must never be reported as a runtime optimization.

## Baseline

The exact baseline is the existing TensorRT `up_proj` FP16 GEMM path under the
frozen shape. Because the committed Phase 3-E/4-A evidence does not provide a
pure `up_proj` per-call latency, a future exact baseline probe must measure the
existing path before comparing a custom implementation.

The observed Phase 3-E kernel family is
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1`, but Phase 4-A.2 records multiple
kernel families for `up_proj`. The probe must verify the actual executed kernel
subset rather than assume that every invocation uses h16816.

## Microbenchmark method

Inherit the Exp04 methodology:

- CUDA Event kernel-only timing.
- Approximately 1000 ms time-based warmup.
- Adaptive measurement targeting approximately 500 ms of CUDA Event work, with a 2x safety factor.
- At least seven independent trials.
- Alternating or paired baseline/custom measurement to reduce drift coupling where possible.
- Report mean, median, sample standard deviation, CV, min, and max for each path.
- Exclude allocation, initialization, H2D, D2H, reference computation, and output checks from the timed region.
- Record device, CUDA/TensorRT/driver versions, clocks/power state policy, shape, dtypes, layout, warmup/measurement iterations, and event timing boundary.

The M=1, K=1024, N=3072 problem has `2*M*N*K = 6,291,456` FLOPs.

## Baseline probe boundary

The future baseline probe must measure only the existing GEMM path or a
faithfully extracted identical GEMM problem. It must not rebuild or replace the
engine, force tactics, alter quantization policy, or modify the runtime. If
extraction changes layout, launcher, synchronization, or allocation behavior,
that limitation must be recorded before comparison.

## Runtime integration protocol

Runtime integration is a separate future phase and must use the persistent
TensorRT runtime, not the legacy short-lived context pattern. Required
conditions:

- Same engines except the explicitly authorized candidate integration.
- Same tokenizer, prompt IDs, batch, prefill sequence lengths, decode steps, warmup, and measured repeats.
- Same-session paired comparison against the Phase 3-C persistent Mixed runtime baseline.
- Report prefill, S=8 and S=16 decode TPOT, decode windows, memory, and context counters.
- Never substitute the microbenchmark result for end-to-end decode evidence.

## No performance claim in Phase 4-C

Phase 4-C records existing evidence only. It does not prove that a custom kernel,
plugin, or GEMM replacement will be faster.
