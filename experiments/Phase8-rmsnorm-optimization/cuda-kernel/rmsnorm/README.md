# Phase 8.1 CUDA RMSNorm kernels

This directory contains the independently compiled C++/CUDA benchmark for three
RMSNorm kernel variants. It is intentionally separate from the Phase 8.0
PyTorch baseline.

- `rmsnorm_v0.cu`: one block per token, shared-memory tree reduction.
- `rmsnorm_v1.cu`: one block per token with warp-shuffle reduction.
- `rmsnorm_v2.cu`: `half2` / `bfloat162` vectorized loads and stores with
  warp-shuffle reduction.

The correctness oracle computes an FP32 reduction reference on the host and
rounds the candidate output to the selected storage dtype. The benchmark uses
CUDA Events. It records both one event pair around each kernel and one event
pair around 200 back-to-back kernels. Neither value is presented as a profiler
kernel-only measurement.

Formal protocol is warmup `50`, `200` repetitions per trial, and `5` trials.
The preset correctness gate is finite outputs and relative-L2 `<= 0.005`.
This is a synthetic hidden-size-1024 kernel benchmark; no Qwen3 checkpoint or
TensorRT engine is used.
