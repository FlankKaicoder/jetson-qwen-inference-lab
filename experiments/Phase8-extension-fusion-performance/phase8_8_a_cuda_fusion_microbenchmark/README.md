# Phase 8.8-A CUDA Fusion Microbenchmark

This benchmark evaluates a bounded FP16 Qwen3 decode-like shape:
`M=1`, `K=1024`, `N=3072`.

- Baseline: one RMSNorm CUDA kernel followed by `cublasGemmEx`.
- Fusion: one intentionally simple RMSNorm + Linear CUDA kernel.
- Correctness: both paths and the cross-comparison must be finite with
  relative-L2 `<= 0.01` before timing is allowed.
- Timing: 50 warmup pairs, 10 alternating-order trials, and 500 operations per
  CUDA Event interval.
- Profiling: Nsight Systems records launch count and GPU kernel time. Nsight
  Compute records global load/store sectors, duration, and achieved active
  warps. Sector counts are multiplied by 32 only when reported as L1 global
  traffic bytes; they are not relabeled as DRAM bytes.

The fused kernel caches a normalized input in shared memory per block and uses
four blocks per SM. It is a prototype for testing whether launch/intermediate
traffic savings survive a simple implementation, not a production GEMM or a
TensorRT plugin.

Run `scripts/run_phase8_8a_jetson.sh` in an isolated Jetson workspace. Raw
`.nsys-rep` files remain in that workspace; only compact CSV/text evidence is
copied into the repository artifact directory.
