# Phase 2.3-C Layer / Operator Sensitivity

Status: `PASS / BOUNDED` (2026-09-03)

This audit covers all 196 decoder Linear weights (28 layers x 7 operators), 34
portable dynamic targets, and 8 deterministic TensorRT confirmation targets.
The frozen Qwen3 revision and Phase 2.3-B corpus (24 calibration, 12 disjoint
evaluation) were reused. Every target input is `EXACT_LINEAR_INPUT_PROVEN` by
real Transformers forward-pre-hook. Activation scales use calibration-only
`BOUNDED_MSE_CLIP` with factors `0.90, 0.925, 0.95, 0.975, 1.00`.

Static PT-W8 symmetric per-tensor and PC-W8 symmetric per-output-channel
(PyTorch axis 0) reconstruction records hashes, shapes, errors and cosine.
Portable evaluation compares F, PT-W8, PC-W8, PT-W8A8 and PC-W8A8 on all held-
out samples with six requested deltas and clipping counts. The largest
portable PT-W8A8-vs-F P95 relative-L2 target is `L27:up_proj` at `0.3665403904`.

The TensorRT subset is Layer 0 q_proj plus the worst portable target per
operator, deduplicated to eight. FP16, PT-W8 and PT-W8A8 build/execute finite
for all targets. TensorRT 10.3 per-channel PC-W8 and PC-W8A8 QDQ graphs are
explicitly `BLOCKED` by parser failure in this harness; portable PC-W8 results
remain valid and are not represented as TRT support.

Gate: `PASS / BOUNDED`. No latency/throughput/power benchmark, Nsight, INT4,
28-layer quantized runtime, or C1/RoPE debugging was performed. Evidence:
`experiments/Phase2-qwen3-quantization/artifacts/phase2_3c_20260903T220600Z/`.
