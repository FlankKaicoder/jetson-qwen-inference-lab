# Phase 9.3-B1 Checkpoint

Phase 9.3-B1 recorded the authorized end-to-end stage latency breakdown for
Qwen3-VL. The frozen workload used the deterministic red-square image, fixed
prompt, 16-token greedy decoding, one warmup, and three measured generations per
backend.

Gate: `PASS / BOUNDED — STAGE_LATENCY_ATTRIBUTION_RECORDED`.

Decode was the dominant generation stage for both PyTorch FP16
(`82.72677578113881%`) and the unchanged TensorRT FP16 visual adapter
(`87.7778526059737%`). TensorRT internal encoder/projector timings and power are
`UNKNOWN`. The backends diverged at token index 8; the cause remains `UNKNOWN`
and was not diagnosed.

Report:
`experiments/Phase9-qwen3-vl-migration/docs/phase9_3B1_end_to_end_stage_latency_breakdown.md`

Evidence:
`experiments/Phase9-qwen3-vl-migration/artifacts/phase9_3B1_20260916T094011Z/`

Stop and await Gate review before any diagnosis, input sweep, optimization,
rebuild, rerun, benchmark, quantization, decoder change, CUDA change, or
environment change.
