# Final Status

- Date: 2026-09-16 (Asia/Shanghai)
- Gate: `PROJECT_COMPLETE`
- Branch: `phase/09-qwen3vl-migration`
- Final experiment commit: `42d77ec804e857d46725314c4a704f4768a8b106`
- Closure commit: verify with `git rev-parse HEAD`; do not embed it circularly in this file.

## Closure Decision

The repository has completed the authorized Phase 9.4 closure. The closure
added final summary/status/handoff documentation and updated pointer documents
only. It did not rerun, benchmark, profile, optimize, quantize, rebuild, modify
CUDA, or alter historical artifacts.

The final experiment remains Phase 9.3-B2 — Qwen3-VL Decoder-Side Runtime
Bottleneck Attribution. Its gate is
`PASS / BOUNDED — DECODER_BOTTLENECK_ATTRIBUTION_RECORDED`.

## Completed Scope

| Area | Result |
| --- | --- |
| Phase 0 CUDA fundamentals | Exp01-Exp04 closed `PASS` with preserved inconclusive microarchitectural causes |
| Phase 1 Qwen3 baseline | Pinned HF BF16 reference and formal prefill/decode baseline established |
| Phase 2 quantization/runtime | Bounded TensorRT component/runtime work; full decoder numerical limitation retained |
| Phase 3-4 runtime/operator attribution | Host-context and operator/runtime boundaries recovered; no proven optimization target |
| Phase 5-7 GEMM/attention reassessment | Evidence surfaces recovered, but no proven custom-kernel target |
| Phase 8 RMSNorm | Isolated kernel/plugin gains only; `BOUNDED / NO_END_TO_END_SPEEDUP` |
| Phase 9 Qwen3-VL | Checkpoint, baseline, static Vision ONNX, TensorRT FP16 build, correctness, latency, integration, and decoder attribution completed |
| Phase 9.4 closure | Final report, status, handoff, README/CHANGELOG updates, and registry verification completed |

## Major Conclusions

- The TensorRT FP16 Vision Encoder was about `2.7x` faster than PyTorch FP16
  in the isolated frozen boundary: `82.9734232584635` ms versus
  `225.396496582031` ms mean.
- End-to-end Qwen3-VL mean latency and throughput were effectively unchanged
  with TensorRT Vision: no meaningful end-to-end speedup is claimed.
- Decode was the dominant remaining stage at `87.17536311733093%` of the fixed
  Phase 9.3-B2 generation.
- Decode kernel time attribution was `78.3598%` GEMM-class,
  `19.5831%` memory-like, and `2.0571%` other. Exact attention share, DRAM
  counters, achieved bandwidth, and power remained `UNKNOWN`.
- TensorRT-LLM was investigated as a direction but was not adopted as a
  successful migration result.

## Known Limitations

- Results are Jetson Orin Nano Super and fixed-workload bounded.
- The Vision correctness check had finite outputs but no deployment tolerance.
- Cross-backend end-to-end token divergence in Phase 9.3-B1 was not diagnosed.
- The Qwen3 full TensorRT decoder retained an unresolved Layer 27 numerical
  limitation.
- No production deployment, service, continuous benchmark, or generalized
  optimization claim is made.

## Future Directions, Record Only

These are not commitments and were not started in closure:

1. Decoder-side GEMM/runtime analysis and a separately authorized TensorRT-LLM
   or native-runtime feasibility study.
2. End-to-end optimization only after a new baseline and explicit gate.
3. Qwen3-VL logit divergence, input sweep, or correctness tolerance study.
4. Production-oriented quantization evaluation beyond the closed bounded
   experiments.
5. Extended profiling for DRAM, occupancy, and attention attribution using
   correctly matched boundaries.

Any future work requires a new owner authorization and must start from Git,
raw artifacts, and experiment reports.
