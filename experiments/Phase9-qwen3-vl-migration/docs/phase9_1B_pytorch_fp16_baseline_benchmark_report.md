# Phase 9.1-B Qwen3-VL PyTorch FP16 Baseline Benchmark

Date: 2026-09-15 (Asia/Shanghai)

## Scope And Authorization

This was an explicitly authorized PyTorch FP16 baseline benchmark. It measured
preprocess, vision encoder, projector, prefill, decode, tokens/s, GPU memory,
and board power on one fixed image-plus-text workload. No TensorRT, ONNX,
quantization, optimization, backend modification, benchmark sweep, or clock and
power-mode change occurred. No package or environment configuration was changed.

## Gate

**PASS / BOUNDED — EAGER_ATTENTION_BASELINE**

The benchmark completed 3 warmups and 10 measured trials under the frozen
protocol. The eager-attention path is a known compatibility choice from Phase
9.1-A, not an optimization claim. These results are a bounded PyTorch baseline
only and do not imply that PyTorch is optimal or that TensorRT will be faster.

## Protocol Freeze

The protocol and configuration were frozen before measurement in
`benchmark_protocol.json`. Key conditions were:

- Model: pinned `Qwen/Qwen3-VL-2B-Instruct` at revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- Dtype: FP16 on `cuda:0`.
- Attention: `eager`, forced by the Phase 9.1-A SDPA/`enable_gqa`
  incompatibility.
- Image: deterministic RGB `448 x 448` white background with a red square.
- Prompt: `"Describe the image."`.
- Sampling: greedy.
- Output: exactly 16 tokens; EOS was masked before argmax for the first 15
  decode selections to fix the workload.
- Repetition: 3 warmup trials and 10 measured trials, with no inter-trial sleep.
- Power: `tegrastats` at 100 ms intervals.
- Clock/power mode: not changed.

The script and protocol hashes recorded in the raw result match the frozen
repository files.

## Runtime

| Component | Value |
| --- | --- |
| Host | `nvidia-desktop` |
| Python | `3.10.12` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| CUDA | `12.6` |
| Device | Orin, SM `(8, 7)` |
| Transformers | `4.57.3` |
| Pillow | `11.3.0` |
| Model class | `Qwen3VLForConditionalGeneration` |
| Model dtype | FP16 |
| Parameters | `2,127,532,032` |

## Stage Definitions

- `preprocess` is host wall-clock time from chat-template plus processor call
  through CUDA transfer, FP16 pixel-value cast, and synchronization.
- `vision encoder` is the CUDA-event duration around `model.visual` minus the
  measured projector module durations.
- `projector` is the sum of CUDA-event durations around `visual.merger` and the
  three `visual.deepstack_merger_list` modules.
- `prefill` is a CUDA-event duration around the complete model forward on the
  full image-plus-text input.
- `decode` is host wall-clock time around the 15 subsequent cached decode calls.
- `tokens/s` is end-to-end generated tokens divided by prefill plus subsequent
  decode latency.

The vision-encoder value is therefore derived as `visual_total - projector`,
not an independently isolated decoder-free module measurement.

## Measured Baseline

Each value is mean ± standard deviation over 10 measured trials.

| Metric | Result |
| --- | ---: |
| Preprocess | `9.010 ± 0.710 ms` |
| Vision encoder | `230.627 ± 4.081 ms` |
| Projector | `7.397 ± 0.310 ms` |
| Prefill | `420.607 ± 5.791 ms` |
| Decode, 15 subsequent calls | `1.832973 ± 0.004183 s` |
| Decode per subsequent token | `122.198 ± 0.279 ms` |
| End-to-end throughput | `7.0999 ± 0.0269 tokens/s` |

Per-trial min/max values are:

| Metric | Min | Max |
| --- | ---: | ---: |
| Preprocess | `8.096 ms` | `10.447 ms` |
| Vision encoder | `224.527 ms` | `237.236 ms` |
| Projector | `6.840 ms` | `7.760 ms` |
| Prefill | `410.311 ms` | `428.767 ms` |
| Decode per subsequent token | `121.750 ms` | `122.639 ms` |
| End-to-end throughput | `7.0580 tokens/s` | `7.1514 tokens/s` |

The output was identical in all measured trials and decoded as:

`"ThisThis is a simple, geometric image of a red square. The square is"`

## GPU And Host Memory

| Snapshot | CUDA allocated | CUDA reserved |
| --- | ---: | ---: |
| After model load | `4,255,079,424` B | `4,309,647,360` B |
| After benchmark | `4,263,599,616` B | `4,433,379,328` B |
| Benchmark peak | `4,363,340,288` B | `4,433,379,328` B |

Host memory available at benchmark end was `403,836,928` bytes and process max
RSS was `1,808,625,664` bytes. No OOM occurred. These are allocator snapshots,
not profiler counters.

## Board Power

`tegrastats` produced 296 samples at 100 ms during the measurement window.

| Rail | Mean | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `VDD_IN` | `12,359.895 mW` | `12,171 mW` | `5,475 mW` | `15,824 mW` |
| `VDD_CPU_GPU_CV` | `3,613.287 mW` | `3,359.5 mW` | `992 mW` | `5,817 mW` |
| `VDD_SOC` | `3,896.304 mW` | `3,955 mW` | `1,587 mW` | `4,591 mW` |

This is board/rail power, not GPU-only power. No clock or power mode was
changed.

## Instrumentation Caveat

The raw JSON records `prefill_logits_finite=false`. This is a benchmark
instrumentation artifact: the EOS mask used during the first decode selection
mutated the same logits view that was later checked. Therefore actual prefill
logit finiteness is `INCONCLUSIVE` in this run. The token IDs and decoded text
remain valid raw outputs, but this benchmark does not establish a numerical
correctness gate.

## Limitations

- One image, one prompt, and one output length were tested.
- Eager attention is not a performance recommendation.
- `vision encoder` is derived from the total visual forward minus projector
  module times.
- Tokens/s is end-to-end throughput, not decode-only throughput.
- Memory and power snapshots do not establish peak board power or DRAM traffic.
- This baseline does not establish any optimization opportunity.

## Evidence

- Frozen protocol:
  `benchmark_protocol.json`
- Raw benchmark result:
  `benchmark_result.json`
- Raw power log:
  `tegrastats.log`
- Benchmark source:
  `experiments/Phase9-qwen3-vl-migration/src/phase9_1B/run_fp16_baseline.py`
