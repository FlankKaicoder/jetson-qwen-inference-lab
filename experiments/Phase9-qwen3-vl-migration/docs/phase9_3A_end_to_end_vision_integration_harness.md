# Phase 9.3-A Qwen3-VL End-to-End TensorRT Vision Integration Harness

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized the first-stage establishment of an end-to-end functional
benchmark harness integrating the validated Phase 9.2-C1 TensorRT FP16 Vision
Encoder into the Qwen3-VL inference path. The controlled comparison used:

1. PyTorch FP16 Vision Encoder plus the existing decoder path.
2. The unchanged TensorRT FP16 Vision Engine plus the same decoder path.

No optimization, quantization, engine rebuild, decoder modification, CUDA
modification, benchmark sweep, input sweep, or persistent environment
modification occurred.

## Gate

**PASS / BOUNDED — END_TO_END_HARNESS_FUNCTIONAL**

The harness established the required controlled comparison and completed all
generation trials. Both backends generated the same 16-token sequence and
decoded text. The direct visual-output integration check produced matching
shapes and fully finite outputs for all four tensors. First-token top-1 outputs
agreed in all three measured TensorRT trials.

This is a functional, bounded harness result. It is not a deployment gate,
optimization result, or statistically strong performance benchmark.

## Protocol And Identity

The protocol was frozen before measurement as `harness_protocol.json`. Its
SHA-256 is
`4f1f83c68e0d6d1edcf389e586e13dd7243989396157a3dcfc5caf7f78dc94b8`.

| Field | Value |
| --- | --- |
| Model identity | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| `config.json` SHA-256 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `model.safetensors` SHA-256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| TensorRT engine SHA-256 before/after | `aa5c200eb5dcbc5abaef55bc014b5210236fb9ca34394217a024d3796e44823c` |
| Device | Jetson Orin, `cuda:0`, capability `8.7` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| CUDA | `12.6` |
| TensorRT | `10.3.0` |
| Transformers | `4.57.3` |
| Attention implementation | `eager` |
| Workload | Deterministic 448x448 red-square image, `"Describe the image."` |
| Sampling | Greedy, fixed 16 generated tokens |
| Warmups per backend | `1` |
| Measured trials per backend | `3` |
| Backend order | PyTorch FP16, then TensorRT FP16 |
| Power mode | `25W`, unchanged |

The integration adapter was assigned at runtime to `model.model.visual`. The
language model was `Qwen3VLTextModel` before and after injection, with
`1,720,574,976` parameters in both cases. The generation rule and decoder
weights were unchanged.

## Direct Vision Integration Check

Before generation, one PyTorch visual pass and one TensorRT visual pass were
compared on the same processor output. All four outputs had shape
`[196,2048]` and finite ratio `1.0` on both sides.

| Output | Max absolute error | Mean absolute error | Cosine similarity |
| --- | ---: | ---: | ---: |
| `final_hidden` | `0.982421875` | `0.009175275452435017` | `0.9991949200630188` |
| `deepstack_0` | `0.046875` | `0.0026899827644228935` | `0.9999522566795349` |
| `deepstack_1` | `1.405029296875` | `0.008119885809719563` | `0.9993446469306946` |
| `deepstack_2` | `1.51953125` | `0.012291578575968742` | `0.9991330504417419` |

TensorRT reported 209 engine layers, 5 IO tensors, 1 optimization profile, and
`14,450,688` bytes context device memory. Runtime logs contained one
default-stream warning and zero errors.

## Generation Timing

First-token latency is the CUDA-event duration around the complete prefill
forward. Total latency is the CUDA-event duration from immediately before
prefill through the final fixed-token decode call. Preprocessing is excluded.
Tokens/s uses 16 generated tokens divided by total latency.

| Backend | First-token mean | First-token median | First-token stddev | Total mean | Total median | Total stddev | Mean tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP16 | `263.0651092529297` ms | `266.63592529296875` ms | `9.823092474506687` ms | `2129.2288411458335` ms | `2132.2744140625` ms | `11.605921012859655` ms | `7.514606813851262` |
| TensorRT FP16 | `260.0856526692708` ms | `259.72393798828125` ms | `4.816746009177781` ms | `2122.2044270833335` ms | `2117.330322265625` ms | `8.526273408900606` ms | `7.539411177985605` |

| Metric | Value |
| --- | ---: |
| First-token latency delta, TensorRT minus PyTorch | `-2.979456583658873` ms |
| Total latency delta, TensorRT minus PyTorch | `-7.0244140625` ms |
| Total latency ratio, TensorRT over PyTorch | `0.9967009586162097` |
| Total latency ratio, PyTorch over TensorRT | `1.00330996108238` |

Only three measured generations per backend were run. The near-identical total
latency is expected evidence for this harness-only stage and must not be
interpreted as a proven optimization or deployment conclusion.

## Output Generation And Agreement

Both backends generated:

```text
ThisThis is a simple, geometric image featuring a solid red square centered on a
```

The token IDs were:

```text
[1986, 1986, 374, 264, 4285, 11, 52484, 2168, 16445, 264,
 6437, 2518, 9334, 30188, 389, 264]
```

All six measured generations were successful. All three trials per backend had
identical token sequences, and the baseline and TensorRT token sequences were
identical. The first-token top-1 token was `1986` on both backends in all three
TensorRT comparisons.

The full-vocabulary first-token logits had finite ratio
`151935 / 151936` on both backends: each contained one negative infinity. The
cause and token index were not diagnosed in this phase. Therefore max absolute
error, mean absolute error, and cosine similarity over full-vocabulary logits
are `UNKNOWN`. No tolerance was applied and no logit rerun or diagnosis was
authorized.

## CUDA Memory

Memory values are PyTorch CUDA allocator snapshots only. They do not include
TensorRT-managed device allocations and are not profiler DRAM counters.

| Snapshot | Allocated | Reserved |
| --- | ---: | ---: |
| After model load | `4,255,079,424` B | `4,324,327,424` B |
| After PyTorch vision reference | `4,269,960,704` B | `4,431,282,176` B |
| After TensorRT injection and cleanup | `3,456,848,896` B | `4,263,510,016` B |
| After PyTorch backend trials | `3,456,851,456` B | `4,299,161,600` B |
| After TensorRT backend trials | `3,456,852,992` B | `4,299,161,600` B |

The recorded process peak allocator usage was `4,363,708,928` bytes allocated
and `4,433,379,328` bytes reserved.

## Execution Attempts

The first execution reached the TensorRT integration boundary but failed during
final JSON serialization because raw trial tensors were retained in the
serializable result object. The failure was a harness serialization defect, not
an inference failure. It is preserved in
`failed_attempt1_serialization_failure.md`. The corrected harness completed
the frozen protocol.

## Limitations And Non-Claims

- The evidence applies only to the deterministic image, prompt, fixed
  `[784,1536]` / `[1,28,28]` boundary, and 16-token greedy workload.
- Three measured trials per backend are harness evidence, not a formal
  benchmark or statistical comparison.
- No preprocessing time is included in the requested timing metrics.
- No power measurement was requested or recorded.
- Full-vocabulary first-token logit error metrics are `UNKNOWN` because both
  backends contained one negative-infinity logit; no diagnosis was authorized.
- The result does not establish optimization benefit, deployment readiness, or
  realistic-image robustness.
- No decoder modification, optimization, quantization, engine rebuild, CUDA
  modification, or environment change occurred.

## Evidence

- Frozen protocol:
  `artifacts/phase9_3A_20260916T085442Z/harness_protocol.json`
- Raw integration result:
  `artifacts/phase9_3A_20260916T085442Z/integration_result.json`
- Command log:
  `artifacts/phase9_3A_20260916T085442Z/command_log.md`
- Preserved failed attempt:
  `artifacts/phase9_3A_20260916T085442Z/failed_attempt1_serialization_failure.md`
- Artifact manifest:
  `artifacts/phase9_3A_20260916T085442Z/artifact_manifest.json`
- Harness script:
  `src/phase9_3A/run_end_to_end_vision_integration.py`

The next action is to stop and await Gate review. No further optimization,
rebuild, benchmark sweep, quantization, decoder change, CUDA change, or
environment change is authorized by this report.
