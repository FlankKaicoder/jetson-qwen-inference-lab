# Phase 9.2-C2-R2 TensorRT FP16 Vision Correctness With Finite Input

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized a rerun of PyTorch FP16 versus TensorRT FP16 numerical
correctness validation using the corrected finite input generation identified
in Phase 9.2-C2-R1. The fixed boundary was `pixel_values=[784,1536]` and
`grid_thw=[1,28,28]`.

No benchmark, latency measurement, optimization, quantization, TensorRT engine
rebuild, CUDA modification, or persistent environment modification occurred.
The unchanged C1 engine was deserialized and executed once.

## Gate

**PASS / BOUNDED — FINITE_INPUT_CORRECTNESS_METRICS_RECORDED**

Both PyTorch and TensorRT completed the corrected-input workload. All four
outputs had finite ratio `1.0`, matching `[196,2048]` shapes, and finite
comparison metrics. No numerical pass/fail tolerance was authorized or applied;
therefore this gate records the measured agreement and does not claim
deployment-level correctness.

## Inputs And Environment

| Field | Value |
| --- | --- |
| Model identity | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Local model path | `/home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203` |
| `config.json` SHA-256 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `model.safetensors` SHA-256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| Engine path | `/tmp/phase9_2c1_20260916T034956Z/engine/qwen3_vl_vision_fp16.trt` |
| Engine size | `818,910,588` bytes |
| Engine SHA-256 | `aa5c200eb5dcbc5abaef55bc014b5210236fb9ca34394217a024d3796e44823c` |
| Device | Jetson Orin, `cuda:0` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| TensorRT | `10.3.0` |
| Attention implementation | `eager` |

The input was generated with direct CUDA FP32 `torch.linspace(-1, 1, 1204224)`,
reshaped to `[784,1536]`, then cast to FP16. Both the FP32 source and FP16 model
input had finite ratio `1.0`, zero NaNs, and zero infinities. The same FP16
tensor was passed to PyTorch and TensorRT.

## Execution Result

The visual state dict loaded strictly with no missing or unexpected keys. The
TensorRT engine deserialized successfully, context creation succeeded, both
shape flags were true, and `execute_async_v3` returned true. The engine had 209
layers, 5 IO tensors, and 1 optimization profile, matching C1 metadata.

All comparison metrics were computed after copying both FP16 outputs to FP32 on
the host.

| Output | Reference finite ratio | TensorRT finite ratio | Max absolute error | Mean absolute error | Cosine similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final_hidden` | `1.0` | `1.0` | `4.001953125` | `0.03959130868315697` | `0.9805147647857666` |
| `deepstack_0` | `1.0` | `1.0` | `0.09375` | `0.004188189283013344` | `0.9998924732208252` |
| `deepstack_1` | `1.0` | `1.0` | `0.64532470703125` | `0.016078025102615356` | `0.9988695383071899` |
| `deepstack_2` | `1.0` | `1.0` | `2.30859375` | `0.027240395545959473` | `0.9978001117706299` |

Summary: all shapes matched, all reference and TensorRT outputs were finite,
and all metrics were finite. Across outputs, maximum absolute error was
`4.001953125`, maximum mean absolute error was `0.03959130868315697`, and
minimum cosine similarity was `0.9805147647857666`.

## GPU Memory

The process measured its current-process PyTorch memory only.

| Field | Bytes |
| --- | ---: |
| Allocated before | `0` |
| Reserved before | `0` |
| Peak allocated | `926,235,648` |
| Peak reserved | `956,301,312` |
| Allocated after | `22,168,064` |
| Reserved after | `48,234,496` |

## Runtime Log

TensorRT collected 9 log records: 2 INFO, 6 VERBOSE, 1 WARNING, and 0 errors.
The warning was TensorRT's default-stream advice for `enqueueV3()`. It was not
an execution error.

## Limitations And Non-Claims

- This result applies only to the deterministic FP32-source/FP16-cast input and
  exact `[784,1536]` / `[1,28,28]` boundary.
- No authorized numerical tolerance, acceptance threshold, or deployment gate
  was applied.
- No realistic image, dynamic workload, multi-image, video, decode path, or
  end-to-end model was evaluated.
- No benchmark, latency, throughput, optimization, quantization, tactic change,
  CUDA change, or engine rebuild occurred.
- Start/finish timestamps are operational metadata, not performance evidence.

## Evidence

- Raw result:
  `artifacts/phase9_2C2R2_20260916T080511Z/correctness_result.json`
- Console log:
  `artifacts/phase9_2C2R2_20260916T080511Z/console.log` (local-only due to the
  repository's `*.log` ignore rule)
- Command log:
  `artifacts/phase9_2C2R2_20260916T080511Z/command_log.md`
- Artifact manifest:
  `artifacts/phase9_2C2R2_20260916T080511Z/artifact_manifest.json`
- Script:
  `src/phase9_2C2R2/run_vision_correctness_finite_input.py`

The next action is to stop and await Gate review. No further TensorRT run,
engine rebuild, optimization, benchmark, quantization, input sweep, or
migration work is authorized by this report.
