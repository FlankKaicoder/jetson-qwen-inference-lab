# Phase 9.2-C2 Qwen3-VL Vision FP16 Numerical Correctness

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized numerical correctness validation only for the C1 TensorRT
FP16 Vision Engine against the PyTorch FP16 `Qwen3VLVisionModel`. The fixed
migration boundary was `pixel_values=[784,1536]` and `grid_thw=[1,28,28]`.

No benchmark, latency measurement, optimization, quantization, CUDA
modification, model conversion, or engine build was performed. The per-process
`PYTHONPATH` bridge used existing venvs only; no package was installed, removed,
upgraded, or persisted.

## Gate

**BLOCKED / NON_FINITE_FP16_FIXED_BOUNDARY**

The fixed-boundary comparison executed, but every PyTorch reference output and
every TensorRT output was non-finite. Therefore max absolute error, mean
absolute error, and cosine similarity are recorded as `NaN`, and numerical
correctness is `INCONCLUSIVE`. This result does not prove an engine bug, a
PyTorch bug, an export bug, or a model bug; root cause remains `UNKNOWN`.

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

The deterministic comparison input was FP16
`torch.linspace(-1, 1, 784 * 1536).reshape(784, 1536)`. Input finiteness was
not separately instrumented and remains `UNKNOWN`.

## Execution Attempts

The first attempt stopped before model loading or engine execution because this
Torch build rejected `torch.cuda.reset_peak_memory_stats(torch.device(...))`.
The console and script are preserved as
`artifacts/phase9_2C2_20260916T042306Z/failed_attempt1_invalid_device/`.

After a narrow device-index handling correction, the second attempt completed
one deterministic PyTorch FP16 pass and one TensorRT engine execution. The
visual state dict loaded strictly with no missing or unexpected keys. TensorRT
deserialized successfully, created an execution context, reported both shape
flags true, and `execute_async_v3` returned true.

## Comparison Result

TensorRT outputs were read into FP16 device buffers, then converted to FP32 on
the CPU for metrics. The `dtype_tensorrt` fields in the JSON therefore describe
those FP32 comparison copies; the C1 engine metadata established the bindings as
FP16.

All outputs had the expected `[196,2048]` shape. No required metric is finite.

| Output | Reference finite | TensorRT finite | Max abs error | Mean abs error | Cosine similarity |
| --- | --- | --- | ---: | ---: | ---: |
| `final_hidden` | `false` | `false` | `NaN` | `NaN` | `NaN` |
| `deepstack_0` | `false` | `false` | `NaN` | `NaN` | `NaN` |
| `deepstack_1` | `false` | `false` | `NaN` | `NaN` | `NaN` |
| `deepstack_2` | `false` | `false` | `NaN` | `NaN` | `NaN` |

Summary: all shapes matched, but all reference outputs, all TensorRT outputs,
and all derived metrics were non-finite.

## GPU Memory

The process used the current CUDA device and measured its process-local PyTorch
memory only.

| Field | Bytes |
| --- | ---: |
| Allocated before | `0` |
| Reserved before | `0` |
| Peak allocated | `921,025,536` |
| Peak reserved | `956,301,312` |
| Allocated after | `17,351,168` |
| Reserved after | `48,234,496` |

## Runtime Log

TensorRT collected 9 log records: 2 INFO, 6 VERBOSE, 1 WARNING, and 0 errors.
The warning was TensorRT's default-stream advice for `enqueueV3()`. It was not
an execution error. A SciPy/NumPy version warning appeared during Python import
but did not stop execution.

## Limitations And Non-Claims

- The result applies only to the deterministic FP16 input and exact fixed
  `[784,1536]` / `[1,28,28]` boundary.
- Because both sides are non-finite, this is not evidence that TensorRT agrees
  or disagrees numerically with PyTorch.
- The non-finite root cause is not diagnosed by this phase.
- No benchmark, latency, throughput, optimization, quantization, tactic change,
  CUDA change, dynamic workload, realistic image workload, or decode path was
  evaluated.
- Start/finish timestamps are operational metadata, not performance evidence.

## Evidence

- Raw result:
  `artifacts/phase9_2C2_20260916T042306Z/correctness_result.json`
- Successful-run console log:
  `artifacts/phase9_2C2_20260916T042306Z/console.log`
- Failed attempt:
  `artifacts/phase9_2C2_20260916T042306Z/failed_attempt1_invalid_device/`
- Command log:
  `artifacts/phase9_2C2_20260916T042306Z/command_log.md`
- Artifact manifest:
  `artifacts/phase9_2C2_20260916T042306Z/artifact_manifest.json`
- Script:
  `src/phase9_2C2/run_vision_correctness.py`

The next action is to stop and await Gate review. No follow-up diagnosis, input
change, rerun, engine rebuild, optimization, benchmark, or migration work is
authorized by this report.
