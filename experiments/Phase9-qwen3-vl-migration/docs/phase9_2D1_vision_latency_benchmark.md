# Phase 9.2-D1 Isolated Qwen3-VL Vision Encoder Latency Benchmark

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized an isolated Qwen3-VL Vision Encoder latency benchmark
comparing PyTorch FP16 `Qwen3VLVisionModel` against the unchanged Phase 9.2-C1
TensorRT FP16 engine. The fixed boundary was `pixel_values=[784,1536]`,
`grid_thw=[1,28,28]`, with direct CUDA FP32 `linspace` generation followed by an
FP32-to-FP16 cast.

No optimization, engine rebuild, quantization, model change, CUDA
modification, or persistent environment modification occurred. No engine or
model file was changed. No realistic-image or dynamic-workload run occurred.

## Gate

**PASS / BOUNDED — VISION_LATENCY_METRICS_RECORDED**

The frozen protocol completed with 3 warmups and 30 measured trials per backend.
The validated finite input and all last-checked outputs were finite. The
unchanged C1 engine executed successfully with zero TensorRT errors.

This is a descriptive fixed-boundary benchmark only. No numerical tolerance,
optimization target, deployment gate, or engine-replacement decision was
authorized or applied.

## Protocol

The protocol was frozen before measurement as
`benchmark_protocol.json`. Its SHA-256 is
`e6a450e9dd8f538b0627aa19bb488300474e843869b55cb1ec3c3f62fb6cc545`.

| Field | Value |
| --- | --- |
| Timing method | CUDA events |
| Timing scope | `Qwen3VLVisionModel` forward or TensorRT `execute_async_v3` only |
| Warmups per backend | `3` |
| Measured trials per backend | `30` |
| Backend order | PyTorch FP16, then TensorRT FP16 |
| Inter-trial sleep | `0` seconds |
| Workload | One deterministic fixed-boundary input |
| Vision tokens per invocation | `196` |
| Throughput definition | `1000 / latency_ms` images/s; `196000 / latency_ms` vision tokens/s |
| Power mode | `25W`, unchanged |
| Clock/power settings | unchanged |

## Environment And Identity

| Field | Value |
| --- | --- |
| Device | Jetson Orin, `cuda:0`, capability `8.7` |
| Model identity | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| `config.json` SHA-256 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `model.safetensors` SHA-256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| TensorRT engine SHA-256 | `aa5c200eb5dcbc5abaef55bc014b5210236fb9ca34394217a024d3796e44823c` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| CUDA | `12.6` |
| TensorRT | `10.3.0` |
| Transformers | `4.57.3` |
| Attention implementation | `eager` |

Both the FP32 source and FP16 input had finite ratio `1.0` and zero NaNs or
infinities. PyTorch loaded 406,957,056 visual parameters strictly with no
missing or unexpected keys. TensorRT reported 209 engine layers, 5 IO tensors,
1 optimization profile, and `14,450,688` bytes context device memory.

## Latency And Throughput

All retained samples are measured CUDA-event latencies. No outlier was
excluded.

| Backend | Mean | Median | Stddev | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP16 | `225.396496582031` ms | `225.261184692383` ms | `1.37536503957161` ms | `223.192184448242` ms | `228.263961791992` ms |
| TensorRT FP16 | `82.9734232584635` ms | `82.4378242492676` ms | `2.36526651267192` ms | `81.8190383911133` ms | `95.1917724609375` ms |

| Backend | Mean images/s | Median images/s | Mean vision tokens/s | Median vision tokens/s |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP16 | `4.43678562153523` | `4.43929122748245` | `869.609981820904` | `870.10108058656` |
| TensorRT FP16 | `12.0604065547899` | `12.1303541076454` | `2363.83968473882` | `2377.5494050985` |

The descriptive comparison was:

| Metric | Value |
| --- | ---: |
| Mean latency delta, TensorRT minus PyTorch | `-142.42307332356773` ms |
| Median latency delta, TensorRT minus PyTorch | `-142.82336044311523` ms |
| Mean latency ratio, TensorRT over PyTorch | `0.368122062750279` |
| Median latency ratio, TensorRT over PyTorch | `0.365965509600977` |
| Mean reciprocal latency ratio, PyTorch over TensorRT | `2.71649026556272` |
| Median reciprocal latency ratio, PyTorch over TensorRT | `2.73249793700595` |

The TensorRT maximum of `95.1917724609375` ms was the first measured trial
after three warmups. It was retained by the frozen protocol; no replacement
trial, sweep, or rerun was performed.

## CUDA Memory

Memory values are PyTorch CUDA allocator snapshots only. They are not Nsight
DRAM counters and do not include TensorRT-managed device allocations. The
engine file itself is `818,910,588` bytes, and TensorRT metadata records
`14,450,688` bytes context device memory.

| Backend | Before allocated | Before reserved | Peak allocated | Peak reserved | After cleanup allocated | After cleanup reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP16 | `2,408,960` | `44,040,192` | `925,023,232` | `960,495,616` | `11,731,456` | `41,943,040` |
| TensorRT FP16 | `10,928,640` | `41,943,040` | `19,359,744` | `44,040,192` | `14,139,904` | `41,943,040` |

All four last-checked PyTorch and TensorRT outputs had finite ratio `1.0` and
shape `[196,2048]`. This benchmark did not rerun or replace the numerical
tolerance analysis from Phase 9.2-C2-R2.

## Board Power

`tegrastats` used a 100 ms interval. For each backend, the scope was backend
setup, warmup, and measurement. These are board/rail measurements, not GPU-only
power and not process-exclusive power. The large PyTorch median-to-mean gap is
consistent with the included model-load/setup interval and should not be read as
measurement-only power.

| Backend | Samples | `VDD_IN` mean | `VDD_IN` median | `VDD_IN` stddev | `VDD_IN` min/max | `VDD_CPU_GPU_CV` mean | `VDD_SOC` mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP16 | `161` | `11077.9254658385` mW | `6140` mW | `6115.21212689388` mW | `5475/18771` mW | `3918.66459627329` mW | `3059.11180124224` mW |
| TensorRT FP16 | `45` | `14440.3333333333` mW | `15876` mW | `5361.14540424475` mW | `6723/20371` mW | `6111.15555555556` mW | `3774.62222222222` mW |

TensorRT collected one default-stream warning about default-stream usage and
zero errors. The warning is runtime advice from the unchanged C1 execution
path, not a build or correctness failure.

## Limitations And Non-Claims

- The result applies only to the deterministic nominal input and exact
  `[784,1536]` / `[1,28,28]` boundary.
- The result does not establish realistic-image, multi-image, video,
  dynamic-token, decode-path, or end-to-end model performance.
- No numerical tolerance, optimization gate, deployment gate, or speedup
  optimization claim was applied.
- CUDA memory is allocator evidence, not total-process GPU memory or a profiler
  DRAM counter.
- Power is board/rail power over backend setup+warmup+measurement, not
  measurement-only or GPU-only power.
- Backend order and cold/warm state were fixed by the frozen protocol; results
  should not be extrapolated to a different ordering or thermal state.
- No optimization, engine rebuild, quantization, model change, CUDA change, or
  persistent environment change occurred.

## Evidence

- Frozen protocol:
  `artifacts/phase9_2D1_20260916T083418Z/benchmark_protocol.json`
- Raw benchmark result:
  `artifacts/phase9_2D1_20260916T083418Z/benchmark_result.json`
- PyTorch power log:
  `artifacts/phase9_2D1_20260916T083418Z/tegrastats_pytorch.log` (local-only
  due to the repository's `*.log` ignore rule)
- TensorRT power log:
  `artifacts/phase9_2D1_20260916T083418Z/tegrastats_tensorrt.log` (local-only
  due to the repository's `*.log` ignore rule)
- Command log:
  `artifacts/phase9_2D1_20260916T083418Z/command_log.md`
- Artifact manifest:
  `artifacts/phase9_2D1_20260916T083418Z/artifact_manifest.json`
- Benchmark script:
  `src/phase9_2D1/run_vision_latency_benchmark.py`

The next action is to stop and await Gate review. No further benchmark, input
sweep, optimization, engine rebuild, quantization, CUDA change, or environment
change is authorized by this report.
