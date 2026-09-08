# Phase 8.3-A TensorRT RMSNorm Plugin Minimal Integration (2026-09-08)

## Objective and Scope

Phase 8.3-A closes the smallest TensorRT integration loop for the Phase 8
RMSNorm work: register an `IPluginV3`, build a synthetic TensorRT engine, run
inference, and compare the output with an explicit FP32 RMSNorm reference. The
network is fixed to FP16 `X [1,8,1024]`, FP16 `gamma [1024]`, and FP16 `Y
[1,8,1024]`. No Qwen3 engine, ONNX graph, model weight, runtime setting, clock,
permission, or system configuration was changed.

## Environment and Evidence

The clean final run used Jetson Orin Nano Super SM87, CUDA 12.6.68, TensorRT
10.3.0, and PyTorch `2.5.0a0+872d972e41.nv24.08`. The repository evidence is
under `artifacts/phase8_3A_20260908T/`; `plugin_stdout.log` and
`pytorch_stdout.log` contain the machine-readable results, while the separate
stderr and exit-code files preserve failure visibility. The run manifest records
the commands and source hashes.

## Plugin Design

`RMSNormPlugin` implements TensorRT 10.3 `IPluginV3` with the Core, Build, and
Runtime capability interfaces. The creator is registered with
`REGISTER_TENSORRT_PLUGIN`, parses the serialized `epsilon` field, and accepts
TensorRT's generated layer name during runtime deserialization. `clone()` is
used for builder/runtime copies and `attachToContext()` returns a clone. The
serialization collection owns one stable `PluginField` member so the serialized
field remains valid across the lifecycle.

The plugin accepts only linear FP16 tensors and the fixed Phase 8.3-A shapes.
`enqueue()` calls `launch_rmsnorm_v2<__half>` from the Phase 8.1
`rmsnorm_v2.cu`; no new RMSNorm kernel was introduced. The demo marks the output
before setting its FP16 type, ensuring the external TensorRT binding is also
FP16.

## Build and Inference

The clean Jetson commands were:

```text
cmake -S tensorrt-plugin -B tensorrt-plugin/build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
cmake --build tensorrt-plugin/build -j2
build/rmsnorm_plugin_demo build/librmsnorm_trt_plugin.so
python3 pytorch_rmsnorm_baseline.py
```

Configuration, build, plugin demo, and PyTorch baseline all returned exit code
0. The only build diagnostic was TensorRT's existing deprecation warning for
`kEXPLICIT_BATCH`.

## Correctness and Latency

| Case | Shape / dtype | Relative L2 | Max absolute error | Mean latency | Stddev | Warmup / repetitions / trials |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| TensorRT `IPluginV3` | `[1,8,1024]` / FP16 | `0.0002063558` | `0.0019426346` | `0.0146312000 ms` | `0.0000246657 ms` | `50 / 200 / 5` |
| PyTorch `torch.nn.functional.rms_norm` | `[1,8,1024]` / FP16 | `0.0003426335` | `0.0046958923` | `0.1691536331 ms` | `0.0025191004 ms` | `50 / 200 / 5` |

The plugin relative-L2 error is below the predefined FP16 gate of `1e-3`.
The latency values are bounded synthetic operator measurements with identical
shape and timing protocol; they do not establish a Qwen3 or end-to-end speedup.

## Limitations

- Fixed shape and FP16 only; no dynamic shape or BF16 support.
- Synthetic two-input network only; no ONNX or Qwen3 integration.
- The comparison measures the complete bounded operator invocation, not a
  production model path or a kernel-only attribution.
- The TensorRT API emits a deprecated explicit-batch warning that does not
  prevent this TensorRT 10.3 demo from building or running.

## Phase 8.3-A Gate

```text
PASS / BOUNDED
```

Registration, builder recognition, engine build, deserialization, inference,
FP16 output binding, and correctness all passed. The result closes only the
minimal TensorRT plugin design objective. Qwen3 replacement and any follow-up
optimization experiment remain outside this phase.
