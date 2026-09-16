# Phase 9.2-C1 Qwen3-VL Vision ONNX FP16 TensorRT Engine Build

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized TensorRT Vision Encoder engine build feasibility only,
using the Phase 9.2-B1 validated ONNX graph. The allowed work was BuilderConfig
creation, FP16 engine build, build-log collection, and engine metadata
recording.

No engine was executed, no execution context was created, no benchmark or
correctness comparison was run, no quantization or optimization was applied, and
no CUDA kernel or persistent environment setting was modified.

## Gate

**PASS / BOUNDED — TENSORRT_FP16_ENGINE_BUILD_ONLY**

TensorRT 10.3.0 successfully built and serialized the fixed-boundary FP16 vision
graph into an 818,910,588-byte plan. This establishes build feasibility for the
exact static graph and configuration only. It does **not** establish runtime
correctness, performance, numerical stability, dynamic-workload support, or
suitability for deployment.

## Input And Environment

The input was the unchanged Jetson-local Phase 9.2-B1 graph:

```text
/tmp/phase9_2b1_20260915T164731Z/qwen3_vl_vision_fp16_opset17.onnx
```

| Field | Result |
| --- | ---: |
| ONNX size | `830,381,691` bytes |
| ONNX SHA-256 | `102143ffd1afa1ff798736fdbe274fd2cab98f9e7a97d690a27ebd73c404c4db` |
| TensorRT | `10.3.0` |
| Build host | Jetson Orin Nano Super |

The run used the existing
`/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/bin/python` environment. No
package was installed, upgraded, removed, or persisted.

## Builder Configuration

The script used the supported TensorRT 10.3 API
`Builder.build_serialized_network`. It enabled the FP16 builder flag and set
`ProfilingVerbosity.DETAILED` for metadata inspection. No timing cache was
created and no algorithm selector was used.

| Memory pool | Default | Used |
| --- | ---: | ---: |
| `WORKSPACE` | `7,989,903,360` bytes | `1,073,741,824` bytes |
| `TACTIC_DRAM` | `7,208,960,000` bytes | `1,073,741,824` bytes |
| `TACTIC_SHARED_MEMORY` | `1,073,741,824` bytes | `1,073,741,824` bytes |

The workspace and tactic DRAM pools were lowered explicitly because the TensorRT
defaults were close to the board's available memory. These values are not an
optimization claim or a recommended production configuration.

## Build Attempts

Four pre-success invocations stopped before engine output because of
TensorRT-Python API incompatibilities in the script itself:

1. `BuilderFlag` was treated as directly iterable.
2. `BuilderFlag` required an explicit `int()` cast for bitmasking.
3. `Builder.build_engine` is absent in TensorRT 10.3 Python; the supported method
   is `Builder.build_serialized_network`.
4. `IHostMemory` did not expose `.ptr`; the supported path is the Python buffer
   protocol via `memoryview(...)`.

All failed attempt manifests are preserved in the artifact directory. They are
not engine-build failures.

The final invocation succeeded.

## Build Result

| Field | Result |
| --- | ---: |
| Build method | `Builder.build_serialized_network` |
| Build success | `true` |
| Serialized network returned | `true` |
| Build elapsed | `168.94034890400508` seconds |
| Build exception | `none` |
| Engine written | `true` |
| Engine size | `818,910,588` bytes |
| Engine SHA-256 | `aa5c200eb5dcbc5abaef55bc014b5210236fb9ca34394217a024d3796e44823c` |
| Engine path | `/tmp/phase9_2c1_20260916T034956Z/engine/qwen3_vl_vision_fp16.trt` |

The engine remains Jetson-local and is intentionally not committed.

The serialized plan was deserialized once for metadata inspection only. No
execution context was created and no engine was executed. This metadata-only
deserialization is recorded explicitly.

## Engine Metadata

| Field | Result |
| --- | ---: |
| Engine name | `Unnamed Network 0` |
| Engine layers | `209` |
| IO tensors | `5` |
| Optimization profiles | `1` |
| Device memory size v2 | `14,450,688` bytes |
| Streamable weights size | `0` bytes |
| Refittable | `false` |
| Profiling verbosity | `DETAILED` |
| Engine capability | `STANDARD` |
| Hardware compatibility level | `NONE` |
| Tactic sources | `8` |
| Auxiliary streams | `0` |

The deserialized engine preserved the expected FP16 contract:

| Tensor | Mode | Shape | Dtype |
| --- | --- | --- | --- |
| `pixel_values` | input | `[784,1536]` | `HALF` |
| `final_hidden` | output | `[196,2048]` | `HALF` |
| `deepstack_0` | output | `[196,2048]` | `HALF` |
| `deepstack_1` | output | `[196,2048]` | `HALF` |
| `deepstack_2` | output | `[196,2048]` | `HALF` |

All tensors were reported as device-located, linear FP16 format.

## Layer And Tactic Summary

TensorRT's engine inspector reported the following layer families:

| Layer type | Count |
| --- | ---: |
| `CaskConvolution` | `1` |
| `NoOp` | `2` |
| `fusion` | `4` |
| `gemm` | `100` |
| `kgen` | `101` |
| `shape_call` | `1` |

The tactic summary is evidence of what the default builder selected under the
recorded configuration. It is not a benchmark or optimization claim.

Representative selected tactic families include:

| Tactic family | Count |
| --- | ---: |
| `_gemm_mha_v2_*` | `24` |
| `ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_gelu_nn_v1` | `24` |
| `sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x256x32_stage3_warpsize2x4x1_tensor16x8x16` | `4` |
| `sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize256x128x32_stage3_warpsize4x2x1_tensor16x8x16` | `24` |
| `sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize160x128x32_stage4_warpsize2x2x1_tensor16x8x16` | `48` |

The complete per-layer information file is committed in compact form at
`artifacts/phase9_2C1_20260916T034956Z/engine_layer_information.jsonl`. The
full JSON result contains additional tactic names not reproduced in this table.

## Build Logs

The collector captured 31,005 TensorRT logger messages.

| Severity | Count |
| --- | ---: |
| INFO | `25` |
| VERBOSE | `30,980` |
| WARNING | `0` |
| ERROR | `0` |

The selected log contains 597 non-VERBOSE and/or tactic-related messages. It is
committed as
`artifacts/phase9_2C1_20260916T034956Z/build_log_selected.jsonl`. The full
console log is `2,576,908` bytes with SHA-256
`b6cbd007ae826585235d2fda60fd37fa3ef75a9e2064e5158a31ba715f654c96`; it remains
Jetson-local.

The only Python-side console warning was a documented TensorRT deprecation
warning for `engine.device_memory_size` in favor of
`engine.device_memory_size_v2`. Both values were recorded and matched.

## Limitations And Non-Claims

- No execution context was created and no engine was executed.
- No benchmark, correctness comparison, or numerical validation occurred.
- No quantization, optimization, timing cache, tactic forcing, or CUDA kernel
  modification occurred.
- No dynamic batch, dynamic resolution, multi-image, video, or original dynamic
  `grid_thw` path was tested.
- The build applies only to the exact FP16 opset-17 graph and 1 GiB pool
  configuration recorded above.
- Build elapsed time is operational metadata, not performance evidence.
- Device memory size is engine metadata, not a measured runtime peak.
- Runtime correctness and performance remain `UNKNOWN`.

## Evidence

- Final build result:
  `artifacts/phase9_2C1_20260916T034956Z/build_result.json`
- Selected build log:
  `artifacts/phase9_2C1_20260916T034956Z/build_log_selected.jsonl`
- Engine layer information:
  `artifacts/phase9_2C1_20260916T034956Z/engine_layer_information.jsonl`
- Failed attempt manifests:
  `artifacts/phase9_2C1_20260916T034956Z/build_result_attempt1_script_flag_iteration.json`
  through `build_result_attempt4_ihostmemory_write.json`
- Artifact hash manifest:
  `artifacts/phase9_2C1_20260916T034956Z/artifact_manifest.json`
- Build script:
  `src/phase9_2C1/build_vision_onnx_fp16.py`

The next action is to stop and await Gate review. No runtime execution,
correctness comparison, benchmark, quantization, optimization, or follow-up
phase is authorized by this report.
