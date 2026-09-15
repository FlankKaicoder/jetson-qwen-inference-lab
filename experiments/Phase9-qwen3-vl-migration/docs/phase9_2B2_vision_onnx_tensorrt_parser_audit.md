# Phase 9.2-B2 Qwen3-VL Vision ONNX TensorRT Parser Audit

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized a TensorRT parser compatibility audit for the static
Qwen3-VL vision graph produced by Phase 9.2-B1. The authorized work was TensorRT
ONNX parser invocation, network parsing inspection, unsupported layer/operator
reporting, and parser log collection.

No BuilderConfig was created, no engine build method was called, no engine was
serialized or deserialized, no benchmark or tactic search was run, no
quantization was performed, and no CUDA kernel or persistent environment setting
was modified.

## Gate

**PASS / BOUNDED — TENSORRT_PARSE_ONLY**

TensorRT 10.3.0 parsed the fixed-boundary FP16 opset-17 ONNX graph without a
parser error and populated a network with 6,276 layers. This establishes parser
compatibility for the static graph only. It does **not** establish numerical
correctness, engine buildability, tactic availability, runtime performance, or
dynamic-workload support.

## Input Identity

The input was the existing Jetson-local Phase 9.2-B1 graph:

```text
/tmp/phase9_2b1_20260915T164731Z/qwen3_vl_vision_fp16_opset17.onnx
```

| Field | Result |
| --- | ---: |
| Size | `830,381,691` bytes |
| SHA-256 | `102143ffd1afa1ff798736fdbe274fd2cab98f9e7a97d690a27ebd73c404c4db` |
| Opset | `17` |
| Precision | FP16 |
| Fixed boundary | `pixel_values=[784,1536]` |

The hash was reverified immediately before the parser run and matches the B1
report.

## Runtime And Method

The run used the existing
`/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/bin/python` environment. No
package was installed, upgraded, removed, or persisted.

| Component | Version |
| --- | ---: |
| TensorRT | `10.3.0` |
| ONNX | `1.22.0` |

The parser script at
`experiments/Phase9-qwen3-vl-migration/src/phase9_2B2/parse_vision_onnx.py` used
TensorRT's Python API:

1. Create a `Builder` with a collecting `ILogger`.
2. Create an explicit-batch `Network`.
3. Invoke `OnnxParser.parse_from_file(...)`.
4. Inspect parser errors, network inputs/outputs, and all network layers.
5. Record logger messages and compact evidence.

The script deliberately has no BuilderConfig construction and no engine build
call. It also records the prohibited-operation assertions in its JSON result.

## Parser Result

| Field | Result |
| --- | ---: |
| Parser invoked | `true` |
| Invocation method | `OnnxParser.parse_from_file` |
| Parser exception | `none` |
| Parser returned success | `true` |
| Parser error count | `0` |
| Unsupported operator count at parser boundary | `0` |
| Unsupported operators | `none` |

Because the parser returned success with zero errors, the graph contained no
operator that TensorRT rejected at the parse stage. This does not enumerate the
concrete TensorRT layer implementation for every ONNX node, nor does it prove
that tactic selection or engine construction will succeed.

## Parser Log

The collector captured 26,516 TensorRT logger messages.

| Severity | Count |
| --- | ---: |
| INFO | `12` |
| VERBOSE | `26,504` |
| WARNING | `0` |
| ERROR | `0` |

The full console log is `2,066,375` bytes with SHA-256
`a6d5de8d0dc6855e9589e575c71ea9c6461fa1ce0bc672c704ba1ecc1d0c67e4`; it remains
Jetson-local. The committed non-verbose log contains the 12 INFO records and
shows process-local CUDA/builder-library initialization plus the ONNX file
metadata. No warning or error was present.

## TensorRT Network Summary

| Field | Result |
| --- | ---: |
| Network creation flags | `EXPLICIT_BATCH` |
| Network layers | `6,276` |
| Network inputs | `1` |
| Network outputs | `4` |

The parsed network contract is:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `pixel_values` | `[784,1536]` | `HALF` |
| `final_hidden` | `[196,2048]` | `HALF` |
| `deepstack_0` | `[196,2048]` | `HALF` |
| `deepstack_1` | `[196,2048]` | `HALF` |
| `deepstack_2` | `[196,2048]` | `HALF` |

The one input and four outputs preserve the Phase 9.2-B1 fixed-boundary graph
contract.

### TensorRT Layer Types

| TensorRT layer type | Count |
| --- | ---: |
| ACTIVATION | `24` |
| CAST | `337` |
| CONCATENATION | `96` |
| CONSTANT | `1,684` |
| CONVOLUTION | `1` |
| ELEMENTWISE | `2,353` |
| GATHER | `96` |
| MATRIX_MULTIPLY | `152` |
| NORMALIZATION | `52` |
| SHUFFLE | `1,237` |
| SLICE | `168` |
| SOFTMAX | `24` |
| UNARY | `52` |

The network layer count is larger than the ONNX graph's 2,977 nodes because the
parser expands ONNX operators into TensorRT layers and helper layers. The large
`CONSTANT`, `SHUFFLE`, and `ELEMENTWISE` counts are therefore parser-graph
structure evidence, not an optimization proposal.

## Limitations And Non-Claims

- No ONNX-to-PyTorch numerical comparison was run.
- No TensorRT BuilderConfig, tactic selection, timing, or engine build occurred.
- No serialized or deserialized engine was inspected or executed.
- No dynamic batch, dynamic resolution, multi-image, video, or original dynamic
  `grid_thw` path was tested.
- The result applies only to the exact FP16 opset-17 graph identified above.
- Process-local GPU memory growth shown by TensorRT initialization logs is not a
  benchmark memory peak and must not be used as one.
- Engine buildability and runtime correctness remain `UNKNOWN`.

## Evidence

- Compact parser summary:
  `artifacts/phase9_2B2_20260915T171236Z/parser_audit_summary.json`
- Non-verbose parser log:
  `artifacts/phase9_2B2_20260915T171236Z/parser_log_nonverbose.txt`
- Artifact hash manifest:
  `artifacts/phase9_2B2_20260915T171236Z/artifact_manifest.json`
- Parser script:
  `src/phase9_2B2/parse_vision_onnx.py`

The full raw parser JSON and full console log remain on Jetson at
`/tmp/phase9_2b2_20260915T171236Z/`; their hashes are recorded in the compact
summary.

The next action is to stop and await Gate review. No engine build, correctness
comparison, benchmark, quantization, optimization, or follow-up phase is
authorized by this report.
