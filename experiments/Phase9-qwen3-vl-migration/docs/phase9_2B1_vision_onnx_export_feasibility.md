# Phase 9.2-B1 Qwen3-VL Vision Encoder ONNX Export Feasibility

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner explicitly authorized a Qwen3-VL vision encoder ONNX export
feasibility attempt using the Phase 9.2-A fixed single-image boundary:

- `grid_thw = [1,28,28]`
- `pixel_values = [784,1536]`
- patch sequence length `784`
- merged sequence length `196`
- four outputs: final merger plus three deepstack outputs

The work covered preparation of an isolated export script, one substantive ONNX
export attempt, and read-only inspection of the exported graph. The exported
ONNX file was not benchmarked, quantized, optimized, parsed by TensorRT, or
built into an engine. No CUDA state or persistent environment configuration was
changed.

## Gate

**PASS / BOUNDED — ONNX_EXPORT_GRAPH_ONLY**

The fixed-boundary export produced a structurally valid ONNX graph and the
graph inspection completed without dynamic-shape indicators. This does **not**
establish numerical correctness, TensorRT parseability, TensorRT runtime
support, or performance.

## Export Result

The first invocation stopped before model preparation because the script's
fixed-shape guard compared against the flat list `[1,28,28]` instead of the
actual nested tensor list `[[1,28,28]]`. This was a script guard error, not a
model or ONNX exporter failure. The failed attempt is preserved.

After correcting the guard, `torch.onnx.export` succeeded.

| Field | Result |
| --- | ---: |
| Export success | `true` |
| ONNX file written | `true` |
| Opset | `17` |
| Export device | `cuda:0` |
| Export dtype | FP16 |
| ONNX file size | `830,381,691` bytes |
| ONNX SHA-256 | `102143ffd1afa1ff798736fdbe274fd2cab98f9e7a97d690a27ebd73c404c4db` |
| Export elapsed | `30.258315` s |
| CUDA peak allocated | `923,724,800` bytes |
| CUDA peak reserved | `968,884,224` bytes |

The exported file remains Jetson-local at
`/tmp/phase9_2b1_20260915T164731Z/qwen3_vl_vision_fp16_opset17.onnx` and is
intentionally not committed.

## Model Identity

- Model path:
  `/home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203`
- `model.safetensors` size: `4,255,140,312` bytes
- `model.safetensors` SHA-256:
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`
- `config.json` SHA-256:
  `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b`
- Visual module parameters: `406,957,056`
- Visual state-dict load: strict load succeeded with no missing or unexpected keys

The export loaded only `model.visual` weights directly from safetensors into a
`Qwen3VLVisionModel`; the language decoder was not instantiated.

## Static Export Wrapper

The export used `StaticQwen3VLVision` at
`experiments/Phase9-qwen3-vl-migration/src/phase9_2B1/export_vision_onnx.py`.

It implements the Phase 9.2-A plan as follows:

- Precomputes the fixed-grid absolute position embedding as a buffer.
- Precomputes fixed-grid vision RoPE `cos`/`sin` buffers.
- Removes runtime `grid_thw` input.
- Uses one static 784-token attention chunk with no `cu_seqlens` runtime input.
- Decomposes each block attention into QKV projection, explicit Q/K/V
  operations, RoPE, scaled QK MatMul, FP32 Softmax, V MatMul, and output
  projection.
- Preserves the original 24 blocks, final merger, and deepstack taps after
  blocks `5/11/17`.
- Emits four named outputs: `final_hidden`, `deepstack_0`, `deepstack_1`, and
  `deepstack_2`.

Wrapper buffer shapes and dtypes were:

| Buffer | Shape | Dtype |
| --- | --- | --- |
| `pos_embeds` | `[784,1024]` | FP16 |
| `cos_emb` | `[784,64]` | FP16 |
| `sin_emb` | `[784,64]` | FP16 |

The input was a deterministic FP16 `torch.linspace(-1,1,784*1536)` tensor, not
the benchmark image and not a correctness workload.

## Environment Bridge

The Phase 1 HF venv has Transformers/Safetensors but no ONNX package. The
Phase 2 TRT-tools venv has ONNX but lacks Transformers/Safetensors. The export
therefore used a per-process `PYTHONPATH` bridge:

- Exporter Python:
  `/home/nvidia/.venvs/jetson-qwen-phase1-hf/bin/python`
- Added runtime module path:
  `/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/lib/python3.10/site-packages`
- Graph inspector Python:
  `/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/bin/python`

No package was installed, upgraded, removed, or persisted. A small linear
export smoke test first confirmed that the bridged PyTorch 2.5.0a0 + ONNX
1.22.0 process could create an ONNX file.

## Graph Inspection

`onnx.checker.check_model(..., full_check=False)` returned `PASS`.

| Field | Result |
| --- | ---: |
| IR version | `8` |
| Opset | `17` |
| ONNX version used for inspection | `1.22.0` |
| Producer | `pytorch 2.5.0` |
| Graph nodes | `2,977` |
| Initializers | `314` |
| Inputs | `1` |
| Outputs | `4` |
| Dynamic shape indicators | `0` |

The graph is self-contained and uses no external-data file. All `314`
initializers are FP16 (`elem_type=10`).

| Graph input | Shape | Dtype |
| --- | --- | --- |
| `pixel_values` | `[784,1536]` | FP16 |

| Graph output | Shape | Dtype |
| --- | --- | --- |
| `final_hidden` | `[196,2048]` | FP16 |
| `deepstack_0` | `[196,2048]` | FP16 |
| `deepstack_1` | `[196,2048]` | FP16 |
| `deepstack_2` | `[196,2048]` | FP16 |

### Operator Inventory

| ONNX operator | Count |
| --- | ---: |
| Add | 149 |
| Cast | 145 |
| Concat | 96 |
| Constant | 1013 |
| Conv | 1 |
| Div | 4 |
| Erf | 4 |
| Gemm | 104 |
| LayerNormalization | 52 |
| MatMul | 48 |
| Mul | 272 |
| Neg | 48 |
| Reshape | 57 |
| Slice | 96 |
| Softmax | 24 |
| Split | 24 |
| Squeeze | 72 |
| Tanh | 24 |
| Transpose | 144 |
| Unsqueeze | 600 |

No operator fell outside the exported ONNX graph's standard opset. No ONNX-level
unsupported operator was found. Because TensorRT parsing/building was not
authorized, TensorRT-level unsupported operators remain `UNKNOWN`.

The single `Conv` corresponds to the 3D patch embedding. `Gemm` count is
consistent with QKV/projection and MLP matrices across 24 blocks plus four
mergers. `LayerNormalization` count is 48 block norms plus four merger norms.
`Tanh` and `Erf` correspond to the tanh-GELU MLP path and exact-GELU merger
path, respectively.

## Dynamic Shape Assessment

The exported graph has no dynamic shape indicators. The original model's
dynamic `grid_thw`, position interpolation, RoPE construction, and
`cu_seqlens` variable-length attention path were removed by the static wrapper.

This is a feasibility result for the fixed single-image boundary only. It does
not prove that the original dynamic vision graph can export, nor that multi-image
or video workloads are supported.

## Limitations And Non-Claims

- No numerical comparison against the original PyTorch visual module was run.
- No ONNX Runtime execution was performed.
- No TensorRT parser, engine build, timing, quantization, or optimization was
  performed.
- The deterministic synthetic input checked export graph construction only.
- The graph omits the unused learned `visual.pos_embed.weight` from ONNX
  initializers because its fixed-grid output was precomputed into `pos_embeds`.
- The static wrapper is not a general replacement for dynamic Qwen3-VL vision
  inference.
- Memory snapshots are allocator counters, not profiler peaks.

## Evidence

- Export manifest:
  `artifacts/phase9_2B1_20260915T164731Z/export_result.json`
- Guard-failure manifest:
  `artifacts/phase9_2B1_20260915T164731Z/export_result_attempt1_grid_shape_guard.json`
- Graph inspection:
  `artifacts/phase9_2B1_20260915T164731Z/onnx_graph_inspection.json`
- Artifact hash manifest:
  `artifacts/phase9_2B1_20260915T164731Z/artifact_manifest.json`
- Export script:
  `src/phase9_2B1/export_vision_onnx.py`
- Graph inspection script:
  `src/phase9_2B1/inspect_onnx_graph.py`

The next action is to stop and await Gate review. No TensorRT parse/build,
correctness gate, benchmark, quantization, or optimization is authorized by
this report.
