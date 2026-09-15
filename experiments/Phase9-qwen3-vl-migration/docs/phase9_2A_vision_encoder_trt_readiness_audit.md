# Phase 9.2-A Qwen3-VL Vision Encoder TensorRT Readiness Audit

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner explicitly authorized an architecture and TensorRT readiness audit of
the Qwen3-VL vision encoder. The allowed work was source inspection, checkpoint
metadata inspection, operator analysis, export-path identification, and
migration planning.

The audit did **not** perform model instantiation, model execution, ONNX
conversion, a TensorRT engine build, benchmarking, quantization, optimization,
CUDA modification, or environment modification. The Jetson repository and
runtime remain unchanged.

## Gate

**PASS / BOUNDED — MIGRATION_PLAN_ONLY**

The requested architecture inventory and migration plan are complete. This does
not prove TensorRT compatibility. Direct export, parser, build, and runtime
compatibility are `UNKNOWN` because none of those operations were authorized.

## Evidence And Identity

- Repository branch: `phase/09-qwen3vl-migration`.
- Starting HEAD: `e73cf0360a57f83e8faa2a84acfad2ce47bc2ddc`.
- Model: `Qwen/Qwen3-VL-2B-Instruct` at revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- `model.safetensors` SHA-256:
  `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
- `config.json` SHA-256:
  `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b`.
- Runtime evidence from Phase 9.0 and Phase 9.1-A: Python `3.10.12`, PyTorch
  `2.5.0a0+872d972e41.nv24.08`, CUDA `12.6`, TensorRT `10.3.0`, Transformers
  `4.57.3`, and FP16 baseline attention `eager`.
- Transformers source inspected:
  `/home/nvidia/.venvs/jetson-qwen-phase1-hf/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`,
  SHA-256
  `685b10cbc48556343df4e159199c5884d02db9ad227640f0f4fbe75ddf0b2a99`.
- Read-only safetensors metadata found `315` visual tensors and
  `406,957,056` visual parameters. Weights were not loaded into a model and the
  visual module was not executed.

## Vision Encoder Architecture

The audited graph is `model.visual`, not the language decoder. Its top-level
flow is:

1. `Qwen3VLVisionPatchEmbed`: Conv3D, kernel and stride `[2,16,16]`, output
   hidden size `1024`.
2. `fast_pos_embed_interpolate`: interpolated learned absolute position
   embeddings from `nn.Embedding(2304,1024)`.
3. Residual addition.
4. `rot_pos_emb`: data-dependent vision RoPE index construction and lookup.
5. `cos`/`sin` position embeddings.
6. `cu_seqlens` construction from `grid_thw`.
7. 24 `Qwen3VLVisionBlock` layers.
8. Deepstack merger taps after blocks `5`, `11`, and `17`.
9. Final `Qwen3VLVisionPatchMerger`.
10. Return final merged output plus three deepstack outputs.

Each block is LayerNorm, Attention, residual add, LayerNorm, MLP, residual add.
Attention uses fused QKV Linear `[1024 -> 3072]`, RoPE, and output projection
`[1024 -> 1024]`. The MLP is `[1024 -> 4096]`, `gelu_pytorch_tanh`,
`[4096 -> 1024]`. The final merger normalizes the unmerged `1024` channel
state, merges the `2x2` spatial patch dimension to `4096`, then projects
`4096 -> 4096 -> 2048`. Deepstack mergers use post-shuffle LayerNorm on `4096`.

## Operator Inventory And Mapping

| Graph region | Operators | TensorRT mapping | Audit readiness |
| --- | --- | --- | --- |
| Patch embed | view, cast, Conv3D, view | IConvolutionLayer; for image T=2, equivalent Conv2D with `[1024,6,16,16]` weight | Transformable; direct 3D support requires export/build verification |
| Position interpolation | linspace, floor/int, ceil, clip, arithmetic, embedding gather, weighted sum, split/repeat/reshape/concat, host lists | Precompute indices and weights for fixed grid; IGatherLayer plus MatMul/ElementWise | Transformable for fixed grid only |
| RoPE preparation | `.item()`, arange, outer, loops, stack, indexing, flatten | Precompute position IDs/cos/sin constants; Gather plus ElementWise | Transformable for fixed grid only |
| Top-level flow | add, reshape, concat, cos, sin, repeat_interleave, cumsum, pad, loop, conditional taps | ElementWise, shuffle, concat, unary; bake fixed-grid cumsum/repeat constants | Transformable for fixed graph |
| Block norms | LayerNorm | INormalizationLayer or primitive reduction chain | Native/standard |
| QKV and projection | Linear, reshape, permute, unbind, transpose | MatrixMultiplication plus shuffle | Native/standard |
| RoPE application | FP32 cast, multiply, rotate/negate, concat, add, cast back | Unary, ElementWise, concat, shuffle | Transformable; verify precision |
| Variable-length dispatch | dynamic `cu_seqlens`, `.tolist()`, split, Python loop, dispatch, concat | Remove for fixed single image; use one 784-token chunk; variable-length support would need bucketing or plugin | Not ready for direct dynamic export |
| Eager attention | QK MatMul, scale, Softmax FP32, V MatMul | MatrixMultiplication, ElementWise, ISoftMaxLayer, MatrixMultiplication | Transformable |
| MLP | Linear, tanh-GELU, Linear | MatrixMultiplication plus GELU or primitive tanh chain | Transformable; verify GELU semantics |
| Mergers | view, LayerNorm, Linear, GELU, Linear | Shuffle, normalization, MatMul, activation | Native/standard |
| Deepstack taps | conditional module selection, list append, additional outputs | Static graph outputs | Transformable for fixed graph |

No operator was proven unsupported. The phrase "unsupported" is therefore not a
compatibility claim. The direct-export blockers identified below are graph-shape
and control-flow issues, not proof that TensorRT cannot execute the semantics.

## Direct-Export Blockers

- `rot_pos_emb` uses `.item()` and builds position IDs through data-dependent
  host logic.
- `fast_pos_embed_interpolate` uses `.tolist()` and constructs index/weight
  tensors from host lists.
- `cu_seqlens` depends on dynamic `grid_thw`; attention lengths are converted
  to host lists and split per image.
- Attention uses implementation dispatch, Python loops, dynamic splits, and
  concatenation for variable-length chunks.
- Deepstack taps add three Python-list outputs and conditional module lookup.
- Position and RoPE preparation include grid-dependent loops and tensor
  construction, preventing a clean fully dynamic export.

## Recommended Export Path

The safest first migration boundary is a fixed single-image graph:

- `grid_thw = [1, 28, 28]`.
- `pixel_values = [784, 1536]` FP16.
- Patch token sequence length `784`.
- Merged sequence length `196`.
- Attention as one 784-token chunk, with no variable-length split.
- Outputs: final `[196,2048]` plus three deepstack `[196,2048]` tensors.

Implementation should use a static vision wrapper that removes runtime
`grid_thw` and supplies precomputed interpolation, RoPE, and sequence boundary
constants. Attention should first be decomposed into explicit QKV MatMul, scale,
FP32 Softmax, V MatMul, and output projection. Deepstack outputs should become
named graph outputs. Opset 17 is a reasonable initial target because TensorRT
10.3 is the installed runtime and the repository has prior opset-17 experience;
the exact opset must be frozen before export.

The first correctness step, when separately authorized, should compare the
static wrapper against the original PyTorch `model.visual` using the same
FP16 input and real checkpoint weights. Only after wrapper and exported-graph
numerical gates pass should TensorRT parsing or engine building be authorized.

## Expected Conversion Issues

- Dynamic host calls may break tracing or become stale constants.
- Dynamic `grid_thw` can create changing tensor counts and sequence lengths.
- Dynamic `cu_seqlens` attention is not a simple one-to-one native mapping.
- Deepstack outputs increase the graph's output surface and downstream decoder
  integration complexity.
- Conv3D may parse, but an equivalent Conv2D path for the fixed image case is
  the lower-risk first boundary.
- GELU, LayerNorm, RoPE, and Softmax precision may differ between PyTorch and
  TensorRT; numerical gates are required.

## Limitations

- No ONNX graph was produced, so exact exported operator names are `UNKNOWN`.
- No TensorRT network was parsed, so unsupported native layers are not proven.
- No engine was built, so tactic selection, shape behavior, and memory use are
  `UNKNOWN`.
- No benchmark or quantization was performed, and no performance claim is made.
- The baseline used eager attention because Phase 9.1-A found the default SDPA
  path incompatible with NVIDIA PyTorch; this is not a recommendation to export
  attention through SDPA.

## Evidence

- Raw read-only introspection:
  `artifacts/phase9_2A_20260915T161337Z/raw_introspection.json`
- Artifact hash manifest:
  `artifacts/phase9_2A_20260915T161337Z/artifact_manifest.json`
- Module checkpoint metadata:
  `artifacts/phase9_2A_20260915T161337Z/vision_encoder_modules.json`
- Readiness/migration manifest:
  `artifacts/phase9_2A_20260915T161337Z/tensorrt_readiness_manifest.json`
- Installed Transformers source snapshot:
  `artifacts/phase9_2A_20260915T161337Z/vision_transformers_source_snapshot.py`
- Prior environment/TensorRT evidence:
  `artifacts/phase9_0_20260915T125952Z/manifest.json`
- Prior PyTorch/Transformers runtime evidence:
  `artifacts/phase9_1A_20260915T134400Z/runtime_dependency_probe.json`
- Prior FP16 baseline:
  `artifacts/phase9_1B_20260915T234500Z/benchmark_protocol.json`

The next action is to stop and await Gate review. Phase 9.3 or any export/build
step is not authorized by this report.
