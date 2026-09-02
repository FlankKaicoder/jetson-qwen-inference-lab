# Phase 2.2-C1 Closeout

## Objective

Build a new full 28-layer TensorRT decoder using the C1K-validated RoPE cache precision correction, then validate `input_ids -> embedding -> decoder -> final hidden` and decode cache behavior. Historical B4.2 engines remained read-only controls.

## Implementation

`c1_closeout_corrected_runtime.py` preserves the Qwen3 structure: input RMSNorm, Q/K/V projections, per-head Q/K RMSNorm, rotary half-split, 16/8 GQA, causal attention, output projection/residual, post-attention RMSNorm and SwiGLU MLP. The only changed boundary is RoPE: source-faithful FP32 frequency/trigonometry is precomputed as FP16 cache and gathered using runtime `position_ids`.

## Build Evidence

New Jetson-local directory: `/tmp/phase2_2c1_closeout_20260903T004926/`. Prefill and decode ONNX graphs were each about 842 MB; TensorRT produced new prefill and decode engines of about 852 MB and 853 MB. Parser/build completed with known warnings for Int64 `position_ids`, FP16 normalization and large protobuf messages. No B4.2 artifact was overwritten.

## Numerical Results

| Tensor | Original B4.2 control vs portable | Corrected engine vs portable | Corrected vs original |
|---|---:|---:|---:|
| Layer 0 hidden | 0.0037876803 | 0.0038153231 | 0.0018616138 |
| Layer 27 hidden | 2.0042469501 | 2.0046324720 | 0.00961355399 |

All outputs were finite and shape-correct. The corrected cache did not remove the full-stack numerical drift; it slightly worsened the Layer 0 hidden relative-L2 versus the B4.2 control for this full export, despite the isolated C1K micrograph improvement.

## Integration And Decode

The exact C1 embedding engine output was byte-identical to the saved C1 FP16 reference (`relative_l2=0`). Feeding that output to the corrected decoder completed successfully. Decode steps 8->9->10->11->12 passed finite CUDA output, exact K/V prefix preservation and per-layer pointer isolation.

## Memory

No OOM or exit 137 occurred. Minimum recorded `MemAvailable` during validation was 1,854,885,888 bytes; maximum recorded RSS was 2,807,032 KB. Engine build memory evidence is in `memory_trace_build.json`.

## Gate

`CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`. Functional runtime and memory checks pass, but the required numerical agreement is not achieved at Layer 27. C1 is closed with the limitation recorded; no C2 was started.

## Artifacts

- `src/phase2_2c1/c1_closeout_corrected_runtime.py`
- `artifacts/phase2_2c1_closeout_20260903T004926/c1_closeout_validation.json`
- `artifacts/phase2_2c1_closeout_20260903T004926/engine_summary.json`
- `artifacts/phase2_2c1_closeout_20260903T004926/memory_trace_build.json`
