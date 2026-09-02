# Phase 2.2-C1E - Layerwise Decoder Numerical Divergence Localization

Date: 2026-09-02
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `2b27b7268feec98c3a630c8875d4c37fa903942a`
Result: **FIRST_DIVERGENCE_FOUND**

## Objective

Locate the first layer where the existing B4.2 TensorRT FP16 28-layer decoder diverges from the source-faithful portable FP16 reference for the byte-identical C1 canonical embedding input. This is a diagnostic only; no engine, runtime, model, or quantization code was changed.

## Experimental Setup

The existing Jetson-local `/tmp/phase2_2b4_2_20260902T082326Z/prefill_28layer.engine` was reused. The portable stack loaded the 28 independently staged B4.1 layer files from `/tmp/phase2_2b4_stream_20260902T070000Z`; the canonical input came from `/tmp/phase2_2c1_20260902T090000Z/reference.pt`. Both paths used `B=1,S=8,H=1024`, FP16 hidden states, and INT64 `position_ids=[0..7]`. TensorRT outputs `hidden_l0` through `hidden_l27` were already exposed by the engine, so no rebuild was required. Each enqueue used the current stream followed by explicit synchronization.

The raw evidence is [phase2_2c1e_layerwise_20260902T190000Z.json](../artifacts/phase2_2c1e_layerwise_20260902T190000Z.json). Metrics are computed after FP32 conversion for comparison.

## Layerwise Comparison

| Layer | Shape | Dtype | Max abs | Mean abs | Relative L2 | Cosine |
|---:|---|---|---:|---:|---:|---:|
| 0 | 1x8x1024 | FP16 | 0.0117 | 0.000637 | 0.003788 | 0.999992 |
| 1 | 1x8x1024 | FP16 | 0.0156 | 0.001263 | 0.004683 | 0.999989 |
| 2 | 1x8x1024 | FP16 | 12 | 0.005113 | 0.002169 | 1.000001 |
| 3 | 1x8x1024 | FP16 | 12 | 0.242257 | 0.005725 | 0.999988 |
| 4 | 1x8x1024 | FP16 | 12 | 0.283111 | 0.006414 | 0.999984 |
| 5 | 1x8x1024 | FP16 | 12 | 0.382046 | 0.008286 | 0.999971 |
| 6 | 1x8x1024 | FP16 | 12 | 0.454958 | 0.009759 | 0.999958 |
| 7 | 1x8x1024 | FP16 | 12 | 0.502170 | 0.010695 | 0.999948 |
| 8 | 1x8x1024 | FP16 | 16 | 0.609426 | 0.013195 | 0.999920 |
| 9 | 1x8x1024 | FP16 | 20 | 0.716962 | 0.015368 | 0.999892 |
| 10 | 1x8x1024 | FP16 | 20 | 0.792537 | 0.016914 | 0.999867 |
| 11 | 1x8x1024 | FP16 | 24 | 0.859975 | 0.018695 | 0.999838 |
| 12 | 1x8x1024 | FP16 | 28 | 0.966563 | 0.021283 | 0.999790 |
| 13 | 1x8x1024 | FP16 | 32 | 1.030989 | 0.023081 | 0.999754 |
| 14 | 1x8x1024 | FP16 | 36 | 1.099264 | 0.025392 | 0.999704 |
| 15 | 1x8x1024 | FP16 | 36 | 1.377667 | 0.032262 | 0.999507 |
| 16 | 1x8x1024 | FP16 | 36 | 1.736765 | 0.040325 | 0.999215 |
| 17 | 1x8x1024 | FP16 | 36 | 2.115928 | 0.049716 | 0.998797 |
| 18 | 1x8x1024 | FP16 | 41.8438 | 2.619930 | 0.061896 | 0.998117 |
| 19 | 1x8x1024 | FP16 | 54.0313 | 3.245159 | 0.074102 | 0.997276 |
| 20 | 1x8x1024 | FP16 | 72.1875 | 4.018312 | 0.091325 | 0.995854 |
| 21 | 1x8x1024 | FP16 | 108.0313 | 5.590650 | 0.124389 | 0.992303 |
| 22 | 1x8x1024 | FP16 | 131.6875 | 6.678936 | 0.148046 | 0.989094 |
| 23 | 1x8x1024 | FP16 | 193.25 | 8.140876 | 0.185188 | 0.982899 |
| 24 | 1x8x1024 | FP16 | 229.4844 | 10.156771 | 0.233029 | 0.972940 |
| 25 | 1x8x1024 | FP16 | 267.7188 | 13.525276 | 0.303228 | 0.954413 |
| 26 | 1x8x1024 | FP16 | 224 | 15.258061 | 0.342038 | 0.945732 |
| 27 | 1x8x1024 | FP16 | 4268 | 21.917423 | 2.004247 | 0.534268 |

All 28 outputs are shape `[1,8,1024]`, FP16, finite, and shape-equal. D0 reproduced the B4.2 random-hidden control: max abs `5.0`, mean abs `0.396791`, relative-L2 `0.0201395`, cosine `0.9997966` (`PASS`).

## First Divergent Layer

The first non-zero difference is **Layer 0** (`max_abs=0.01171875`, relative-L2 `0.00378768`, cosine `0.9999921`). The first layer crossing relative-L2 `0.10` is **Layer 21**; the first layer with cosine below `0.99` is **Layer 22**. Divergence then grows progressively, ending at Layer 27 relative-L2 `2.004247`, cosine `0.5342684`.

## Component Analysis

`NOT PERFORMED`. The existing engine exposes only per-layer hidden and K/V outputs; RMSNorm, QK normalization, RoPE, attention, and MLP intermediates are not bindings. Obtaining them would require rebuilding or materially modifying the engine, which is outside C1E.

## Root Cause Hypothesis

The pattern is **Layer-0 onset followed by progressive accumulation**, not a single late-layer crash. This is consistent with a numerical semantic difference inside the FP16 decoder path (for example normalization or fused attention/MLP accumulation), but the layerwise boundary alone cannot distinguish RMSNorm, QK norm/RoPE, attention, or MLP. It does establish that embedding handoff and transport are not the source: C1D measured a byte-identical canonical embedding, and C1E reproduced the same decoder trajectory.

## Limitations and Safety

This is not a repair, benchmark, profiler run, quantization result, or full-generation result. No OOM, exit 137, CUDA execution failure, engine rebuild, or historical artifact modification occurred. Component attribution remains `UNKNOWN`.

## Next Step

Keep Phase 2.2-C1 **BLOCKED** and do not start C2. A separately authorized, smallest-next investigation would expose or compare one layer's internal normalization/attention/MLP intermediates under the same canonical input; that requires a new diagnostic engine artifact and is not performed here.
