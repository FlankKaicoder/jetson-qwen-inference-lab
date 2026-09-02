# Phase 2.2-B3 — Real Qwen3 Four-Layer TensorRT Stack

Date: 2026-09-02

## Scope

This bounded audit integrates real Qwen3-0.6B decoder layers 0–3 with frozen checkpoint weights. It covers HF BF16 and portable BF16 oracles, explicit BF16-to-FP16 conversion, one TensorRT 10.3 FP16 prefill graph, one TensorRT 10.3 FP16 dynamic decode graph, persistent per-layer KV state, and numerical propagation. Layers 4–27, embedding, final norm, LM head, sampling, performance benchmarking, Nsight, and quantization are out of scope.

## Identity and mapping

Model `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, checkpoint SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. Each of layers 0–3 mapped the same 11 required tensors exactly. Each layer contains 15,730,944 parameters and 31,461,888 BF16 bytes; the four-layer total is 62,923,776 parameters and 125,847,552 bytes. The manifest is `artifacts/phase2_2b3_20260902T055036Z/layers_0_3_weight_manifest.json`.

## Oracle and semantic gates

Real Transformers 4.57.3 layers 0–3 were instantiated individually with eager attention and independent cache indices. The portable source-faithful four-layer stack reproduced the HF BF16 prefill and four decode steps exactly (zero max absolute error for hidden/K/V at every layer). The bounded handoff was reloaded in the TRT-tools environment and reproduced by portable BF16 with zero error. Handoff SHA256 is `ebd6af6100131228d1cddf298e9b42d8c0b4a93d29f7b6bf93c65a2c81d3a7f1`.

## TensorRT stack architecture

The primary architecture is `SINGLE_4_LAYER_STACK_ENGINE`: one prefill graph and one decode graph, each containing layers 0→1→2→3 and exposing intermediate hidden states, per-layer K/V, and attention outputs. ONNX opset 17 checker and TensorRT parser/build passed. Prefill ONNX/engine sizes were 125,937,091/127,228,924 bytes; decode sizes were 125,934,404/127,387,788 bytes. Engine files remain Jetson-local under `/tmp/phase2_2b3_trt_20260902T055036Z/`.

## Runtime validation

Prefill `B=1,S=8` produced four hidden tensors `[1,8,1024]` and four K/V pairs `[1,8,8,128]`; all were finite and CUDA-resident. Decode used one dynamic engine for `8→9→10→11→12`. Every step produced four present caches at `L+1`, finite CUDA tensors, zero prefix mutation, and distinct K/V pointers for all four layers. No host payload roundtrip occurred; execution used `torch.cuda.current_stream()` with explicit synchronization.

## Numerical propagation

The full metrics are in `artifacts/phase2_2b3_20260902T055036Z/layer_numerical_propagation.json` and `attention_propagation.json`. Prefill portable-FP16 versus TensorRT-FP16 relative-L2 hidden error by layer was 0.001092, 0.002512, 0.008068, and 0.007498, with cosine similarity 0.9999995, 0.9999970, 0.9999830, and 0.9999814. K relative-L2 error was 0.002544, 0.001959, 0.003503, and 0.004594; K cosine remained above 0.999989. V relative-L2 error was 0.001793–0.003845. Attention-output relative-L2 error was 0.003205–0.006425 with cosine at least 0.999980. Decode maxima across layers remained bounded: hidden relative-L2 0.00316–0.00423, K 0.00408–0.00446, V 0.00356–0.00372, and attention output 0.00486–0.00707. These observations are limited to four measured layers and are not extrapolated to 28 layers.

The B2 K max-absolute value of 0.875 is contextualized here: at layer 0 prefill, K reference RMS is 9.514, relative-L2 error 0.002544, and cosine 0.9999968. Thus max absolute error alone is not used as a production accuracy criterion.

## Memory and resource accounting

Four-layer FP16 KV storage is 16,384 bytes per token across all layers; measured totals are 131,072 bytes at L=8 and 196,608 bytes at L=12. This is tensor accounting only, not model capacity. Build-time process VmSwap increased from 0 to 84,800 kB in the recorded snapshot; no system swap or power configuration was changed.

## Gates

| Gate | Result |
| --- | --- |
| B3-1 frozen identity and layers 0–3 mapping | PASS |
| B3-2 HF four-layer oracle | PASS |
| B3-3 portable semantic fidelity | PASS |
| B3-4 dual-environment handoff | PASS |
| B3-5 single four-layer stack engine | PASS |
| B3-6 dynamic multi-layer decode | PASS |
| B3-7 numerical propagation | ACCEPTABLE_FOR_28L_FEASIBILITY_STEP |
| B3-8 runtime state integrity | PASS |

Overall Phase 2.2-B3: **PASS / BOUNDED**. Decision: `READY_FOR_28_LAYER_DECODER_STACK_FEASIBILITY`.

## Explicit limits and stop

No layers 4–27 were integrated. No 28-layer decoder stack, embedding, final RMSNorm, LM head, sampling runtime, formal benchmark, Nsight Compute/System, INT8, INT4, TensorRT-LLM, B4, Phase 2.2-C, Phase 2.2-D, Phase 2.3, or Phase 3 was started. The Phase 1 BF16 reference remains unchanged. This report does not claim full-model readiness or 28-layer capacity.
