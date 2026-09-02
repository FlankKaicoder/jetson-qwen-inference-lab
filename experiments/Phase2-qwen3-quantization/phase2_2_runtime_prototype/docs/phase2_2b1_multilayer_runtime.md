# Phase 2.2-B1 — Multi-Layer TensorRT Runtime Prototype

Date: 2026-09-02
Scope: four logical layers using one shared synthetic Qwen3-like FP16 decoder engine; no benchmark.

## Motivation and architecture

This bounded prototype tests runtime orchestration rather than model coverage. The scheduler invokes the same Phase 2.2-A prefill/decode engine four times in layer order. TensorRT owns graph computation; the scheduler owns layer order, stream, hidden-state handoff and cache mapping.

Each `LayerState` owns an independent K/V pair with layout `[B,8,L,128]`, plus `layer_id` and `current_position`. No layer shares another layer's cache buffer. All four runtime instances bind CUDA tensor addresses and use the same `torch.cuda.current_stream()` with explicit synchronization.

## Prefill

For `B=1,S=8`, hidden state passes Layer0 -> Layer1 -> Layer2 -> Layer3. Every layer produces a finite, shape-correct hidden output and an independent K/V cache at position 8. Final hidden output is compared with a four-layer PyTorch reference built from the same synthetic layer weights.

## Decode

Four sequential decode steps use positions 8, 9, 10 and 11. Every layer advances its own cache `8->9->10->11->12`; final hidden state and each layer's hidden/K/V are compared with the independent PyTorch reference chain. Results are in `artifacts/phase2_2b1_20260902/multilayer_validation.json`.

## Validation summary

- Prefill: PASS, all four layers and final hidden finite and shape-correct.
- Decode: PASS, 4 steps and 16 layer invocations, expected lengths 9/10/11/12.
- Cache isolation: PASS, every layer's pre-existing TRT prefix has zero max difference after append; layer positions remain synchronized; the artifact records distinct K/V data pointers for all four layers.
- GPU residency: PASS, layer outputs are `cuda:0`; no host payload roundtrip.
- Memory: prefill K/V tensors use 131,072 bytes; current tensors grow to 196,608 bytes at `L=12`. This is a prototype tensor count, not a capacity claim.

PyTorch-vs-TensorRT differences remain informational FP16 numerical differences. Final hidden max absolute error was 0.1973 after prefill and 0.2036/0.2034/0.3588/0.3018 across decode steps. TensorRT's default-stream warning and FP16 Reduce/Pow normalization warning remain open limitations.

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| B1-1 layer orchestration | PASS | Four ordered engine invocations; per-layer hidden records. |
| B1-2 multi-layer KV isolation | PASS | Independent `LayerState` caches and zero prefix mutation for all 4 layers × 4 steps. |
| B1-3 prefill | PASS | `B=1,S=8`, all layers produce K/V at position 8. |
| B1-4 decode | PASS | Four dynamic steps, each layer L→L+1 through length 12. |
| B1-5 GPU residency | PASS | CUDA devices recorded for every layer output; direct tensor binding and explicit stream. |

Overall Phase 2.2-B1: **PASS / BOUNDED**. This validates orchestration for four synthetic layers only. It does not establish 28-layer deployment, production accuracy, performance, memory capacity or full Qwen3 readiness.

## Limitations and stop

No real Qwen3 checkpoint loaded. No full Qwen3 export. No 28-layer deployment. No INT8. No INT4. No TensorRT-LLM. No benchmark. No Nsight. Phase 2.2-B2 NOT STARTED. Phase 3 NOT STARTED. Phase 1 BF16 reference unchanged.
