# Phase 2.2-A — Single-Layer TensorRT KV-Cache Runtime Integration

Date: 2026-09-02
Scope: bounded synthetic Qwen3-like single decoder layer, FP16 only.

## Scope and non-goals

This checkpoint validates a runtime-shaped prefill/decode contract with explicit K/V tensors. It does not load a Qwen3 checkpoint, export the full model, build a full-model engine, benchmark, profile, quantize, or use TensorRT-LLM.

The synthetic layer uses hidden size 1024, 16 query heads, 8 KV heads, head dimension 128, and intermediate size 3072. K/V cache layout is `[B, 8, L, 128]`; K is Q/K-normalized and RoPE-transformed, while V is the projected value tensor. `position_ids` is an INT64 binding.

## Evidence

- `artifacts/phase2_2a_20260902/build_result.json`: prefill and decode parser/build both PASS.
- `artifacts/phase2_2a_20260902/runtime_validation.json`: CUDA execution, finite outputs, shape checks and independent reference/TRT cache chains.
- `artifacts/phase2_2a_20260902/cache_growth.json`: decode lengths 8→9→10→11→12.
- `artifacts/phase2_2a_20260902/cache_accuracy.json`: prefix immutability and new-slot comparisons.
- `artifacts/phase2_2a_20260902/rmsnorm_precision_probe.json`: FP16 versus FP32 accumulation probe.

## Results

Prefill executed with `B=1,S=8`. Outputs were `[1,8,1024]`, `[1,8,8,128]`, `[1,8,8,128]`, all on `cuda:0` and finite. Relative to the PyTorch synthetic reference, hidden max absolute error was 0.0510, K 1.6836 and V 0.0078125. These are informational FP16 numerical differences, not a full-model accuracy claim.

Decode executed four times with dynamic past lengths 8, 9, 10 and 11. Every output had the expected present length 9, 10, 11 and 12, respectively; all tensors were finite and on `cuda:0`. The TRT input prefix was unchanged in every step (zero max error), and the reference cache chain was kept independent from the TRT chain. New-slot K max absolute error ranged from 1.5527 to 2.3779; new-slot V ranged from 0.00488 to 0.00684. Hidden max absolute error ranged from 0.0477 to 0.0894.

The runtime binds CUDA tensor `data_ptr()` values, allocates outputs on CUDA, uses `torch.cuda.current_stream()` and synchronizes explicitly before returning. TensorRT emitted its default-stream performance warning; this is recorded as a limitation, not hidden.

## Gates

| Gate | Result | Evidence / rationale |
| --- | --- | --- |
| C1 source semantics and RMSNorm policy | PASS | Qwen3 source semantics and FP32-accumulation policy documented; probe saved. |
| C2 prefill integration | PASS (bounded) | ONNX parse/build PASS, CUDA execute PASS, finite shape-correct hidden/K/V outputs. |
| C3 dynamic decode | PASS (bounded) | Four dynamic L→L+1 executions, 8→12, all shape-correct. |
| C4 cache correctness | PASS (bounded) | Independent reference/TRT chains, prefix immutability, new-slot and finite checks saved. Numerical tolerance remains informational for synthetic FP16. |
| C5 CUDA-resident ownership | PASS (bounded) | All inputs/outputs `cuda:0`, direct addresses, explicit stream and no host payload roundtrip. |

Overall Phase 2.2-A: **PARTIAL / BOUNDED PASS**. The bounded integration path is demonstrated, but this does not establish production accuracy, capacity, performance, or full Qwen3 readiness. The default-stream warning and FP16 normalization numerical sensitivity remain open runtime concerns.

## Explicit stop

No real Qwen3 checkpoint was loaded. No full Qwen3 export or TensorRT engine was performed. No formal benchmark or Nsight profiler was run. No INT8, INT4 or TensorRT-LLM work was performed. Phase 2.2-B and Phase 3 have not started. The Phase 1 BF16 reference remains unchanged.
