# Phase 2.1.8 Qwen3 Decoder Block TensorRT Feasibility Audit

Date: 2026-09-01

## Scope

This bounded audit tests one synthetic Qwen3-like decoder block through `PyTorch -> ONNX -> TensorRT parser/build -> FP16 CUDA execution`. It uses synthetic FP16 weights and the recorded Qwen3-0.6B configuration only. No full checkpoint was loaded or exported; no quantization, benchmark, TensorRT-LLM, Phase 2.2 or Phase 3 work was performed.

## Configuration and architecture

The block uses hidden size 1024, 16 query heads, 8 KV heads, head dimension 128, intermediate size 3072, RoPE theta 1,000,000, RMSNorm epsilon 1e-6 and fixed sequence length 8. It contains RMSNorm, Q/K/V projections, RoPE, GQA KV repetition, causal attention with softmax, output projection/residual, RMSNorm, SwiGLU-style MLP and residual output. Dynamic batch profile is `[1, 2]`.

## Evidence

`export_block_onnx.py` produced an opset-17 ONNX graph. `onnx.checker.check_model` passed. The graph contains 201 nodes, including `MatMul`, `Add`, `Mul`, `Div`, `Softmax`, `Transpose`, `Reshape`, `Sin`, `Cos`, `ReduceMean`, `Sqrt`, `Tile`, `Trilu` and `Where`; the complete operator count is recorded in `artifacts/phase2_1_8_20260901/graph_operators.txt`.

TensorRT 10.3 parsed the graph with zero parser errors, built a 31-32 MB FP16 engine, and executed batch 1 and batch 2 with output shape `[B,8,1024]`, FP16 output and finite values. Results are in `execution_result.json` and `execution_batch2.json` (the latter is a runtime-profile record). PyTorch/TensorRT comparisons were finite and shape-equal: batch 1 max absolute error 0.005859375, RMSE 0.000831708; batch 2 max absolute error 0.005859375, RMSE 0.000835744. Relative error is ill-conditioned near zero (366.21 and 122.07) and is informational only, not an accuracy gate.

## Warnings and limitations

TensorRT warned that FP16 Reduce/Pow in layernorm-like arithmetic may overflow and suggested FP32 reduction or `INormalizationLayer`. It also warned that the default CUDA stream may add synchronization overhead. These warnings are retained in `artifacts/phase2_1_8_20260901/parser_warnings.txt`; neither was hidden or converted into a performance claim. The RoPE export emitted a tracer warning because the constant theta is folded. The implementation is a feasibility graph, not production Qwen3 export readiness.

## Gates and decision

- Gate A (existing environment): `PASS`.
- Gate B (Qwen3 config without checkpoint loading): `PASS`.
- Gate C (ONNX export/checker): `PASS`.
- Gate D (TensorRT parser/build/execution): `SUPPORTED` for this synthetic FP16 block and dynamic batch profile.
- Final decision: `Qwen3 TensorRT path: READY` for the bounded synthetic decoder-block feasibility path only.

The next recommendation is an owner-reviewed production export design that separately resolves normalization precision, dynamic sequence/KV-cache semantics and full-model memory constraints. Stop here. No full Qwen3 export, INT8, INT4, TensorRT-LLM, benchmark, Phase 2.2 or Phase 3 was started; the BF16 reference is unchanged.
