# Phase 2.2-B2 — Real Qwen3 Layer 0 Integration via Dual Environment

Date: 2026-09-02
Scope: frozen Qwen3-0.6B Layer 0 only; synthetic hidden inputs; BF16 HF/portable references and FP16 TensorRT.

## Identity and environments

Model `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, snapshot `/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca`. `model.safetensors` SHA256 is `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

Solution C was used. `phase1-hf` provided Transformers 4.57.3, Safetensors 0.8.0 and the real HF layer. `phase2-trt-tools` provided ONNX 1.22.0, Polygraphy 0.53.4 and TensorRT 10.3.0. Both use unchanged NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08` and CUDA 12.6.

## Real semantics and mapping

Layer 0 contains 11 required tensors, 15,730,944 parameters and 31,461,888 raw BF16 bytes. All keys, shapes and hashes are recorded in `layer0_weight_manifest.json`; mapping audit is PASS. Real source semantics are recorded in `transformers_real_semantics.md`: RMSNorm accumulates in FP32, Q/K are normalized before RoPE, cache stores post-RoPE K and projected V in `[B,8,L,128]`, GQA repetition occurs only for attention consumption, and residual/MLP ordering matches `Qwen3DecoderLayer`.

## Oracle chain

Oracle A is real Transformers Layer 0 BF16. Oracle B is the source-faithful portable layer. Their prefill and four decode steps are exactly equal (0 max abs/RMSE for hidden, K, V and new slots), establishing portable semantic fidelity. The same portable BF16 was independently reloaded and reproduced in `phase2-trt-tools`, also with zero error; handoff SHA is recorded in `handoff_integrity.json`. Portable FP16 is obtained by an explicit BF16-to-FP16 cast documented in `weight_cast_manifest.json`.

## TensorRT integration

Portable FP16 Layer 0 exported opset-17 prefill and decode graphs. TensorRT 10.3 parser/build passed for both; serialized engines remain only at `/tmp/phase2_2b2_trt_20260902/`. Prefill input was `[1,8,1024]`, outputs were hidden `[1,8,1024]`, K/V `[1,8,8,128]`; all were finite and CUDA-resident. Decode used dynamic past lengths 8, 9, 10 and 11, producing 9, 10, 11 and 12. Prefix max error was zero at every step; new-slot K max abs was at most 0.5 and V at most 0.001708984375. Full metrics are in `runtime_validation.json`.

Compared with Portable FP16, TensorRT prefill max abs errors were hidden 0.006958, K 0.875, V 0.001953. Decode hidden max abs ranged 0.003906-0.005859; K 0.875; V 0.001953. Compared directly with HF BF16, prefill hidden/K/V max abs were the same bounded values and decode hidden/K/V maxima ranged 0.012695-0.019531, 1.25-1.75 and 0.0029297. This decomposition is recorded separately and is not a bit-exact or production accuracy claim.

## Gates

| Gate | Result |
|---|---|
| B2-1 frozen identity | PASS |
| B2-2 real weight mapping | PASS |
| B2-3 HF oracle | PASS |
| B2-4 portable semantic fidelity | PASS |
| B2-5 dual-environment bridge | PASS |
| B2-6 real-weight TensorRT prefill | PASS (bounded) |
| B2-7 real-weight dynamic decode | PASS (bounded) |
| B2-8 numerical characterization | ACCEPTABLE_FOR_NEXT_FEASIBILITY_STEP |

Overall Phase 2.2-B2: **PASS / BOUNDED**. Decision: `READY_FOR_REAL_MULTILAYER_INTEGRATION`. This is not production readiness: FP16 Reduce/Pow warning remains, no formal tolerance or benchmark was defined, and only one real layer was tested.

## Explicit stop

No full Qwen3 model was exported. No 28-layer TensorRT runtime was built. No embedding, final RMSNorm, LM head or sampling integration was performed. No formal benchmark or Nsight profiler was run. No INT8, INT4 or TensorRT-LLM was built. Phase 2.2-B3, Phase 2.2-C, Phase 2.3 and Phase 3 have not started. The Phase 1 BF16 reference remains unchanged.
