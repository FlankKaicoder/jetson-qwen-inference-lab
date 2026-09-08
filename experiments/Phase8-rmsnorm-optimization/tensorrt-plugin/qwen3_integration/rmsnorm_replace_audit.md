# Phase 8.3-B Integration Map

- Model: Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Checkpoint: Jetson `/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors`, SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Target node: `model.layers.0.input_layernorm`, directly upstream of `model.layers.0.self_attn.q_proj`.
- Contract: hidden size 1024, epsilon `1e-6`, FP16 verification shape `[1,8,1024]`.
- Weight source: checkpoint key `model.layers.0.input_layernorm.weight`; the verification gamma is an FP16 cast of this real BF16 tensor.
- Real input source: `/tmp/phase2_2b2_handoff_20260902/layer0_handoff.pt` key `x`, produced by the frozen Layer 0 Qwen3 handoff; FP16 binary SHA256 is recorded in `sha256sums.txt`.
- Historical full-model source: `/tmp/phase2_2b4_2_20260902T082326Z/{prefill,decode}_28layer.{onnx,engine}` and `/tmp/phase2_3e_20260904T020000Z/work/mixed_{prefill,decode}_28layer.{onnx,engine}`. These artifacts were not modified.
- Replacement route: a new standalone TensorRT network reproduces only the original RMSNorm primitive chain, alongside a second network containing exactly one registered `RMSNormPlugin`; no 28-layer graph or checkpoint was rewritten.
- Parser/build path: TensorRT 10.3 C++ API `IBuilder::buildSerializedNetwork`; plugin registration is loaded with `dlopen` and `getPluginRegistry()`.
