# Phase 2 Quantization Backend Feasibility and TensorRT Capability Audit

Phase 2.0 status: `BLOCKED`; Phase 2.1 status: `INCONCLUSIVE`

This experiment audits quantization backend feasibility on Jetson Orin Nano Super without quantizing Qwen3 or running a formal quantized benchmark. The Phase 1 BF16 reference remains frozen.

## Scope

- Isolated venv: `/home/nvidia/.venvs/jetson-qwen-phase2-quant`
- NVIDIA PyTorch 2.5.0a0+872d972e41.nv24.08 and CUDA 12.6 are preserved.
- TorchAO `0.12.0` is the only installed quantization package.
- bitsandbytes and TensorRT are survey-only in this phase.

## Current checkpoint

TorchAO wheel installation succeeded with `--no-deps`, but import failed before API discovery because the NVIDIA PyTorch build does not provide `torch._C._distributed_c10d`, required by TorchAO's eagerly imported Float8 modules. No Torch/PyTorch component was replaced. The INT8/INT4 CUDA micro-probes are therefore `BLOCKED_BY_IMPORT_GATE` rather than claimed as supported.

See `docs/phase2_0_quantization_backend_audit.md` and timestamped artifacts for evidence.

## Phase 2.1 TensorRT capability audit

The authorized follow-on audit is documented in `docs/phase2_1_tensorrt_capability_audit.md`. The frozen Jetson environment exposes TensorRT 10.3 FP16 and explicit Q/DQ INT8 construction and synthetic CUDA execution. The ONNX route is blocked because `onnx` is absent and package installation is forbidden. INT4 flags/types are visible, but no public packed weight-only construction path was identified. No Qwen3 quantization or formal performance benchmark was run.

## Phase 2.1.5 TensorRT graph pipeline enablement

Phase 2.1.5 is a bounded synthetic follow-up and is `PASS`. An isolated tools venv at `/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools` supplies ONNX 1.22, GraphSurgeon, Polygraphy and the existing NVIDIA Torch/TensorRT stack. Three FP16 opset-17 graphs (Linear, MLP/GELU, and RMSNorm-like arithmetic plus Linear) passed PyTorch export, `onnx.checker`, TensorRT parsing/building, and CUDA execution for dynamic `M=1` and `M=32` (6/6 cases, finite correctly typed outputs).

This validates only the synthetic `PyTorch -> ONNX -> TensorRT -> CUDA` plumbing. It does not claim Qwen3 exportability, INT8/INT4 support, performance, memory, power, or production readiness. Evidence and the stop-point rationale are in `docs/phase2_1_5_graph_enablement.md` and `artifacts/phase2_1_5_20260901/`. Phase 2.2 and Phase 3 remain unstarted.

## Phase 2.1.8 Qwen3-like decoder block feasibility

A single Qwen3-like block with the recorded Qwen3-0.6B dimensions and synthetic FP16 weights passed opset-17 ONNX checking, TensorRT 10.3 parsing/building and dynamic batch-1/2 CUDA execution. This is a graph feasibility result only. It makes no full checkpoint/export, quantization, benchmark, TensorRT-LLM, Phase 2.2 or Phase 3 claim. The report is `docs/phase2_1_8_qwen3_block_feasibility.md`; FP16 normalization/default-stream warnings remain explicit limitations.

## Phase 2.1.9 Full Qwen3 TensorRT architecture audit

This read-only audit maps the full Qwen3-0.6B architecture, theoretical weight/KV-cache memory and native TensorRT versus TensorRT-LLM runtime contracts without loading weights or building an engine. Gate A/B/C are `PASS` as environment, architecture and theoretical planning evidence. Gate D is `BLOCKED_NEEDS_RUNTIME_WORK`: full-model export, persistent KV cache, prefill/decode scheduling, sampling and memory ownership remain unimplemented. See `docs/phase2_1_9_full_qwen3_tensorRT_architecture_audit.md`.

## Phase 2.2 runtime prototype preparation

The CPU-only preparation under `phase2_2_runtime_prototype/` defines KV-cache ownership, prefill/decode request types, theoretical memory estimation and the future Gate A-D plan. Synthetic byte-layout validation passes; no Qwen3 checkpoint, CUDA/TensorRT execution, engine, benchmark or quantization was run. See `phase2_2_runtime_prototype/docs/phase2_2_runtime_design.md`.

## Phase 2.3-A Explicit INT8 Q/DQ feasibility

Phase 2.3-A selected the real `model.layers.0.self_attn.q_proj` component from
the B2 Layer 0 handoff and used one shared real activation input for FP16,
W8-QDQ and W8A8-QDQ variants. TensorRT 10.3 passed the independent Q/DQ
sanity graph and all three target parser/build/execute checks with finite
outputs. Symmetric zero-point `0` is supported; non-zero zero-points are
rejected by the parser. W8-QDQ is `QDQ_EXECUTION_PROVEN` but its target
arithmetic remains `INT8_COMPUTE_NOT_PROVEN`. W8A8-QDQ has an EngineInspector
Int8/Int8 fusion and explicit INT8 GEMM tactic, so this target/profile is
`INT8_COMPUTE_PROVEN`. Gate: `PASS`. This is a component feasibility result;
no calibration sweep, full 28-layer quantized runtime, benchmark, Nsight or
INT4 work was performed. See `docs/phase2_3a_explicit_qdq_feasibility.md` and
`artifacts/phase2_3a_20260903T173658Z/`.

## Phase 2.3-B Calibration / activation range audit

Phase 2.3-B proved the exact real tensor entering Layer 0 `q_proj` using a
Transformers forward-pre-hook after `input_layernorm`, then collected 24
calibration and 12 disjoint evaluation prompts through the local tokenizer.
GLOBAL_ABSMAX, P99.9, P99.99 and a bounded MSE clipping grid were derived from
calibration data only. The selected policy is `BOUNDED_MSE_CLIP` with scale
`0.0243602362`; held-out activation-only `W8A8 vs W8` relative-L2 is
`0.019526` median, `0.020111` P95 and `0.020117` maximum. The selected dynamic
TensorRT profile retains Int8 activation/weight fusion and an explicit Int8
GEMM tactic. This is a target/corpus-specific calibration result; Phase 2.3-C
has not started. See `docs/phase2_3b_calibration_activation_range_audit.md` and
`artifacts/phase2_3b_20260903T205610Z/`.

## Phase 2.3-C Layer / operator sensitivity

Phase 2.3-C is `PASS / BOUNDED`: all 196 decoder Linear weights were audited,
34 exact-input portable targets were evaluated, and eight deterministic targets
were confirmed in TensorRT. FP16/PT-W8/PT-W8A8 TensorRT paths executed finite;
per-channel PC-W8/PC-W8A8 QDQ parsing is explicitly `BLOCKED` on TensorRT 10.3.
No benchmark, Nsight, INT4, C1 debugging or 28-layer quantized runtime was run.
See `docs/phase2_3c_layer_operator_sensitivity.md` and
`artifacts/phase2_3c_20260903T220600Z/`.
