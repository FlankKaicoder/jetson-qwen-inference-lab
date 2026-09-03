# Phase 2.3-A Explicit INT8 Q/DQ Feasibility and Quantization Baseline

Date: 2026-09-03

## Objective

Establish an executable TensorRT 10.3 Explicit INT8 Q/DQ path for one real
Qwen3 Linear component and measure quantization-induced component deltas against
the existing TensorRT FP16 component baseline. This is a feasibility and
measurement experiment, not a calibration, full-runtime, performance, or
production precision-policy experiment.

## Frozen Baselines

- Reference A: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, semantic BF16 reference only.
- Reference B: the TensorRT FP16 implementation of the same Linear component;
  all formal quantization deltas below are W8/W8A8 versus this baseline.
- Existing Phase 2.2 and C1 conclusions remain frozen. HF BF16 versus TRT FP16
  is reference context, not quantization error.

## Frozen C1 Limitation

C1 remains `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`. This experiment did not
reopen C1, inspect RoPE, rebuild the decoder, or run a full-stack numerical
investigation.

## Environment

Execution was on a Jetson Orin Nano Super (NVIDIA Jetson Orin Nano Engineering
Reference Developer Kit Super), SM 8.7, CUDA 12.6, NVIDIA PyTorch
`2.5.0a0+872d972e41.nv24.08`, Python 3.10.12, ONNX 1.22.0 and TensorRT Python
10.3.0. `/usr/src/tensorrt/bin/trtexec` reported TensorRT v100300. `nvcc` was
not on PATH. The Jetson checkout was at `a1317a06f83634406bfb732a61f57a698e6aee2d`
and contained pre-existing untracked files; none were changed. The experiment
source was executed from a new `/tmp/phase2_3a_20260903T/` directory.

## Explicit Q/DQ Sanity

The independent 4x4 graph passed ONNX checker, with one `QuantizeLinear` and
two `DequantizeLinear` nodes, TensorRT parser/build, deserialization, CUDA
execution and finite output. Sanity engine size was 16,404 bytes. This proves
environment plumbing only.

## TensorRT Quantization Capability Matrix

| Capability | Result | Evidence |
| --- | --- | --- |
| Weight per-tensor INT8 | SUPPORTED | ONNX checker/parser/build pass; W8 target executes finite |
| Weight per-channel INT8 | SUPPORTED for tested graph | Axis 1 on transposed `[1024,2048]` weight; parser/build pass |
| Activation per-tensor INT8 | SUPPORTED | W8A8 parser/build/execute pass |
| Activation per-channel INT8 | SUPPORTED for parser/build | Axis 2 on `[1,1,1024]`; capability probe build pass, not selected for target execution |
| Symmetric zero-point 0 | SUPPORTED | All selected graphs pass |
| Non-zero zero-point | UNSUPPORTED | Parser error: TensorRT only supports symmetric quantization |
| Scale dtype | FP16 in ONNX initializers; TensorRT constants exposed as Float | Network and inspector evidence |
| Per-channel axis semantics | Weight axis 1; activation axis 2 | Explicit ONNX attributes and parser/build pass |
| MatMul with explicit Q/DQ | SUPPORTED | W8 and W8A8 target graphs execute finite |
| Engine precision introspection | AVAILABLE | TensorRT EngineInspector JSON |

The matrix is a capability probe, not a claim that every tested capability was
executed in the final real-target variant.

## Real Qwen3 Target

The selected component is `model.layers.0.self_attn.q_proj` using the B2 real
Layer 0 handoff. The checkpoint key is
`model.layers.0.self_attn.q_proj.weight`, shape `[2048,1024]`, checkpoint dtype
BF16, and parameter count 2,097,152. The weight SHA256 is
`7592f6e783c400f502653f77d9b483cb03e96c3061af16f538da0d695d21bc53`.
The model safetensors SHA256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
This target was selected because B2 already provides real weights, a saved
activation handoff and an independent component harness; it does not require
the 28-layer runtime or C1 work.

## Canonical Input

The single canonical input is the B2 `layer0_handoff.pt` `x` tensor, stored once
as FP16 for all variants. Source dtype is BF16, target dtype is FP16, shape is
`[1,8,1024]`, and the NPY SHA256 is
`be3abe23fece3b0f3c6eb8163229777eb041ed9728524411beb4666047c93182`.
The source/target values are finite with min `-4.21875`, max `4.125`, mean
`0.0009352206` and standard deviation `1.0008498`. No variant generated a
separate random input.

## Weight Quantization Policy

W8 uses symmetric per-tensor INT8 quantization, `zero_point=0`, FP16 scale
`0.00507354736328125`, and q-range `[-116,127]`. The raw INT8 payload is
2,097,152 bytes versus 4,194,304 BF16/FP16 bytes; scale plus zero-point
metadata is 3 bytes in the theoretical accounting, for a 1.999997x ratio.

## Activation Feasibility Scale

W8A8 uses the canonical-input symmetric absmax feasibility scale
`0.03321850393700788`, `zero_point=0`, FP16 scalar. This is explicitly
`FEASIBILITY_SCALE; NOT_FINAL_CALIBRATION_POLICY`; no calibration sweep was run.

## FP16 Baseline

The FP16 graph passed ONNX checker, TensorRT parser/build, deserialization and
CUDA execution with finite `[1,8,2048]` output. Output distribution was min
`-7.125`, max `7.22265625`, mean `0.0013887788`, standard deviation `0.9918699`,
with zero NaN and Inf values.

## W8-QDQ

The W8 graph uses the real INT8 weight followed by explicit weight DQ and the
same FP16 activation. ONNX checker, TensorRT parser/build and execution passed;
output was finite. The target MatMul inspector input was Half and its tactic was
an FP16 GEMM, so this result is `QDQ_EXECUTION_PROVEN` but
`INT8_COMPUTE_NOT_PROVEN` for W8 weight-only.

## W8A8-QDQ

The W8A8 graph uses activation Q/DQ plus weight Q/DQ with the same canonical
input. ONNX checker, TensorRT parser/build and execution passed; output was
finite. The selected TensorRT EngineInspector fusion layer had Int8 activation
and Int8 weight inputs and tactic
`sm80_xmma_gemm_i8f32_i8i32_f32_tn_n_tilesize32x64x64_stage6_warpsize2x2x1_tensor16x8x32_by_fusion_tactic`.
This is the allowed engine evidence for `INT8_COMPUTE_PROVEN` for this target
graph/profile.

## Quantization-Induced Numerical Delta

Compared with the same-input TRT FP16 output:

| Variant | Max abs | Mean abs | RMSE | Relative L2 | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| W8-QDQ | 0.16796875 | 0.03713523 | 0.04654559 | 0.04692697 | 0.99889922 |
| W8A8-QDQ | 0.18164062 | 0.03786176 | 0.04748704 | 0.04787614 | 0.99885541 |

Both outputs were finite with zero NaN and Inf values. These are component
deltas only; they are not full-model accuracy claims.

## Weight Reconstruction Error

Original BF16 weight versus dequantized INT8 weight was finite with max abs
`0.00390625`, mean abs `0.0012666553`, RMSE `0.0014643172`, relative L2
`0.04668706` and cosine `0.99854684`. Original weight statistics were min
`-0.58984375`, max `0.64453125`, mean `-0.0000274830`, standard deviation
`0.03134126`; dequantized statistics were min `-0.58853149`, max `0.64434052`,
mean `-0.0000248506`, standard deviation `0.03136690`.

## Output Distribution

| Variant | Min | Max | Mean | Std | Finite |
| --- | ---: | ---: | ---: | ---: | --- |
| FP16 | -7.125 | 7.22265625 | 0.00138878 | 0.9918699 | true |
| W8 | -7.15234375 | 7.2421875 | 0.00109037 | 0.9928526 | true |
| W8A8 | -7.14453125 | 7.2109375 | 0.00113953 | 0.9932798 | true |

## Engine Evidence

| Variant | ONNX bytes | Engine bytes | Parser/build | Execution | Precision evidence |
| --- | ---: | ---: | --- | --- | --- |
| FP16 | 4,194,486 | 4,199,860 | PASS/PASS | PASS, finite | Half input and FP16 GEMM tactic |
| W8-QDQ | 2,097,468 | 4,200,820 | PASS/PASS | PASS, finite | DQ output Int8 then Half GEMM; compute not proven |
| W8A8-QDQ | 2,097,684 | 2,196,940 | PASS/PASS | PASS, finite | Int8/Int8 fusion and explicit INT8 GEMM tactic |

Engine size is not runtime GPU memory, and a component engine does not prove
full-runtime memory savings.

## Memory / Storage Evidence

Recorded `MemAvailable` was 5,231,095,808 bytes at start and 3,222,745,088
bytes at completion. No OOM or exit 137 occurred. Maximum process RSS was not
captured (`UNKNOWN`). No latency, throughput, power, speedup, Nsight, SASS or
formal benchmark claim is made.

## Gate

`Phase 2.3-A = PASS`.

- Minimal Explicit Q/DQ sanity: PASS.
- Real Qwen3 component and frozen checkpoint: PASS.
- Same canonical input and FP16 baseline: PASS.
- W8-QDQ build/execute/reconstruction/delta: PASS.
- W8A8-QDQ build/execute/delta: PASS.
- Target W8A8 engine evidence: `INT8_COMPUTE_PROVEN`.
- W8 weight-only classification: `INT8_COMPUTE_NOT_PROVEN`.
- No OOM, exit 137, raw artifact overwrite, C1 reopening, Nsight or benchmark.

## Limitations

This is one real Linear, one deterministic activation, one feasibility scale and
one TensorRT profile. Per-channel probes are parser/build capability evidence,
not a calibration or sensitivity result. W8A8 engine evidence is specific to
the observed target graph/profile. No 28-layer quantized runtime, semantic
generation validation, final calibration policy, or performance conclusion is
established.

## Next Authorized Boundary

The next boundary is `Phase 2.3-B - Calibration / Activation Range Audit`.
It was not started by this experiment.

## Artifacts

All committed evidence is under
`experiments/Phase2-qwen3-quantization/artifacts/phase2_3a_20260903T173658Z/`:

- `start_audit.txt`, `environment.json`, `qdq_capability.json`, `sanity_validation.json`
- `target_audit.json`, `canonical_input.npy`, `canonical_input.json`
- `quantization_scales.json`, `weight_reconstruction.json`, `numerical_comparison.json`
- `output_distribution.json`, `engine_summary_fp16.json`, `engine_summary_w8.json`, `engine_summary_w8a8.json`
- `memory_trace.json`, `final_validation.json`, and `output_{fp16,w8,w8a8}.npy`

ONNX and engine binaries remain in the Jetson-local `/tmp/phase2_3a_20260903T/`
directory and are not forced into Git.
