# Phase 8.3-B Qwen3 TensorRT RMSNorm Plugin Single-Node Integration

Date: 2026-09-08. Branch: `phase/08-rmsnorm-optimization`. Starting HEAD:
`f08ff5913149e7f48ccfe39e4612db88a20d67f1`.

## Integration Objective

This experiment replaces exactly one audited Qwen3 RMSNorm contract with the
Phase 8.3-A TensorRT 10.3 `IPluginV3`: Layer 0
`model.layers.0.input_layernorm`. It proves the isolated node can be built,
serialized, deserialized, and run as a custom plugin using real Qwen3 weights
and a real Qwen3 Layer 0 activation. It does not rebuild, patch, or execute a
full 28-layer Qwen3 graph.

## Qwen3 TensorRT Flow and Node Location

The frozen checkpoint is Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`; its `model.safetensors` SHA256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
The target node is directly upstream of
`model.layers.0.self_attn.q_proj`, has hidden size 1024 and epsilon `1e-6`,
and uses checkpoint key `model.layers.0.input_layernorm.weight`.

The existing full-model provenance artifacts are the historical FP16
`{prefill,decode}_28layer.{onnx,engine}` and Mixed
`mixed_{prefill,decode}_28layer.{onnx,engine}` paths recorded in
`rmsnorm_replace_audit.md`. They were audited by SHA256 only and were not
opened for modification. The input is the frozen Layer 0 handoff
`layer0_handoff.pt` key `x`, converted to the fixed FP16 `[1,8,1024]` node
contract; gamma is an FP16 cast of the real BF16 checkpoint tensor.

## Replacement Design

`qwen3_rmsnorm_integration` builds two new, separate TensorRT networks with
the TensorRT 10.3 C++ API. The original control is the primitive chain
`square -> reduce_mean -> add epsilon -> sqrt -> divide -> gamma`. The
replacement network contains exactly one registered `RMSNormPlugin` with the
same `X`, `gamma`, shape, epsilon, and FP16 output contract. Both serialized
engines are immediately deserialized before inference. The source never
rewrites the checkpoint, ONNX, full-model engine, or more than this one node.

## Reproducible Build and Execution

The Jetson run copied only the Phase 8 plugin source and Phase 8.1 RMSNorm V2
source to `/tmp/phase8_3b_repro_20260908T/`. It invoked
`qwen3_integration/run_phase8_3b_qwen3_integration.sh`, which configured with
`/usr/local/cuda-12.6/bin/nvcc`, built for SM87, and wrote compact logs. The
Jetson checkout remained on its pre-existing branch and its pre-existing
untracked paths were preserved.

Environment: Jetson Orin Nano Super, L4T R36.4.3, CUDA 12.6, TensorRT
`10.3.0.30-1+cuda12.5`, and PyTorch `2.5.0a0+872d972e41.nv24.08`.
`configure_exit_code.txt`, `build_exit_code.txt`, and
`integration_exit_code.txt` are all `0`.

## Serialization, Inference, and Correctness

Plugin registration, both engine builds, serialization, deserialization, and
inference are all `true` in `integration.json`. The correctness gate is
relative L2 `<= 1e-3`.

| Comparison | Result |
| --- | ---: |
| Original vs plugin relative L2 | 0.0003934488 |
| Original vs plugin max absolute error | 0.0009765625 |
| Original vs FP32 reference relative L2 | 0.0003932635 |
| Plugin vs FP32 reference relative L2 | 0.0000165036 |

The plugin comparison passes the declared node-level correctness gate.

## Latency

Timing used 50 warmup enqueues, 200 repetitions per trial, five trials, and
CUDA Events. These are isolated node-engine timings, not Qwen3 end-to-end
latency.

| Metric | Result |
| --- | ---: |
| Original primitive node mean | 0.0131521601 ms |
| Plugin node mean | 0.0147100797 ms |
| Original standard deviation | 0.0000955416 ms |
| Plugin standard deviation | 0.0000284038 ms |
| Plugin minus original | 0.0015579196 ms (11.84%) |

The plugin is slower in this isolated measurement. No performance benefit or
end-to-end Qwen3 speedup is claimed.

## Evidence Integrity and Gate

The initial collection at `phase8_3B_20260908T/` is retained unchanged. Its
JSON reported a successful result but its `integration_exit_code.txt` was `1`,
so it is not used as final gate evidence. A clean isolated reproduction at
`phase8_3B_20260908T_repro/` resolved the inconsistency with all three exit
codes equal to zero and preserved source/input, generated-engine, plugin, and
historical-engine SHA256 values. The generated original and plugin engine
SHA256 values match the initial collection; all four historical full-model
engine SHA256 values were unchanged.

Limitations remain explicit: the control is a standalone reconstruction of the
real node contract, no existing full-model TensorRT graph was parsed and
rewritten, and no whole-model output or throughput was measured.

**Phase 8.3-B Gate: PASS / BOUNDED.** One real Qwen3 RMSNorm node contract was
replaced by the custom plugin in a separately built TensorRT engine, and the
build, serialization, deserialization, inference, and node-level correctness
gate succeeded. Full-model integration and performance claims remain out of
scope.

Final raw evidence:
`experiments/Phase8-rmsnorm-optimization/artifacts/phase8_3B_20260908T_repro/`.
