# Phase 8 Extension Final Closeout

Date: 2026-09-20

Branch: `phase/08-extension-fusion-analysis`

Starting HEAD for this closeout: `b1cc044b372e95a3e531e77adf57cb1634c7e748`

## 1. Motivation

Phase 8 investigated Transformer operator optimization on the Jetson Qwen3
stack. The practical question was not whether a kernel or plugin can be made
to run, but whether an operator-level intervention survives the complete chain:

```text
CUDA kernel -> TensorRT operator -> Transformer graph -> runtime -> end-to-end inference
```

The extension focused on RMSNorm and fusion because they expose the central
engineering risk: an isolated operator improvement can disappear at graph
boundaries, accumulate numerical error through layers, or fail to reduce the
memory and launch costs that dominate the real workload.

This closeout consolidates existing Phase 8 evidence. It does not start a new
experiment, rebuild an engine, modify a plugin, or change a gate.

## 2. Experimental Pipeline

The evidence chain completed these bounded steps:

```text
Existing profiling and operator attribution
        |
        v
Optimization hypothesis: RMSNorm implementation or fusion may help
        |
        v
CUDA RMSNorm kernels (Phase 8.1) and microarchitectural evidence (Phase 8.2)
        |
        v
TensorRT IPluginV3 integration (Phase 8.3)
        |
        v
One-node Qwen3 replacement and isolated full-graph impact (Phase 8.3-B/8.4-A)
        |
        v
Scaling and numerical stability audit (Phase 8.5)
        |
        v
TensorRT fusion-boundary evidence audit (Phase 8.7)
        |
        v
CUDA RMSNorm + Linear fusion prototype (Phase 8.8-A)
        |
        v
Bounded optimization decision: stop; no proven scalable plugin or fusion gain
```

The chain deliberately separates implementation success from optimization
success. Phase 8.8-B TensorRT fusion-plugin implementation and Phase 8.8-C
Qwen3 runtime validation were not authorized by the 8.8-A dependency gate.

## 3. Phase 8.4 Summary

### Implementation and integration

The Phase 8.3 plugin implemented TensorRT 10.3 `IPluginV3` Core, Build, and
Runtime capabilities. The creator registered `RMSNormPlugin`, serialized the
epsilon field, supported clone/attach-to-context lifecycle calls, and exposed
an FP16 output contract. `enqueue()` called the existing Phase 8.1 V2 CUDA
RMSNorm kernel, which used warp-shuffle reduction and FP16 vectorized access.
No new RMSNorm kernel was introduced in the plugin phase.

The minimal synthetic plugin engine built, serialized, deserialized, enqueued,
and passed its FP16 node gate. The reported synthetic plugin relative-L2 was
`0.0002063558` against the FP32 reference with mean latency `0.0146312 ms`.
The separate real-weight Layer 0 integration also passed build, serialization,
deserialization, enqueue, and node-level correctness. Its plugin-versus-FP32
relative-L2 was `0.0000165036`.

### Layer 0 replacement and full-graph impact

Phase 8.4-A replaced exactly one real Qwen3 node:
`model.layers.0.input_layernorm`, one of 113 RMSNorm nodes. The original ONNX,
checkpoint, and historical engines were preserved; baseline and plugin graphs
were separate builds.

| Evidence | Prefill | Decode |
| --- | ---: | ---: |
| Layer 0 exposed relative-L2 | `0.0006377380` | `0.0008134234` |
| Layer 0 max absolute error | `0.001953125` | `0.001953125` |
| `hidden_l27` relative-L2 | `0.0221897` | `0.0143852` |
| Baseline latency | `38.5183 ms` | `41.7803 ms` |
| Plugin latency | `39.6601 ms` | `42.6711 ms` |
| Plugin change | `+2.96%` | `+2.13%` |

The Layer 0 node contract passed its bounded `1e-3` check, but numerical error
accumulated through the full stack and full-model numerical equivalence was
`INCONCLUSIVE`. The plugin did not improve fixed-shape latency.

Gate:

```text
BOUNDED / NO_END_TO_END_SPEEDUP
```

This phase proved how to put a custom operator into TensorRT and replace one
real node. It did not prove that replacing RMSNorm is a model-level
optimization.

Evidence:

- Report: `experiments/Phase8-rmsnorm-optimization/docs/phase8_4A_qwen3_end_to_end_plugin_impact_report.md`
- Artifacts: `experiments/Phase8-rmsnorm-optimization/artifacts/phase8_4A_20260909T/`
- Plugin design: `experiments/Phase8-rmsnorm-optimization/docs/phase8_3A_tensorrt_plugin_design_report.md`
- Real node integration: `experiments/Phase8-rmsnorm-optimization/docs/phase8_3B_qwen3_single_rmsnorm_integration_report.md`

## 4. Phase 8.5 Summary

Phase 8.5 tested whether the one-node result scaled across a reconstructed
isolated FP16 28-layer Qwen3 graph. The unchanged plugin was applied at scales
`0/1/4/8/28/56`; no fusion or plugin optimization was performed.

### Numerical and generation evidence

| Kind | Scale | Replaced RMSNorm max relative-L2 | `hidden_l27` relative-L2 | Replaced gate | Hidden gate |
| --- | ---: | ---: | ---: | --- | --- |
| Prefill | 0 | `0.0000000` | `0.0000000` | PASS | PASS |
| Prefill | 1 | `0.0005071` | `0.0250546` | PASS | PASS |
| Prefill | 4 | `0.0031609` | `0.0256095` | FAIL | PASS |
| Prefill | 8 | `0.0072243` | `0.0250121` | FAIL | PASS |
| Prefill | 28 | `0.0221710` | `0.0255153` | FAIL | PASS |
| Prefill | 56 | `0.0232615` | `0.0267502` | FAIL | PASS |
| Decode | 0 | `0.0000000` | `0.0000000` | PASS | PASS |
| Decode | 1 | `0.0005001` | `0.0164170` | PASS | PASS |
| Decode | 4 | `0.0026417` | `0.0151888` | FAIL | PASS |
| Decode | 8 | `0.0046953` | `0.0154184` | FAIL | PASS |
| Decode | 28 | `0.0138256` | `0.0153235` | FAIL | PASS |
| Decode | 56 | `0.0148196` | `0.0171933` | FAIL | PASS |

All outputs were finite and the final hidden-state threshold stayed below
`0.05`, but the stricter replaced-node threshold `1e-3` failed from scale 4.
Generation consistency passed only at scale 0 for prefill and scales 0/1 for
decode. Prefill scale 1 and every prefill scale 4+ case failed; decode scales
4/8/28/56 failed.

### Latency and execution evidence

Each row used 10 warmups and 30 CUDA-event repetitions around
`execute_async_v3`. Kernel counts remained near 5.1k for prefill and 5.4-5.5k
for decode rather than decreasing monotonically with replacement scope.

| Kind | Scale 0 mean | Scale 1 mean | Scale 4 mean | Scale 8 mean | Scale 28 mean | Scale 56 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill (ms) | `1248.7623` | `1244.7753` | `1523.9312` | `1421.0274` | `2231.4096` | `2226.7146` |
| Decode (ms) | `2033.2361` | `2056.9066` | `2029.8394` | `2058.8182` | `2238.0595` | `2007.7546*` |

`*` Decode scale 56 used the separately preserved `opt1` build after two
OOM-killed default build attempts. It is not treated as a clean opt0
comparison.

Core conclusion:

```text
PASS / BOUNDED / NO_PROVEN_SCALABLE_PLUGIN_PATH
```

One successful node replacement is not evidence for Transformer-scale plugin
deployment. The numerical contract and generation behavior must hold as scope
grows; here they did not.

Evidence:

- Report: `experiments/Phase8-extension-rmsnorm-scaling-audit/docs/phase8_5_rmsnorm_scaling_audit_report.md`
- Artifacts: `experiments/Phase8-extension-rmsnorm-scaling-audit/artifacts/phase8_5_20260920T075327Z/`

## 5. Phase 8.7 Summary

Phase 8.7 was a repository-only TensorRT fusion-boundary analysis. It did not
implement a kernel, change a plugin, rebuild or execute an engine, or collect
a new profile.

### What the evidence proves

Detailed Mixed/INT8 EngineInspector evidence records repeated fusion patterns:

| Pattern | Mixed Prefill | Mixed Decode |
| --- | ---: | ---: |
| `input_layernorm` with Q/K/V activation quantization | `27/28` | `27/28` |
| `post_attention_layernorm` with gate activation quantization | `27/28` | `27/28` |
| `o_proj` with residual | `28/28` | `28/28` |
| `down_proj` with residual | `28/28` | `28/28` |
| Gate projection with SwiGLU | `27/28` | `28/28` |

These results prove that TensorRT can fuse normalization-chain, projection,
residual, quantization, and activation work in the recorded Mixed/INT8 path.
They are not FP16 proof.

### What remains UNKNOWN or INCONCLUSIVE

- FP16 EngineInspector is compact: Prefill has 613 layers with 84 GEMM and
  529 `UNKNOWN`; Decode has 420 layers with 84 GEMM and 336 `UNKNOWN`.
- Exact FP16 operator fusion boundaries are therefore `UNKNOWN`.
- Plugin EngineInspector layer count and fusion-pattern inventory are `UNKNOWN`.
- Baseline/plugin kernel-count and launch-count deltas are `UNKNOWN`.
- The Phase 8.4-A plugin was slower, but whether a specific native fusion was
  removed or bypassed is `INCONCLUSIVE`.
- The plugin-specific fusion-loss status is `INCONCLUSIVE`; native TensorRT
  failure to fuse RMSNorm is `NOT_PROVEN`.

Gate:

```text
PASS / BOUNDED / NO_PROVEN_FUSION_OPTIMIZATION_TARGET
```

Evidence:

- Report: `experiments/Phase8-extension-fusion-analysis/docs/phase8_7_tensorRT_fusion_boundary_report.md`
- Artifacts: `experiments/Phase8-extension-fusion-analysis/artifacts/phase8_7_20260920T071013Z/`

## 6. Phase 8.8-A Summary

Phase 8.8-A tested a standalone FP16 CUDA fusion prototype at the Qwen3-like
decode shape `M=1/K=1024/N=3072`. Baseline was one RMSNorm kernel plus
`cublasGemmEx`; fusion was one simple RMSNorm + Linear kernel. The experiment
did not modify TensorRT or Qwen3.

| Metric | Baseline | Fusion | Change |
| --- | ---: | ---: | ---: |
| Kernel count | `2` | `1` | `50.0%` lower |
| CUDA Event mean | `0.1058619 ms` | `0.0991772 ms` | `6.3146%` lower |
| CUDA Event median | `0.1092214 ms` | `0.1073771 ms` | `1.6886%` lower |
| NCU L1 global-load sectors | `199872` | `202752` | `1.4409%` higher |
| NCU L1 global-store sectors | `256` | `3072` | `12x` higher |
| NSYS GPU kernel time | `0.256064 ms` | `0.241824 ms` | `5.5611%` lower |

Correctness passed for baseline/reference, fusion/reference, and
fusion/baseline comparisons. The lower mean and one-kernel boundary did not
survive the full predeclared gate: median latency did not improve by 5%, and
the measured global traffic increased. The L1 sector metrics are not DRAM
counters; DRAM traffic remains `UNKNOWN`.

Conclusion:

```text
PASS / BOUNDED / NO_PROVEN_FUSION_BENEFIT
```

Evidence:

- Report: `experiments/Phase8-extension-fusion-performance/docs/phase8_8a_cuda_fusion_report.md`
- Summary report: `experiments/Phase8-extension-fusion-performance/docs/phase8_8_fusion_performance_report.md`
- Artifacts: `experiments/Phase8-extension-fusion-performance/artifacts/phase8_8a_20260920T123247Z/`

## 7. Final Findings

### Proven or bounded successful parts

- A custom CUDA RMSNorm kernel was implemented, correctness-tested, and
  profiled in a bounded synthetic workload.
- A TensorRT 10.3 `IPluginV3` was registered with Core/Build/Runtime
  interfaces, built into an engine, serialized, deserialized, enqueued, and
  checked against an FP32 reference.
- The plugin replaced one real Qwen3 Layer 0 `input_layernorm` node in separate
  FP16 prefill/decode graphs.
- The end-to-end graph build and execution path was functional for that one
  node, with a bounded Layer 0 numerical contract.
- The fusion prototype demonstrated a one-kernel launch boundary reduction in
  the standalone CUDA test.

### Not proven

- Full Transformer RMSNorm replacement with a preserved numerical and
  generation contract.
- A scalable TensorRT RMSNorm plugin deployment across the 28-layer graph.
- A specific FP16 native TensorRT fusion boundary that is lost because of the
  plugin.
- Reduced global memory traffic from the simple fused kernel.
- TensorRT runtime speedup or Qwen3 end-to-end speedup from fusion.
- Any Phase 8.8-B plugin result or Phase 8.8-C Qwen3 runtime result.

The final Phase 8 Extension decision is bounded and negative for optimization,
not a claim that no future implementation could ever help:

```text
NO_PROVEN_SCALABLE_PLUGIN_PATH
NO_PROVEN_FUSION_OPTIMIZATION_TARGET
NO_PROVEN_FUSION_BENEFIT
```

## 8. Engineering Lessons

1. A faster isolated kernel does not guarantee model speedup. Graph boundaries,
   dispatch, tactic selection, memory traffic, and numerical propagation can
   dominate the result.
2. Operator optimization requires graph and runtime validation. Registering an
   `IPluginV3` and passing a node-level correctness check is only an integration
   milestone, not an end-to-end optimization result.
3. Fusion benefit depends on memory behavior and execution context. Removing a
   launch is useful only if the fused implementation also preserves or reduces
   the relevant traffic and remains stable under the target shape and runtime.
4. Optimization decisions require evidence at each boundary. `UNKNOWN` and
   `INCONCLUSIVE` are valid outcomes when Inspector or profiler evidence cannot
   identify a causal path.

## 9. Resume-oriented Summary

Implemented and evaluated a Jetson Orin Nano Super Transformer optimization
chain for Qwen3 RMSNorm: custom CUDA reduction/vectorization, TensorRT 10.3
`IPluginV3` integration, real Layer 0 replacement, 28-layer scaling audit,
TensorRT fusion-boundary analysis, and an FP16 RMSNorm + Linear fusion
prototype. The work established bounded node-level correctness and plugin
integration, but did not prove scalable full-Transformer RMSNorm deployment or
end-to-end fusion speedup. The final evidence-based decisions were
`NO_PROVEN_SCALABLE_PLUGIN_PATH`, `NO_PROVEN_FUSION_OPTIMIZATION_TARGET`, and
`NO_PROVEN_FUSION_BENEFIT`; no production optimization claim is made.

## Closeout Boundary

Phase 8 Extension closeout is complete. Do not implement a fusion plugin,
modify the CUDA prototype, rebuild TensorRT engines, or begin another Phase 8
experiment without fresh human authorization.
