# Phase 8.7 TensorRT Fusion Boundary Analysis Report

Date: 2026-09-20 (Asia/Shanghai)

## Executive Summary

Phase 8.7 was a repository-only fusion-boundary audit. It read committed
TensorRT EngineInspector and Nsight Systems evidence plus the frozen Phase
8.4-A baseline/plugin comparison. It did not implement a kernel, modify a
plugin, rebuild or execute an engine, or create a new profile.

The requested final classification is:

```text
PASS / BOUNDED / NO_PROVEN_FUSION_OPTIMIZATION_TARGET
```

This does not prove that no fusion opportunity exists. It means the repository
does not currently provide a directly attributable FP16 graph/kernel boundary
that both identifies a fusion loss and proves a measurable optimization
opportunity. The plugin-specific fusion-loss question is:

```text
INCONCLUSIVE
```

The strongest available fusion evidence is from the Mixed/INT8 engines, not the
FP16 target engines. That evidence proves that TensorRT can fuse normalization,
quantization, projection, residual, and activation work in this workload, but it
must not be relabeled as FP16 proof.

## Scope And Authorization

Authorized work was limited to TensorRT graph-fusion boundary analysis for the
Phase 8 extension. Forbidden actions were not performed: no new kernel, no
plugin modification, no Phase 8 file modification, no Phase 8.5 work, no engine
build, no engine execution, no new Nsight collection, no artifact overwrite,
and no destructive Git operation.

## Starting Git State

| Field | Value |
| --- | --- |
| Working directory | `E:\nvidia-qwen` |
| Branch | `phase/08-extension-fusion-analysis` |
| Starting HEAD | `12c6e56fbdda6eb74d8261f769a5a88b2363ae05` |
| Pre-existing untracked paths | Preserved and not staged |
| New extension directory | `experiments/Phase8-extension-fusion-analysis/` |

The starting working tree already contained unrelated untracked audit and
artifact paths. They were neither modified nor staged.

## Environment And Evidence Boundary

The source evidence was collected previously on `nvidia-desktop`, the Jetson
Orin Nano Super host. Existing metadata records TensorRT `10.3.0`, CUDA
`12.6.68`, SM87, Qwen3 with 28 layers and hidden size 1024, and 113 RMSNorm
tensors. No new environment probe was run in Phase 8.7.

Phase 8.7 is a secondary analysis of committed compact evidence. It therefore
inherits three boundaries:

1. The FP16 EngineInspector records are `COMPACT` and do not expose tactic,
   shape, or detailed ONNX metadata.
2. The Mixed/INT8 EngineInspector records are `DETAILED`, but their precision
   is not the FP16 target precision.
3. The committed Nsight Systems files are derived CSV summaries, not raw
   `.nsys-rep` timelines, and there is no plugin-specific Nsight artifact.

## Engine Generation And Tooling Audit

| Item | Evidence |
| --- | --- |
| Phase 8.4-A graph rewrite | `experiments/Phase8-rmsnorm-optimization/scripts/phase8_4a_rewrite_onnx.py` |
| Phase 8.4-A build/measure/compare entry point | `experiments/Phase8-rmsnorm-optimization/scripts/phase8_4a_end_to_end.py` |
| Build method | `trt.OnnxParser` followed by `Builder.build_serialized_network` with `BuilderFlag.FP16` |
| Engine Inspector extraction tool | `experiments/Phase2-qwen3-quantization/src/phase3e/extract_engine_inspector.py` |
| Historical Inspector result | `results/phase3e_kernel_attribution/20260904T121007Z/inspector_v2/layer_inventory.csv` |
| Nsight derived summaries | Phase 3A and Phase 3C committed CSV/JSON artifacts |

The plugin build path loads the plugin shared object before parsing and
building. It records parser errors, engine bytes, build seconds, measurement
warmup/repetition counts, and output metrics. It does not call
`create_engine_inspector()` or save a plugin-engine layer inventory.

## Engine Sources

| Engine | Precision | Source context | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| FP16 Prefill | FP16 | Phase 2.2-B4.2 historical engine | 892,483,860 | `c31f161396c2dc211c418640bce241d3b612e88be0c2da285ec088639ffe960b` |
| FP16 Decode | FP16 | Phase 2.2-B4.2 historical engine | 890,363,012 | `16377216614750dfb126e384f9cc26613c558a528e6dd83ee112c25e1f91e60b` |
| Mixed Prefill | INT8/FP16 | Phase 2.3-E historical engine | 651,671,676 | `3258b0a82fd4884a291e2bf0e3dded179434a9d1a5d034c4fc4e9f772c8964ec` |
| Mixed Decode | INT8/FP16 | Phase 2.3-E historical engine | 650,285,868 | `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c` |
| Phase 8.4-A Baseline Prefill | FP16 | Separate Phase 8.4-A build | 887,464,492 | `a847237c1b456efe79079279ff9ab906953af5329962c46bb53f75a6a8d0c06b` |
| Phase 8.4-A Plugin Prefill | FP16 | Separate Phase 8.4-A build with one Layer 0 RMSNorm plugin | 887,446,972 | `8c7d450cd44d41c1377066d5baabd7be7eac5cec19baf0f7ceb960bbabca572f` |
| Phase 8.4-A Baseline Decode | FP16 | Separate Phase 8.4-A build | 892,739,268 | `ca01030f9bbf82c3ea3c02874aaa242c00de94e30918c54b773f630624f34f59` |
| Phase 8.4-A Plugin Decode | FP16 | Separate Phase 8.4-A build with one Layer 0 RMSNorm plugin | 892,527,348 | `a9ff939b2de30c51926d324b60bf4d2cb435aa827e98a2656770f61e41f976a0` |

Large ONNX and engine binaries remain Jetson-local and are not Git artifacts.

## EngineInspector Layer Counts

| Engine | Inspector layers | GEMM-class | Other recorded categories |
| --- | ---: | ---: | --- |
| FP16 Prefill | 613 | 84 | 529 `UNKNOWN` |
| FP16 Decode | 420 | 84 | 336 `UNKNOWN` |
| Mixed Prefill | 649 | 194 + 28 attention GEMM | 205 `NO_KERNEL`; 222 TensorRT internal |
| Mixed Decode | 699 | 250 | 201 `NO_KERNEL`; 248 TensorRT internal |

These counts are Inspector-layer counts, not CUDA kernel-launch counts. In
particular, `UNKNOWN` means metadata is unavailable in the compact FP16 dump;
it does not mean those layers are unfused, and it does not identify their
runtime kernels.

## Fusion Pattern Table

The Mixed/INT8 detailed Inspector records identify these direct ONNX-node
fusion patterns. Counts are logical engine-layer instances.

| Pattern | Mixed Prefill | Mixed Decode | Interpretation |
| --- | ---: | ---: | --- |
| `input_layernorm` fused with Q/K/V activation quantization | 27 / 28 | 27 / 28 | TensorRT fuses substantial RMSNorm-chain work with downstream QKV quantization. |
| `post_attention_layernorm` fused with gate activation quantization | 27 / 28 | 27 / 28 | TensorRT fuses substantial RMSNorm-chain work with downstream MLP quantization. |
| Q/K/V projection fused | 1 / 28 | 1 / 28 | A direct three-projection fusion is present in the detailed Inspector evidence, but not generally. |
| `o_proj` fused with residual | 28 / 28 | 28 / 28 | Projection plus residual fusion is broadly represented. |
| `down_proj` fused with residual | 28 / 28 | 28 / 28 | Projection plus residual fusion is broadly represented. |
| Gate projection fused with SwiGLU | 27 / 28 | 28 / 28 | Projection plus activation fusion is broadly represented. |
| Q/K norm with RoPE/data movement | 1 / 28 | 55 / 28* | The count is summed across recurring per-layer Inspector rows and is pattern evidence only, not a one-kernel-per-layer claim. |
| Attention MatMul mapped | 135 | 55 | Inspector metadata maps some attention MatMul work to TensorRT layers; exact runtime kernels remain not attributable from these artifacts. |

The asterisk denotes a tool-derived aggregate rather than a structural count.
It should not be compared directly with the other `/28` cells.

The same table cannot be constructed for the FP16 engines. Their compact
Inspector output has `METADATA_ABSENT` in 280 / 613 prefill and 279 / 420 decode
rows, and the remaining rows lack enough metadata to prove the patterns above.

## RMSNorm Boundary

The detailed Mixed/INT8 evidence shows that TensorRT already fuses most
`input_layernorm` and `post_attention_layernorm` chains with downstream
quantized projection work. Therefore the repository does not prove that
RMSNorm is generally excluded from TensorRT fusion.

A plugin can be useful when the goal is to inject a specific custom
implementation, force an explicit operator boundary, expose an intermediate
output, or test a separately implemented reduction/vectorization strategy. That
is a control/implementation reason, not proof of a native-fusion defect.

The Phase 8.4-A one-node plugin was accepted and node-local correctness passed,
but it was slower in the fixed CUDA-event benchmark and no plugin Inspector or
plugin timeline exists. Consequently:

| Question | Result |
| --- | --- |
| Does native TensorRT fail to fuse RMSNorm? | `NOT_PROVEN` |
| Does the custom plugin destroy an existing fusion? | `UNKNOWN` |
| Does the custom plugin bypass fusion? | `INCONCLUSIVE` |
| Is a plugin required for RMSNorm? | `NOT_PROVEN` |

## Plugin Fusion Boundary

Phase 8.4-A replaced only Layer 0 `input_layernorm`, one of 113 RMSNorm nodes.
The measured means were:

| Mode | Baseline mean (ms) | Plugin mean (ms) | Change |
| --- | ---: | ---: | ---: |
| Prefill `[1,8,1024]` | 38.5183 | 39.6601 | `+2.96%` |
| Decode `[1,1,1024]`, past KV length 8 | 41.7803 | 42.6711 | `+2.13%` |

The protocol used 10 warmups and 30 CUDA-event repetitions. These are not
profiler kernel-time means. The engine bytes change slightly, but bytes are not
kernel-count or fusion evidence.

Missing plugin-boundary evidence:

| Item | Status |
| --- | ---: |
| Plugin Engine Inspector layer count | `UNKNOWN` |
| Plugin Engine Inspector fusion-pattern inventory | `UNKNOWN` |
| Baseline/plugin layer-type delta | `UNKNOWN` |
| Baseline/plugin runtime kernel-count delta | `UNKNOWN` |
| Baseline/plugin kernel-launch-count delta | `UNKNOWN` |
| Plugin-specific Nsight Systems timeline | `UNKNOWN` |

Therefore no specific fusion can be named as disappearing after plugin
replacement. The observed latency increase is a real fixed-protocol observation
but its fusion-loss attribution is `INCONCLUSIVE`.

## Nsight Systems Auxiliary Evidence

The available Nsight Systems evidence compares FP16 versus Mixed, not baseline
versus plugin. It is attribution context only.

| Range | Runtime | GPU kernels | GPU kernel time (ms) | CUDA API calls |
| --- | --- | ---: | ---: | ---: |
| A1 Prefill S8 | FP16 | 396 | 47.2 | 4068 |
| A1 Prefill S8 | Mixed | 448 | 42.7 | 4528 |
| A1 Decode Step 0 | FP16 | 422 | 48.9 | 2514 |
| A1 Decode Step 0 | Mixed | 501 | 43.0 | 4516 |

For the persistent kernel summaries:

| Trace summary | Unique kernel names | Instances | Total kernel time (ms) |
| --- | ---: | ---: | ---: |
| FP16 persistent | 40 | 2876 | 310.51568 |
| Mixed persistent | 144 | 3348 | 230.950048 |

The FP16 versus Mixed name-based categories record, for prefill, 141 FP16 TRT
GEMM launches versus 62 Mixed FP16 TRT GEMM launches plus 133 Mixed INT8 TRT
GEMM launches, and 226 versus 224 Myelin-fused launches with unknown mapping.
For decode step 0, the corresponding values are 197 versus 118 FP16 TRT GEMMs,
133 INT8 TRT GEMMs, and 225 versus 250 Myelin-fused launches.

These are kernel-name-aggregation boundaries, not per-Inspector-layer
attribution. `cuda_api_time_ms` includes all CUDA APIs, including module
load/unload setup, and must not be called kernel-launch overhead.

## Kernel Count Answer

| Comparison | Kernel-count difference |
| --- | --- |
| FP16 engine versus plugin engine | `UNKNOWN` |
| FP16 versus Mixed fixed NVTX prefill | `+52` kernels for Mixed, but precision/workload changes confound causal attribution |
| FP16 versus Mixed fixed NVTX decode | `+79` kernels for Mixed, but precision/workload changes confound causal attribution |
| FP16 persistent versus Mixed persistent | Mixed has `+64` more unique kernel names and `+472` more instances across its summary boundary |

The baseline/plugin kernel-count question remains open because no plugin
timeline or plugin Inspector dump was collected.

## Conclusions

1. Proven in Mixed/INT8 reference evidence: TensorRT fuses RMSNorm-chain,
   quantization, projection, residual, and activation work in many instances.
2. Not proven for FP16: the compact FP16 Inspector does not expose enough
   metadata to say which exact Transformer operators remain independent.
3. Not proven for plugin: the Phase 8.4-A plugin has runtime and node-correctness
   evidence but no plugin Inspector or plugin Nsight artifact.
4. No proven target: no evidence names an independent FP16 operator, proves a
   fusion loss caused by the plugin, and demonstrates a beneficial replacement.

Therefore the overall gate is:

```text
PASS / BOUNDED / NO_PROVEN_FUSION_OPTIMIZATION_TARGET
```

The specific plugin boundary remains:

```text
PLUGIN_FUSION_LOSS: INCONCLUSIVE
```

## Validation

- Re-ran the repository-only analyzer once to create the timestamped artifact
  directory; no existing artifact was overwritten.
- Verified engine bytes and hashes against the source Inspector and Phase 8.4-A
  artifacts.
- Verified that the analyzer records its input SHA-256 values in
  `input_manifest.json`.
- Confirmed that no large engine, ONNX, raw `.nsys-rep`, model weight, or
  credential file was added.
- The report deliberately uses `UNKNOWN` and `INCONCLUSIVE` where plugin- or
  FP16-specific evidence is absent.

## Artifacts

| Item | Path |
| --- | --- |
| Analyzer | `experiments/Phase8-extension-fusion-analysis/scripts/analyze_existing_fusion_evidence.py` |
| Evidence directory | `experiments/Phase8-extension-fusion-analysis/artifacts/phase8_7_20260920T071013Z/` |
| Input manifest | `experiments/Phase8-extension-fusion-analysis/artifacts/phase8_7_20260920T071013Z/input_manifest.json` |
| Inspector analysis | `experiments/Phase8-extension-fusion-analysis/artifacts/phase8_7_20260920T071013Z/inspector_analysis.json` |
| Fusion pattern summary | `experiments/Phase8-extension-fusion-analysis/artifacts/phase8_7_20260920T071013Z/fusion_pattern_summary.csv` |
| Norm fusion summary | `experiments/Phase8-extension-fusion-analysis/artifacts/phase8_7_20260920T071013Z/norm_fusion_summary.csv` |
| Nsight summary | `experiments/Phase8-extension-fusion-analysis/artifacts/phase8_7_20260920T071013Z/nsys_summary.json` |
| Plugin boundary summary | `experiments/Phase8-extension-fusion-analysis/artifacts/phase8_7_20260920T071013Z/plugin_boundary_summary.json` |

## Gate And Stop

The audit completed the available repository-driven evidence chain. Its gate is
`PASS / BOUNDED / NO_PROVEN_FUSION_OPTIMIZATION_TARGET`; the plugin-specific
fusion-loss question is `INCONCLUSIVE`.

Stop here. Do not start Phase 8.5 or any fusion follow-up without a fresh
explicit authorization. The minimum useful future evidence would require a
separately authorized plugin Engine Inspector dump and a matched baseline/plugin
Nsight Systems run; neither is authorized by this report.
