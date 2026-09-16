# Project Final Report

- Date: 2026-09-16 (Asia/Shanghai)
- Final gate: `PROJECT_COMPLETE`
- Final experiment: Phase 9.3-B2 — Qwen3-VL Decoder-Side Runtime Bottleneck Attribution
- Final experiment commit: `42d77ec804e857d46725314c4a704f4768a8b106`

This report is a repository-driven closeout. It does not rerun experiments, add
new measurements, or upgrade a bounded result into a deployment recommendation.
Raw artifacts and experiment reports remain the controlling evidence when a
summary is ambiguous.

## Project Overview

The Jetson Qwen Transformer AI Infra Optimization Lab studied how to move from
running Transformer models on an embedded NVIDIA platform to understanding the
real inference bottleneck. Work progressed from CUDA fundamentals through
Qwen3 baselines, TensorRT runtime construction, quantization experiments,
operator and kernel attribution, RMSNorm kernels and plugins, and finally a
Qwen3-VL multimodal migration.

The project deliberately treated `BLOCKED`, `INCONCLUSIVE`, `REJECT`, and
bounded `PASS` as valid outcomes. Where evidence could not separate mechanisms,
the records say `UNKNOWN` rather than inventing a conclusion. Consequently, the
final closure proves evidence-chain completeness, not an end-to-end optimized
production deployment.

## Platform And Model Scope

The primary hardware was Jetson Orin Nano Super with CUDA capability `8.7`.
Core software included CUDA `12.6`, TensorRT `10.3.0`, NVIDIA PyTorch
`2.5.0a0+872d972e41.nv24.08`, Transformers `4.57.3`, Nsight Compute
`2024.3.1`, and Nsight Systems `2024.5.4`.

Qwen3-0.6B supported the text-model runtime, quantization, RMSNorm, GEMM, and
decoder-profiling work. Qwen3-VL-2B-Instruct at pinned revision
`89644892e4d85e24eaac8bacfd4f463576704203` supported the final multimodal
migration. Large checkpoints, ONNX files, engines, and raw Nsight reports
remained Jetson-local; Git archived compact JSON, logs, reports, and evidence
manifests.

## Architecture And Evidence System

The repository used a stateless, repository-driven workflow. A new session
recoverable state from Git, `docs/PROJECT_STATE.md`,
`docs/experiment_index.md`, `results/experiment_registry.csv`, experiment
reports, and raw result directories rather than chat history.

The repeating evidence chain was:

```text
Problem and frozen protocol
  -> code/build/execution
  -> correctness or attribution evidence
  -> benchmark/profiling where authorized
  -> analysis and gate
  -> Markdown report
  -> raw artifact manifest
  -> Git commit and push
```

Protocols were frozen before measurement. Benchmarks recorded device, software
versions, model identity, shapes, warmup, repetitions, timing boundary, and
memory/power meaning. Profiler-derived names were not confused with DRAM
counters, achieved occupancy, effective bandwidth, or exclusive device power.

## Experiment Timeline

### Phase 0: CUDA Fundamentals

Exp01 through Exp04 established correctness-first CUDA practice and stable
event timing for vector addition, reduction, matrix transpose, and GEMM/WMMA.
All four major experiments closed `PASS`; their stability and profiling work
also preserved inconclusive causes where microarchitectural attribution was not
resolved.

### Phase 1: Qwen3 Baseline

Phase 1.0 found no natively supported Qwen3 + Jetson SM87 + current-stack
TensorRT-LLM intersection and selected the Hugging Face BF16 reference as a
safe baseline. Phase 1.1 verified pinned checkpoint loading, CUDA forward, and
bounded generation. Phase 1.2 established a reproducible prefill/decode
baseline across input sequence lengths 32/128/512/1024.

### Phase 2: Quantization And TensorRT Runtime

Phase 2 audited TorchAO/TensorRT quantization feasibility, built synthetic
TensorRT enablement graphs, and then progressively integrated real Qwen3
components: Layer 0, four layers, the full 28-layer decoder, embeddings,
final RMSNorm, LM head, greedy sampling, and autoregressive generation.

The full runtime exposed a real numerical limitation. RoPE/cache diagnostics
localized and repaired a major Layer 0 source, but Layer 27 drift remained
unresolved. Phase 2.3 later demonstrated a 133-layer INT8 Q/DQ policy with
INT8 compute proven, but the mixed runtime was slower than FP16 and numerical
equivalence remained bounded.

### Phase 3-4: Runtime And Operator Attribution

Phase 3 showed that a major Mixed-runtime slowdown was host-side context and
module-lifetime behavior, not simply GPU compute. Persistent contexts recovered
most of that gap. Phase 4 recovered TensorRT operator mappings and profiled
GEMM/attention/runtime boundaries, but did not prove a custom-kernel
optimization target.

### Phase 5-7: GEMM, Attention, And Target Reassessment

Phase 5 compared frozen TensorRT evidence with standalone cuBLASLt/CUTLASS and
NCU samples. The event-time gap remained `INCONCLUSIVE` and no tactic defect
was proven. Phase 6 recovered attention-related GEMM surfaces, including
Attention x V, but direct inefficiency and replacement feasibility were not
proven. Phase 7 closed with `NO_PROVEN_CUSTOM_KERNEL_OPTIMIZATION_TARGET`.

### Phase 8: RMSNorm Kernels And Plugin

Phase 8 established PyTorch RMSNorm baselines, implemented V0/V1/V2 CUDA
kernels, captured NCU evidence, and built a TensorRT `IPluginV3`. Isolated
kernels and single-node integration improved or passed bounded correctness, but
the end-to-end one-node plugin experiment was slower and final full-model
equivalence remained inconclusive. The durable conclusion is
`BOUNDED / NO_END_TO_END_SPEEDUP`.

### Phase 9: Qwen3-VL Migration

Phase 9 audited the environment, downloaded and checksum-verified the pinned
Qwen3-VL checkpoint, passed an FP16 eager smoke test, and froze a PyTorch FP16
baseline. The vision encoder was audited and exported as a static FP16 opset-17
ONNX graph at `pixel_values=[784,1536]`, `grid_thw=[1,28,28]`.

TensorRT parsed and built the graph. A first correctness run was blocked by a
non-finite input generation path. The follow-up diagnosis identified direct
CUDA FP16 `linspace` as the input/boundary problem, not reproduced FP16
instability or a specific submodule failure. With direct FP32 input cast to
FP16, all four vision outputs were finite and correctness metrics were
recorded without applying a deployment tolerance.

Isolated vision latency means were PyTorch `225.396496582031` ms versus
TensorRT `82.9734232584635` ms, about `2.7x` under the frozen fixed-workload
protocol. This is not an end-to-end claim.

The unchanged TensorRT FP16 vision engine was then integrated through a runtime
`model.model.visual` adapter. The end-to-end harness produced identical token
sequences for its bounded comparison, but mean total latency and throughput
were essentially unchanged: `2129.2288411458335` ms and
`7.514606813851262` tokens/s for PyTorch versus `2122.2044270833335` ms and
`7.539411177985605` tokens/s for TensorRT. No meaningful end-to-end speedup is
claimed.

Phase 9.3-B1 attributed the remaining fixed-workload time to decode. Phase
9.3-B2 recorded decode as `87.17536311733093%` of generation. Nsight kernel
family attribution measured decode kernel time as `78.3598%` GEMM-class,
`19.5831%` memory-like, and `2.0571%` other. Exact attention share inside
generic kernels, DRAM counters, achieved bandwidth, and power were `UNKNOWN`.
Gate: `PASS / BOUNDED — DECODER_BOTTLENECK_ATTRIBUTION_RECORDED`.

## Key Findings

1. The strongest supported performance result is isolated and bounded: the
   static TensorRT FP16 Vision Encoder reduced the fixed vision-boundary mean
   from about `225.4 ms` to `83.0 ms`.
2. Vision acceleration did not translate into a meaningful end-to-end
   generation speedup in the bounded harness because decode dominated the
   measured workload.
3. Decode-side attribution points to GEMM-class kernels as the largest named
   component of decode kernel time. This does not prove which GEMMs are
   inefficient, what optimization should be applied, or that memory is not a
   limiter.
4. TensorRT-LLM was investigated as a route, not adopted as a successful
   migration result. It remains an unstarted future direction, not a completed
   optimization.
5. Full Qwen3 TensorRT decoder work exposed real numerical divergence that was
   reduced but not eliminated. It should not be represented as production-ready.
6. Quantization feasibility was real in bounded synthetic and component
   experiments, but no project-wide production quantization win was proven.
7. RMSNorm work demonstrated useful isolated engineering, but not model-level
   end-to-end acceleration.

## Technical Stack

| Layer | Final evidence | Notes |
| --- | --- | --- |
| Hardware | Jetson Orin Nano Super, SM `8.7` | `25W` mode in Phase 8/9 protocols |
| CUDA | `12.6` | CUDA events and Nsight attribution |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` | HF/eager reference and visual model |
| TensorRT | `10.3.0` | Parser, builder, runtime, EngineInspector |
| Hugging Face | Transformers `4.57.3` | Qwen3/Qwen3-VL processor and model APIs |
| Profiling | NCU `2024.3.1`, NSYS `2024.5.4` | Permission and boundary limits recorded |
| Models | Qwen3-0.6B; Qwen3-VL-2B-Instruct pinned revision | Large assets stayed Jetson-local |

## Repository Resume Value

The repository is useful as a worked example of disciplined inference
engineering on constrained hardware. It shows how to freeze a benchmark, verify
a model, export/parse/build one bounded component, compare correctness without
pretending production tolerance, isolate a latency boundary, and then honestly
attribute why that isolated gain does not automatically become an end-to-end
win.

It also preserves the process of ruling things out: failed attempts, blocked
permissions, numerical divergence, negative plugin speedups, and
`UNKNOWN` attribution boundaries are retained. Those records are the project's
most transferable artifact.

## Final Limitations

- All performance conclusions are fixed-workload and platform-bounded.
- The end-to-end comparison was functional/attribution evidence, not a
  deployment benchmark or optimization campaign.
- The Vision correctness comparison applied no acceptance tolerance.
- Exact attention share, DRAM counters, achieved bandwidth, and some power
  attribution remain `UNKNOWN`.
- TensorRT-LLM, decoder optimization, quantization productionization,
  FlashAttention, CUDA kernel work, and further input sweeps were not performed
  by Phase 9.4 closure.

## Closure Record

`results/experiment_registry.csv` contains 87 experiment rows plus its header.
The final experiment row remains Phase 9.3-B2 at commit
`e2d109c25202c58b93255b23afffaa0b3892a8b9`. Phase 9.4 is documentation closure
and is intentionally not represented as a new experiment row.

The project closes as `PROJECT_COMPLETE`. Future directions may be proposed
only as new, explicitly authorized work outside this closure.
