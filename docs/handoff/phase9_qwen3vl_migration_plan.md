# Phase 9 Qwen3-VL Migration Startup Plan

Date: 2026-09-15 (Asia/Shanghai)

## Status

**PLANNED / NOT_AUTHORIZED**

This document records the startup audit and the migration plan only. No model
download, benchmark, profiling, code execution, engine build, dependency change,
or model environment change is authorized by this plan. Phase 9.0 requires a
separate ChatGPT Gate authorization.

## Startup Audit

The three-side audit was performed at `2026-09-15T20:46:08+08:00`.

| Side | Branch | HEAD | Working tree | Audit result |
| --- | --- | --- | --- | --- |
| Windows | `phase/09-qwen3vl-migration` created from `phase/08-rmsnorm-optimization` | Starting HEAD `b1dccad72f96578f5326099f1720e739f5c3279d` | Preserved unrelated untracked Phase 2/6 and audit-report paths | Live |
| Jetson | `phase/08-rmsnorm-optimization` | `6fb18773014a43f51c82b91338f2972131a4ce86` | Untracked documentation, registry, Phase 2, Phase 3, and results paths | Live |
| GitHub | `phase/08-rmsnorm-optimization` | `b1dccad72f96578f5326099f1720e739f5c3279d`; remote `HEAD` is `d42ab4aeabc751723a4a2c1036b93a5ed16d3d01` | Not applicable to a remote ref | Live |

Windows and GitHub agree on the Phase 8 branch tip. Jetson is on the same
branch but one commit behind: `6fb1877...` is an ancestor of Windows
`b1dccad...`, and the missing commit is the Phase 8 evidence-extraction commit.
Therefore the Phase 9 startup audit is **NOT_CONSISTENT**, not three-side
synchronization. Jetson was intentionally not fetched, pulled, reset, or synced
during this startup task.

The Jetson and Windows working trees are not clean, but their untracked paths
were recorded and preserved. No untracked file was added to this commit unless
it is explicitly listed in the session closeout.

## Phase 8 Freeze Confirmation

Phase 8 is frozen at the repository evidence level:

- Final gate: `BOUNDED / NO_END_TO_END_SPEEDUP`.
- Report: `experiments/Phase8-rmsnorm-optimization/docs/phase8_4A_qwen3_end_to_end_plugin_impact_report.md`.
- Compact evidence: `experiments/Phase8-rmsnorm-optimization/artifacts/phase8_4A_20260909T/`.
- Experiment index and registry identify Phase 8.4-A as complete and stop
  further RMSNorm optimization.

The end-to-end comparison replaced only Qwen3 Layer 0 `input_layernorm`.
Layer 0 relative-L2 passed the bounded `1e-3` node gate, but final
`hidden_l27` equivalence was `INCONCLUSIVE`, and plugin Prefill/Decode latency
was `+2.96%`/`+2.13%` slower than baseline. This is a valid reason to stop
RMSNorm expansion, not a license to claim an optimization or continue tuning.

## Objective And Boundary

Phase 9 migrates the deployment-analysis workflow from Qwen3-0.6B LLM to the
owner-selected candidate `Qwen/Qwen3-VL-2B-Instruct` on Jetson. The comparison
target is the owner's existing RK3588 multimodal deployment. The goal is
AI-infra/deployment analysis, not algorithm innovation.

No committed repository evidence currently defines a Qwen3-VL local model path,
revision, checkpoint hash, tensor inventory, supported runtime version, or
on-device dependency set. Unless and until Phase 9.0 records those facts, each
such field is `UNKNOWN`. Candidate-name familiarity or upstream documentation
must not be treated as on-device evidence.

## Phase 9.0: Model And Environment Audit

**Status: PLANNED / NOT_AUTHORIZED**

### Required inputs to freeze

1. Candidate model identity:
   - Repository identifier and exact revision.
   - Local path, file manifest, and SHA-256 for config/tokenizer/weights.
   - Safetensors metadata and tensor-name inventory.
   - Total and per-module parameter count.
2. Vision encoder structure:
   - Layer count, hidden size, attention/MLP dimensions.
   - Patch, merge, temporal, and positional-processing parameters.
   - Vision-to-text projector interface and output shape.
3. Decoder structure:
   - Layer count, hidden size, Q/KV head layout, head dimension, GQA ratio.
   - RMSNorm/RoPE semantics and epsilon/theta values.
   - Vocab size, special image/video token IDs, and embedding tying.
4. Input contract:
   - Image and video preprocessing pipeline, resizing/padding rules.
   - Vision token embedding path and text prompt construction.
   - Fixed prefill and decode shapes for the first benchmark.
5. Runtime dependency audit:
   - OS/JetPack/L4T, CUDA, cuDNN, PyTorch, Transformers/TensorRT versions.
   - Whether the installed Transformers/runtime can load and execute the model.
   - Disk, RAM, process memory, CUDA memory, power mode, and clocks before load.

### Exit gate

Phase 9.0 may close only with a compact machine-readable manifest and Markdown
report. It must distinguish on-device facts from static config facts, record
checksums for model inputs, and explicitly mark unsupported/missing dependencies.
If any mandatory item cannot be verified, the gate is `UNKNOWN` or `BLOCKED`.

## Phase 9.1: PyTorch FP16 Baseline

**Status: PLANNED / REQUIRES_PHASE_9_0_GATE**

### Purpose

Establish the first measurable, reproducible image-to-text baseline before any
TensorRT work. The model must remain in FP16 unless Phase 9.0 proves a different
native checkpoint dtype and the owner explicitly approves a deviation.

### Required measurements

For each frozen image and prompt contract:

1. Vision encoder latency.
2. Vision-to-projector handoff latency.
3. Text prefill latency and TTFT.
4. Decode latency per token and total decode time.
5. Output tokens/s.
6. Process and CUDA memory before/peak/after.
7. Board or process power according to the available metering boundary.

### Benchmark requirements

The report must record hardware/software versions, fixed image dimensions,
prompt/input length, output length, seed, sampling rule, warmup count,
repetitions/trials, timing boundary, synchronization, cache state, and all raw
outputs. Correctness evidence must include finite outputs and, where practical,
a deterministic reference or self-consistency check. Memory snapshots alone are
not peak-allocation measurements.

### Exit gate

The baseline is complete only when all requested metrics have raw and summary
evidence, the run is reproducible, and unknown metrics remain `UNKNOWN`. No
optimization claim may be made from the baseline itself.

## Phase 9.2: TensorRT Vision Encoder Analysis

**Status: PLANNED / REQUIRES_PHASE_9_1_GATE**

### Scope

Analyze the FP16 vision encoder path before changing the decoder stack:

1. Patch embedding and merge.
2. Attention/QKV path.
3. MLP path.
4. Layout changes, precision boundaries, and projector interface.

### Required evidence

- TensorRT engine plan/layer inventory and EngineInspector export.
- NSYS end-to-end and per-stage timing.
- NCU kernels only after selecting bounded, representative targets.
- Kernel-level SM, memory, occupancy, and Tensor Core metrics with exact
  workload shapes.

The plan must preserve the distinction between theory and achieved values.
Theoretical occupancy is not achieved occupancy, effective bandwidth is not a
DRAM counter, and device utilization is not occupancy.

### Exit gate

Phase 9.2 is analysis-first. It may propose an optimization target only with a
complete bottleneck chain and measurable baseline. No custom kernel or plugin
implementation is authorized by this startup plan.

## Phase 9.3: TensorRT-LLM Decoder Migration Analysis

**Status: PLANNED / REQUIRES_PHASE_9_1_GATE**

### Scope

Analyze whether the Qwen3-VL text decoder can be migrated or modeled through
TensorRT-LLM without losing the vision interface:

1. KV-cache shape, dtype, ownership, and growth protocol.
2. GQA head mapping and decode attention behavior.
3. Tensor Core GEMM shapes and layout.
4. RoPE/RMSNorm precision semantics.
5. Vision token/projector handoff into the TensorRT-LLM decoder.
6. Supported engine build/runtime path on Jetson.

### Required evidence

- Installed TensorRT-LLM/API audit before implementation.
- Static model/config/weight mapping.
- Minimum viable decoder execution or, if blocked, the exact failure boundary.
- NSYS or IProfiler evidence for prefill and decode where available.

### Exit gate

The result may be `PASS`, `REJECT`, `INCONCLUSIVE`, or `BLOCKED`. In particular,
a missing Jetson/TensorRT-LLM/version intersection must be recorded as blocked
rather than forced through an unsupported environment change.

## Phase 9.4: Optional Quantization Exploration

**Status: NOT_STARTED / REQUIRES_FP16_BASELINE**

Quantization is optional and must not precede the Phase 9.1 FP16 baseline. A
future proposal must first define:

1. FP16 baseline metric and correctness reference.
2. Quantization scheme, per-layer/per-module policy, calibration data, and
   calibration protocol.
3. Engine/runtime compatibility on Jetson.
4. Accuracy gate and performance gate.
5. Memory and power measurement boundary.

No INT8/INT4/W8A8/W4A16 result may be extrapolated from this plan.

## Planned Experiment Sequencing

1. `Phase 9.0`: audit and freeze model/environment facts.
2. `Phase 9.1`: PyTorch FP16 multimodal baseline.
3. `Phase 9.2`: TensorRT vision encoder analysis.
4. `Phase 9.3`: TensorRT-LLM decoder migration analysis.
5. `Phase 9.4`: optional quantization, only after a new Gate authorization.

Each phase requires its own report, compact raw evidence, registry/index update,
project-state update, Git commit, and Gate decision before the next phase.

## Immediate Next Action

Await ChatGPT Gate authorization for Phase 9.0 only. If authorized, the first
execution should be a read-only local-model/environment audit with checksums and
a compact manifest; it must not begin benchmarking or migration.

## Open Risks

- Jetson is one Phase 8 commit behind Windows/GitHub; synchronization requires a
  separate non-destructive, explicitly authorized action.
- No committed Qwen3-VL checkpoint or model manifest exists in this repository.
- Qwen3-VL support in the installed Jetson runtime stack is `UNKNOWN`.
- Memory pressure, long-context behavior, image preprocessing, and projector
  integration may change all downstream benchmarks.
- Quantization and TensorRT-LLM compatibility must not be assumed.
