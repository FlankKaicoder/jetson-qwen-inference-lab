# Final Analysis Version

This branch is a read-oriented snapshot created from
`phase/09-qwen3vl-migration` at repository closure. It does not replace the
historical experiment branches and does not claim to be a production
deployment. Its purpose is to make the complete evidence chain and implementation
material easy to analyze without changing history.

## 1. Repository Purpose

This repository studies Transformer inference optimization on a Jetson Orin
Nano Super platform. It progresses through CUDA fundamentals, Qwen3 baselines,
TensorRT runtime construction, quantization feasibility, operator/kernel
attribution, RMSNorm CUDA kernels and TensorRT plugins, and finally Qwen3-VL
vision-encoder migration.

The project is evidence-first: code running is not treated as an experiment
being complete. Each result is bounded by a frozen protocol, correctness gate,
benchmark boundary, profiler context, and recorded limitations. `REJECT`,
`BLOCKED`, `INCONCLUSIVE`, and bounded `PASS` are preserved as valid outcomes.
The final project gate is `PROJECT_COMPLETE`, meaning evidence closure, not a
proven production-optimized inference product.

Primary entry points:

- `README.md`: project summary and final result pointers.
- `docs/PROJECT_FINAL_REPORT.md`: complete closeout report.
- `docs/FINAL_STATUS.md`: final gate and limitations.
- `docs/PROJECT_STATE.md`: recovery-first project state.
- `docs/experiment_index.md`: experiment-to-report index.
- `results/experiment_registry.csv`: compact experiment registry.

## 2. Overall Architecture

```text
Model
  Qwen3-0.6B / Qwen3-VL-2B-Instruct checkpoints
  (large weights and engines remain Jetson-local)
        |
        v
Conversion / Export
  Model and environment audit
  PyTorch FP16/BF16 reference
  ONNX export, TensorRT parse/build
        |
        v
Runtime
  Hugging Face eager/PyTorch path
  TensorRT component runtime
  KV cache, CUDA Graph, persistent context
  Qwen3-VL visual adapter integration
        |
        v
Optimization / Attribution
  CUDA kernels
  TensorRT plugin
  mixed precision / INT8 QDQ studies
  NSYS + NCU operator/kernel attribution
        |
        v
Benchmark / Evidence
  CUDA-event latency and throughput
  NCU / NSYS compact summaries
  correctness and numerical comparisons
  CSV / JSON / Markdown evidence and gate
```

The strongest bounded result is isolated: a static Qwen3-VL Vision Encoder
TensorRT FP16 path was about `2.7x` faster than the PyTorch FP16 boundary. This
did not become a meaningful end-to-end generation speedup because decode
dominated the measured workload. Decode kernel time was recorded as mostly
GEMM-class, while exact attention share, DRAM counters, achieved bandwidth, and
power remain `UNKNOWN`.

## 3. Code Organization

The repository has two kinds of top-level directories:

- **Index directories**: `cuda/`, `tensorrt/`, `transformer/`, `runtime/`,
  `benchmark/`, and `scripts/` currently contain scope-setting README files.
- **Evidence directories**: `experiments/` and `results/` contain the actual
  code, reports, benchmark artifacts, and profiling summaries.

### `docs/`

**Purpose:** Project state, workflow rules, final reports, experiment index,
phase-level summaries, and handoff records.

**Core files:**

- `docs/README.md`
- `docs/PROJECT_FINAL_REPORT.md`
- `docs/FINAL_STATUS.md`
- `docs/PROJECT_STATE.md`
- `docs/experiment_index.md`
- `docs/project_management.md`
- `docs/handoff/current_state.md`

### `experiments/`

**Purpose:** The main implementation and evidence tree. Each experiment
directory keeps its own README, reports, source code, scripts, benchmark data,
and compact artifacts.

**Core files and subdirectories:**

- `experiments/README.md`
- `experiments/_template/`
- `experiments/Exp01-vector-add/`
- `experiments/Exp02-reduction/`
- `experiments/Exp03-matrix-transpose/`
- `experiments/Exp04-gemm/`
- `experiments/Phase1-qwen3-baseline/`
- `experiments/Phase2-qwen3-quantization/`
- `experiments/Phase5-cuda-feasibility/`
- `experiments/Phase6-attention-matmul-attribution/`
- `experiments/Phase8-rmsnorm-optimization/`
- `experiments/Phase9-qwen3-vl-migration/`

The Phase 3/4/6 runtime and attribution sources are retained under
`experiments/Phase2-qwen3-quantization/src/` because those studies evolved from
the Qwen3 TensorRT runtime work.

### `results/`

**Purpose:** Compact evidence summaries and the experiment registry. Large
model weights, engines, and raw profiler reports are intentionally not
committed.

**Core files:**

- `results/README.md`
- `results/experiment_registry.csv`
- Phase-specific evidence directories such as
  `results/phase3a_runtime_attribution/`,
  `results/phase5a_cuda_feasibility_baseline/`, and
  `results/phase6a_unknown_attention_matmul_attribution/`.

### `cuda/`

**Purpose:** Learning-scope index for CUDA kernels and profiling. This is not
where the actual experiment code lives.

**Core file:** `cuda/README.md`.

### Actual CUDA kernel locations

The CUDA implementations are under experiment trees:

- `experiments/Exp01-vector-add/src/vector_add.cu`
- `experiments/Exp02-reduction/src/reduction.cu`
- `experiments/Exp02-reduction/src/reduction_kernels.cuh`
- `experiments/Exp03-matrix-transpose/src/transpose.cu`
- `experiments/Exp03-matrix-transpose/src/transpose_kernels.cuh`
- `experiments/Exp04-gemm/src/gemm.cu`
- `experiments/Phase5-cuda-feasibility/src/phase5a_cublaslt_benchmark.cu`
- `experiments/Phase5-cuda-feasibility/src/phase5a_cutlass_benchmark.cu`
- `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/`

### `tensorrt/`

**Purpose:** Learning-scope index for TensorRT, plugins, and TensorRT-LLM.

**Core file:** `tensorrt/README.md`.

### Actual TensorRT and TensorRT Plugin locations

- `experiments/Phase2-qwen3-quantization/src/trt_precision_probe/`
- `experiments/Phase2-qwen3-quantization/src/phase2_1_5_graph_pipeline/`
- `experiments/Phase2-qwen3-quantization/src/phase2_1_8_qwen3_block/`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_2B1/`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_2B2/`
- `experiments/Phase9-qwen3-vl-migration/src/phase9_2C1/`
- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/`

The dedicated plugin implementation is especially:

- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.h`
- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin.cpp`
- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin_kernel.cu`
- `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/rmsnorm_plugin_creator.cpp`

### `runtime/`

**Purpose:** Learning-scope index for KV cache, CUDA Graph, memory management,
and LLM runtime work.

**Core file:** `runtime/README.md`.

Actual runtime material is distributed across:

- `experiments/Phase2-qwen3-quantization/src/phase3a/`
- `experiments/Phase2-qwen3-quantization/src/phase3b/`
- `experiments/Phase2-qwen3-quantization/src/phase3c/`
- `experiments/Phase2-qwen3-quantization/src/phase3d0/`
- `docs/phase2_2c_runtime_architecture.md`
- `docs/phase3a_runtime_bottleneck_attribution.md`
- `docs/phase3b_runtime_object_lifetime.md`

### `transformer/`

**Purpose:** Learning-scope index for Transformer operators, attention, and
quantization concepts.

**Core file:** `transformer/README.md`.

Actual Transformer/model work lives in:

- `experiments/Phase1-qwen3-baseline/`
- `experiments/Phase2-qwen3-quantization/`
- `experiments/Phase9-qwen3-vl-migration/`

### `benchmark/`

**Purpose:** Scope-setting benchmark standards. The actual measurements are in
each experiment directory and compact result directories.

**Core file:** `benchmark/README.md`.

Representative benchmark locations:

- `experiments/Exp01-vector-add/benchmark/`
- `experiments/Exp02-reduction/benchmark/`
- `experiments/Exp03-matrix-transpose/benchmark/`
- `experiments/Exp04-gemm/benchmark/`
- `experiments/Phase1-qwen3-baseline/artifacts/`
- `experiments/Phase9-qwen3-vl-migration/artifacts/`

### `scripts/`

**Purpose:** Scope-setting index for reusable, reviewable environment/build/run
scripts. Most experiment-specific scripts live beside their experiment code.

**Core file:** `scripts/README.md`.

Representative script locations:

- `experiments/Exp01-vector-add/scripts/`
- `experiments/Exp02-reduction/scripts/`
- `experiments/Exp03-matrix-transpose/scripts/`
- `experiments/Phase8-rmsnorm-optimization/scripts/`
- `experiments/Phase9-qwen3-vl-migration/src/`

### `references/`

**Purpose:** Learning-material index only. Large PDFs and source archives are
not committed.

**Core file:** `references/README.md`.

## 4. Learning Path

The recommended order follows the historical evidence chain. Read reports before
deeper code so that each implementation is understood with its measurement
boundary and limitations.

### Step 1: Recover Project Context

1. Read `README.md`.
2. Read `docs/PROJECT_FINAL_REPORT.md`.
3. Read `docs/FINAL_STATUS.md`.
4. Read `docs/PROJECT_STATE.md` and `docs/experiment_index.md`.
5. Skim `results/experiment_registry.csv` for experiment order and gates.

### Step 2: Learn CUDA Fundamentals

1. Read `experiments/Exp01-vector-add/README.md`, then inspect
   `experiments/Exp01-vector-add/src/vector_add.cu`.
2. Read `experiments/Exp02-reduction/README.md`, then inspect
   `experiments/Exp02-reduction/src/reduction.cu`.
3. Read `experiments/Exp03-matrix-transpose/README.md`, then inspect
   `experiments/Exp03-matrix-transpose/src/transpose.cu`.
4. Read `experiments/Exp04-gemm/README.md`, then inspect
   `experiments/Exp04-gemm/src/gemm.cu`.
5. Compare each experiment's correctness, benchmark, and NCU evidence.

### Step 3: Establish The Model Baseline

1. Read `experiments/Phase1-qwen3-baseline/README.md`.
2. Read `experiments/Phase1-qwen3-baseline/docs/phase1_0_runtime_feasibility_audit.md`.
3. Read `experiments/Phase1-qwen3-baseline/docs/phase1_2_formal_benchmark.md`.
4. Note the pinned model revision, attention path, workload, and baseline
   latency/throughput boundaries.

### Step 4: Analyze Quantization And TensorRT Runtime Construction

1. Read `experiments/Phase2-qwen3-quantization/README.md`.
2. Start with quantization feasibility reports under
   `experiments/Phase2-qwen3-quantization/docs/`.
3. Inspect synthetic TensorRT graph code under
   `experiments/Phase2-qwen3-quantization/src/phase2_1_5_graph_pipeline/`.
4. Inspect real-component runtime work under
   `experiments/Phase2-qwen3-quantization/src/`.
5. Preserve the distinction between component-level bounded PASS and unresolved
   full-decoder numerical limitations.

### Step 5: Analyze Runtime And Operator Attribution

1. Read `docs/phase2_2c_runtime_architecture.md`.
2. Read Phase 3 reports under `docs/` and compact evidence under `results/`.
3. Inspect runtime/profile code under
   `experiments/Phase2-qwen3-quantization/src/phase3a/` through
   `phase3d0/`.
4. Read Phase 4 operator attribution reports and inspect
   `results/phase4a_operator_attribution/`.
5. Treat profiler names as attribution evidence, not proof of a replacement
   target.

### Step 6: Analyze GEMM And Attention Target Reassessment

1. Read Phase 5 evidence under
   `results/phase5a_cuda_feasibility_baseline/` and
   `results/phase5b_tensorrt_gemm_path_investigation/`.
2. Inspect `experiments/Phase5-cuda-feasibility/src/`.
3. Read Phase 6 attention attribution reports under `results/`.
4. Inspect `experiments/Phase6-attention-matmul-attribution/src/`.
5. End with the Phase 7 reassessment evidence under
   `results/phase7_global_optimization_target_reassessment/`.

### Step 7: Analyze CUDA Kernel To TensorRT Plugin Flow

1. Read
   `experiments/Phase8-rmsnorm-optimization/docs/phase8_0_baseline_audit_report.md`.
2. Inspect V0/V1/V2 kernels in
   `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/`.
3. Compare the Phase 8.2 NCU report and compact summaries.
4. Read `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/README.md`.
5. Inspect the plugin source and Qwen3 integration code.
6. Keep the final Phase 8.4 conclusion separate: isolated gains did not become
   an end-to-end speedup.

### Step 8: Analyze The Final Qwen3-VL Migration

1. Read `experiments/Phase9-qwen3-vl-migration/docs/phase9_0B_checkpoint_preparation_audit.md`.
2. Read the FP16 smoke test and baseline benchmark reports.
3. Follow the Vision Encoder path: readiness audit, ONNX export, TensorRT parse,
   FP16 engine build, correctness, and latency.
4. Inspect the corresponding source files under
   `experiments/Phase9-qwen3-vl-migration/src/`.
5. Read Phase 9.3-A/B1/B2 reports for integration, stage attribution, and
   decoder bottleneck attribution.
6. Finish with `docs/PROJECT_FINAL_REPORT.md` and `docs/FINAL_STATUS.md`.

## 5. Important Modules

### CUDA Kernel

- Scope: vector add, reduction, transpose, GEMM/WMMA, RMSNorm V0/V1/V2.
- Code: `experiments/Exp01-vector-add/src/`,
  `experiments/Exp02-reduction/src/`,
  `experiments/Exp03-matrix-transpose/src/`,
  `experiments/Exp04-gemm/src/`, and
  `experiments/Phase8-rmsnorm-optimization/cuda-kernel/rmsnorm/`.
- Evidence: correctness CSV/TXT, CUDA-event benchmarks, NCU summaries.

### TensorRT Plugin

- Scope: RMSNorm `IPluginV3`, synthetic demo, and single-node Qwen3 integration.
- Code: `experiments/Phase8-rmsnorm-optimization/tensorrt-plugin/`.
- Reports:
  `experiments/Phase8-rmsnorm-optimization/docs/phase8_3A_tensorrt_plugin_design_report.md`
  and
  `experiments/Phase8-rmsnorm-optimization/docs/phase8_3B_qwen3_single_rmsnorm_integration_report.md`.

### Transformer / Qwen Model Path

- Scope: Qwen3 baseline, decoder construction, Qwen3-VL checkpoint and vision
  encoder migration.
- Code: `experiments/Phase1-qwen3-baseline/`,
  `experiments/Phase2-qwen3-quantization/`, and
  `experiments/Phase9-qwen3-vl-migration/`.
- Evidence: pinned model manifests, smoke tests, benchmark JSON, and compact
  correctness artifacts.

### Quantization

- Scope: backend feasibility, TensorRT INT8/FP16 capability, explicit QDQ,
  sensitivity, calibration, and mixed precision policy.
- Code: `experiments/Phase2-qwen3-quantization/src/phase2_3a/` through
  `phase2_3f/`.
- Evidence: reports/artifacts under
  `experiments/Phase2-qwen3-quantization/`.

### Runtime

- Scope: TensorRT runtime ownership, KV cache, persistent context, CUDA Graph,
  runtime object lifetime, and Qwen3-VL visual adapter.
- Code: `experiments/Phase2-qwen3-quantization/src/phase3*/` and
  `experiments/Phase9-qwen3-vl-migration/src/phase9_3*/`.
- Evidence: `results/phase3*/`, Phase 9.3 reports, and final report.

### Benchmark

- Scope: formal CUDA-event benchmarking, NCU/NSYS profiling, tegrastats, and
  fixed-workload end-to-end attribution.
- Code: scripts under each experiment directory.
- Evidence: `experiments/*/benchmark/`, `experiments/*/artifacts/`, and
  `results/`.

## Reading Warnings

- Do not treat isolated component speedups as end-to-end speedups.
- Do not treat profiler kernel names as direct DRAM, occupancy, or power proof.
- Do not infer production readiness from `PROJECT_COMPLETE`.
- Keep `UNKNOWN` and `INCONCLUSIVE` boundaries intact when summarizing results.
