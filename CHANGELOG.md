# Changelog

本文件记录项目管理、实验和基础设施的重要变化。

## [Unreleased]

### Phase 5-B Step 1 TensorRT GEMM Path Investigation (2026-09-05)

- Recovered bounded tactic attribution for all 28 `up_proj` GEMM layers from
  frozen Engine Inspector, ONNX mapping, and historical NSYS kernel artifacts.
  No engine was deserialized, executed, rebuilt, or modified.
- Recorded that 25 layers use an `f16f16_f16` xmma tactic label while decoder
  layers `8`, `14`, and `27` use an `f16f32_f32` label; all 28 labels contain
  `tensor16x8x16`. Label semantics are not promoted to proven accumulator or
  layout behavior.
- Recorded `NOT_AVAILABLE` for numeric tactic ID, runtime workspace, and a
  dedicated kernel-name field. Backend identity, accumulator dtype, physical
  layout semantics, and complete tactic-to-kernel mapping remain `UNKNOWN`.
- Kept the direct cuBLASLt algorithm 21 result as comparison context only.
  Phase 5-B Step 1 is `CASE_B_SUPPORTED_BOUNDED`, with no proven tactic defect
  and no implementation authorized.

### Phase 5-A TensorRT Boundary Reconciliation (2026-09-05)

- Re-read the existing Phase 4-F Mixed persistent NSYS SQLite trace in read-only
  mode and resolved 196 `up_proj` NVTX range instances, each with one correlated
  CUDA kernel and no attributed non-GEMM kernel.
- Recorded `h16816` at 84 launches / median `253.680 us` and `sm80_xmma_gemm`
  at 112 launches / median `118.800 us`; the steady-state correlated kernel
  median is `147.424 us`.
- Explained that negative `NVTX-kernel` residuals reflect asynchronous kernels
  extending beyond host NVTX boundaries; launch API duration alone cannot
  explain the gap versus direct cuBLASLt.
- Updated Phase 5-A Gate to `CASE_2_SUPPORTED_BOUNDED`. No tactic defect is
  proven, TensorRT backend identity remains UNKNOWN, and no CUDA kernel,
  Plugin, engine change, tactic forcing, or runtime redesign is authorized.

### Phase 5-A Benchmark Execution (2026-09-05)

- Completed direct cuBLASLt and CUTLASS `v3.5.1` library-generated benchmarks
  for the frozen decode-only `up_proj` FP16 GEMM shape.
- Primary cuBLASLt FP32 accumulate: median `0.080077961 ms`, CV `0.0073666%`,
  correctness PASS. Best CUTLASS candidate: median `0.083619133 ms`,
  CV `0.0104699%`, correctness PASS.
- Excluded all failing FP16-accumulate cuBLASLt records from performance
  ranking and preserved their raw evidence.
- Reused Phase 4-E TensorRT evidence without rerun and kept its `IProfiler`
  boundary caveat; TensorRT backend identity remains UNKNOWN.
- Phase 5-A Gate: `INCONCLUSIVE / BOUNDED / NO_PROVEN_CUTLASS_OPTIMIZATION`.
  No CUDA kernel implementation or Phase 5-B work is authorized.

### Phase 5-A Design Checkpoint (2026-09-05)

- Opened `phase/05a-cuda-feasibility-baseline-study` from the Phase 4 evidence
  checkpoint and completed a read-only Jetson environment freeze.
- Recorded CUDA 12.6.68, cuBLASLt runtime 120601, PyTorch 2.5.0a0, TensorRT
  10.3.0, NCU 2024.3.1.0, NSYS 2024.5.4.34, SM 8.7, and no standard CUTLASS
  installation.
- Froze the Phase 5-A benchmark design without implementing a kernel: keep
  Phase 4-E as TensorRT Baseline 1, pre-register direct cuBLASLt Baseline 2,
  and pre-register a CUTLASS library-generated Baseline 3.
- Pre-registered correctness, CV, bootstrap CI, and Case A/B/C decision
  thresholds. Phase 5-A remains `IN_PROGRESS / DESIGN_ONLY` pending review
  before benchmark execution.

### Phase 4 Evidence Checkpoint (2026-09-05)

- Added compact Phase 4-A through 4-G raw reports and artifacts under
  `results/phase4a_operator_attribution/` through
  `results/phase4g_optimization_hypothesis/`.
- Recorded recovery of 250 GEMM candidates and selection of decode-only
  `up_proj` FP16 GEMM as the first optimization-research target.
- Preserved existing standalone, TensorRT `IProfiler`, and NSYS kernel
  attribution baselines without changing their measurement boundaries or
  conclusions.
- Phase 4-G remains `PASS / BOUNDED / NO_PROVEN_OPTIMIZATION_HYPOTHESIS` with
  CUDA readiness `NOT READY`; Phase 5-A is limited to environment freeze and
  benchmark harness design until its protocol is reviewed.

### Phase 3-E (2026-09-04)

- Added TensorRT EngineInspector attribution for the frozen FP16 and Mixed
  decoder engines and joined it with Phase 3-C NSYS kernel totals.
- Selected three unique GEMM kernels for targeted NCU with narrow kernel
  filters and `--clock-control none`; raw `.ncu-rep` files remain Jetson-local.
- Rank-1 h16816 was L2/memory-limited at 97.06% with 39.673148% HMMA pipe
  active and near-ceiling achieved occupancy. Rank-2/3 small decode GEMMs had
  17.167166-17.697509% HMMA active, but compact FP16 metadata left operator
  attribution UNKNOWN.
- Final gate is `PASS / BOUNDED / NO_PROVEN_CUDA_OPTIMIZATION_TARGET`; Phase 4
  implementation was not started.

### Phase 3-D0 (2026-09-04)

- Added a CUDA Graph compatibility audit, full-window capture and a per-engine
  graph mode while preserving the persistent stream reference unchanged.
- Both capture modes completed topology/address checks but failed hidden, logits
  and KV validation, so the formal benchmark gate is
  `BLOCKED_NO_VALID_GRAPH_PATH`.
- Separate 8-step NSYS traces showed persistent stream with 3,376 kernels and
  217.708 ms kernel time versus graph replay with 64 `FillFunctor<long>` kernels
  and 0.298 ms kernel time and no TensorRT kernels.
- Final gate is `BLOCKED / BOUNDED`. The current PyTorch/TensorRT interop path is
  `DISPROVEN`; Phase 3-D1 was not started.

### Phase 3-C (2026-09-04)

- Added persistent FP16/Mixed residual profiling and a Phase 3-C-only NSYS
  analyzer. Legacy runtime, CUDA kernels, plugins and quantization redesign
  were not revisited.
- C1 persistent benchmark showed FP16 and Mixed near parity at S=8/S=16, with
  5 context creations and 448 reuses for each runtime and no OOM/exit137.
- C2 steady-state NSYS showed module load/unload remained `0/0` for both
  persistent runtimes. Aggregate kernel time was 219.710 ms FP16 and
  147.831 ms Mixed; CUDA API time was 186.374 ms FP16 and 116.749 ms Mixed.
- C3 name-based attribution found GEMM dominant, but the largest single kernel
  was only 35.31% (FP16) / 26.78% (Mixed) of kernel time. TensorRT internal
  operator identity remained `UNKNOWN`. NCU was not required.
- Final decision gate is `Case C`: no justified Phase 3-D CUDA operator target
  was identified. Gate: `PASS / BOUNDED`; Phase 3-D was not started.

### Phase 3-B (2026-09-04)

- Added an explicit `legacy_context_lifetime` /
  `persistent_context_lifetime` A/B. The audit proved the original hot path
  created a TensorRT execution context per `TRT.run`, while persistent mode
  creates one context per engine wrapper and continues setting shapes and
  tensor addresses on every call.
- Mixed legacy-vs-persistent hidden, logits and K/V outputs were exactly equal
  and finite at S=8 plus 8 decode steps and S=16 prefill; KV prefix, length and
  28-layer pointer invariants passed with no OOM/exit137.
- The same-session four-way benchmark reduced Mixed prefill from
  2259.919/2255.619 ms to 47.627/42.406 ms at S=8/S=16 and Mixed TPOT from
  2662.990/2662.266 ms to 43.362/43.580 ms. Gap recovery was 99.7-100.3%.
- Steady-state NSYS collapsed `cuModuleLoadData`/`cuModuleUnload` counts and
  time to zero; Mixed wall fell from 2310.857 to 33.217 ms at prefill and
  2881.660 to 65.848 ms at decode step 0. H3B-1 is `PROVEN`.
- Initialization increased by 3.430/4.019 s for FP16/Mixed. Observational
  loaded memory decreased, but exact retained-context cost remains
  `INCONCLUSIVE`. Gate: `PASS / CLOSED / PROVEN`; Phase 3-C remains not started.

### Phase 3-A (2026-09-04)

- Added controlled Phase 3-A benchmark and NSYS attribution. Mixed remained
  49.0-49.1% slower at prefill and 35.7-36.0% slower at decode than FP16.
- Measured GPU busy at 1.46-3.00% of profiled wall; Mixed GPU kernel time was
  lower than FP16, while its wall delta was dominated by host CUDA API time.
  `cuModuleLoadData`/`cuModuleUnload` accounted for +565.3/+693.8 ms of the
  prefill/decode API delta.
- Classified per-call execution-context creation/module-load storm as
  `SUPPORTED` and direct Q/DQ GPU overhead as `DISPROVEN`; Myelin operator
  mapping remains `UNKNOWN`. NCU was `NOT REQUIRED`.
- Gate: `PASS / BOUNDED`. Recommended Phase 3-B target is persistent TensorRT
  execution contexts; Phase 3-B remains unstarted.

### Phase 2.3-F (2026-09-04)

- Compared the Phase 2.3-E mixed runtime against the current TRT FP16 runtime
  on the 12 disjoint Phase 2.3-B evaluation prompts plus a same-session
  benchmark (warmup 5, repeats 10, batch 1, S=8/16).
- Mixed-vs-FP16 prefill last-token logits have relative-L2 median `0.3311` and
  top-1 agreement `4/12`; forced-decode cosine median is `0.8879`. Free-run
  trajectories diverge at step 0 for a majority of prompts.
- The mixed runtime is ~48% slower at prefill and ~35% slower at decode
  (`MIXED_RUNTIME_SLOWER`), while engine storage is 27% smaller. No OOM/exit137.
- Phase 2.3 closeout: `CLOSED / PASS / BOUNDED`. C1 remains
  `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`; Phase 3 / INT4 / Nsight remain
  not started.

### Phase 2.3-E (2026-09-04)

- Applied the frozen Phase 2.3-D `P2_FAMILY_GUARD_REFINED` policy to the
  existing B4.2 28-layer FP16 ONNX by injecting explicit weight/activation
  Q/DQ for 133 PT-W8A8 Linear MatMuls and preserving 63 FP16 targets.
- Built mixed prefill/decode engines; EngineInspector confirms 133 INT8
  tensor-core GEMM tactics per engine (`INT8_COMPUTE_PROVEN` for the
  deployable PT-W8A8 path). Full runtime passed prefill, decode, KV cache,
  Final RMSNorm, LM Head and `Hello` generation with no OOM/exit137.
- Gate `PASS / BOUNDED` (`PRIMARY_POLICY_RUNTIME`); mixed-vs-FP16 forced-decode
  hidden/logits relative-L2 are bounded and mixed engines are ~27% smaller.
  No fallback policy, benchmark, Nsight, INT4, or C1 re-opening occurred.

### Phase 2.3-D (2026-09-04)

- Derived P0/P1/P2 mixed-precision policies from the Phase 2.3-C 196-Linear evidence set. The selected P2 family guard preserved all `up_proj`/`down_proj` layers and the C gate outlier as FP16.
- Used real Qwen3 BF16 forward-pre-hooks in a two-pass streaming calibration and a disjoint 12-prompt component prevalidation. No raw activations were retained or committed.
- One bounded robust refinement moved six additional targets to FP16. Final coverage is 63 FP16 and 133 PT-W8A8 Linear assignments; no TensorRT quantized decoder, benchmark, Nsight, INT4, or C1 work occurred.

### Phase 2.3-C (2026-09-03)

- Audited 196 decoder Linear weights and a 34-target portable sensitivity matrix.
- Confirmed eight deterministic TensorRT targets; per-channel QDQ parsing is
  explicitly blocked while FP16/PT-W8/PT-W8A8 paths remain finite.
- Gate `PASS / BOUNDED`; no benchmark, Nsight, INT4 or full quantized runtime.

### Phase 2.3-A (2026-09-03)

- Validated an explicit TensorRT 10.3 INT8 Q/DQ path on the real Qwen3 Layer 0 `self_attn.q_proj` component using one shared canonical activation input.
- FP16, W8-QDQ and W8A8-QDQ graphs passed parser/build/execution with finite outputs. Symmetric zero-point `0` is required; non-zero zero-points are rejected by the parser.
- W8-QDQ remains `INT8_COMPUTE_NOT_PROVEN`; W8A8 EngineInspector exposes an Int8/Int8 fusion with an `i8` GEMM tactic, classified `INT8_COMPUTE_PROVEN` for this graph/profile. Phase 2.3-A Gate is `PASS`; calibration, benchmark, Nsight and full-runtime quantization remain unstarted.

### Phase 2.3-B (2026-09-03)

- Proved the exact real Layer 0 `q_proj` input as post-`input_layernorm` and pre-`q_proj` using a forward-pre-hook across 24 calibration and 12 disjoint evaluation prompts.
- Evaluated calibration-only GLOBAL_ABSMAX, P99.9, P99.99 and bounded MSE clipping candidates. Selected `BOUNDED_MSE_CLIP` at scale `0.0243602362`; held-out activation-only W8A8-vs-W8 relative-L2 was `0.019526` median, `0.020111` P95 and `0.020117` maximum.
- Selected-policy TensorRT detailed EngineInspector retained Int8 activation/weight inputs and an explicit Int8 GEMM tactic. Phase 2.3-B Gate is `PASS`; Phase 2.3-C, full-model calibration, benchmark and Nsight remain unstarted.

### Phase 2.2-C5 (2026-09-03)

- Completed the minimal real Qwen3 autoregressive runtime for prompt `Hello`: pinned tokenizer, TensorRT embedding, 28-layer prefill/decode, KV cache, Final RMSNorm, LM Head and CPU greedy sampling.
- HF reference IDs were `[21806, 0, 358, 2776]`; TensorRT IDs were `[0, 46309, 46309, 46309]`, diverging at step 0 under the documented C1 decoder numerical limitation. All decode outputs remained valid and finite.
- KV prefix invariants, cache growth `1->4`, 28-way pointer isolation and tokenizer decode passed. Phase 2.2 C0-C5 closeout is `PASS / BOUNDED`; Phase 2.3 was not started.

### Phase 2.2-C4 (2026-09-03)

- Added deterministic CPU/NumPy greedy sampling with explicit first-index-wins tie semantics for `[B,1,151936]` logits.
- Synthetic clear-winner, near-tie, exact-tie, C3 decode and last-token logits all achieved exact integer token agreement. The read-only embedding -> decoder -> Final RMSNorm -> LM Head single-step produced token `42` on both portable and TensorRT paths.
- C4 result: `PASS / BOUNDED`. Full path remains `END_TO_END_DIAGNOSTIC_ONLY` due to the known C1 decoder drift; C5 generation loop was not started.

### Phase 2.2-C3 (2026-09-03)

- Audited the pinned Qwen3 `lm_head.weight`: `[151936,1024]` BF16, bias-free, and byte-identical to `model.embed_tokens.weight` under tied embeddings.
- Built an independent TensorRT 10.3 FP16 MatMul engine. Synthetic prefill, decode, and last-token paths passed finite/shape, argmax, and top-5 checks against the portable oracle.
- Read-only embedding -> corrected 28-layer decoder -> C2 Final RMSNorm -> LM Head integration completed with finite CUDA logits. Full-path numerical comparisons remain `END_TO_END_DIAGNOSTIC_ONLY` due to the closed C1 decoder drift. C3 result: `PASS / BOUNDED`; C4 was not started.

### Phase 2.2-C2 (2026-09-03)

- Integrated and validated the pinned Qwen3 final RMSNorm as an independent TensorRT FP16 component. `model.norm.weight` was confirmed at shape `[1024]`, BF16, epsilon `1e-6`, with checkpoint and weight hashes recorded.
- Built independent native and explicit FP32-reduction TensorRT engines. Synthetic same-input operator metrics passed for prefill/decode; the FP32-reduction semantic path was selected. Existing C1 embedding and corrected 28-layer decoder assets remained read-only.
- Decoder-to-final-norm prefill `[1,8,1024]` and decode `[1,1,1024]` CUDA integration passed with no OOM or exit 137. Full-path comparisons remain `END_TO_END_DIAGNOSTIC_ONLY` because of the closed C1 decoder drift. C2 result: `PASS / BOUNDED`; C3 was not started.

### Added

- 初始化项目结构、实验规范、Git 工作流与交接文档。
- 完成 Exp00 Windows/Jetson 仓库 bootstrap、SSH 验证与离线 Git 历史同步。
- 创建 Public GitHub 仓库并完成 Windows、GitHub、Jetson 三端 Git 同步。
- 完成 Exp01 CUDA Vector Add correctness 与 block-size benchmark；Nsight Gate 因 profiling 权限不足记录为 BLOCKED。
- 完成 Exp01.1 五轮稳定性与 Nsight 权限审计：block 128 为 5/5 observed fastest，csvEscape 修复验证通过，Nsight 仍因非交互 sudo 认证受阻。
- 建立无状态、仓库驱动的 Codex 工作流，新增项目状态入口并扩展实验索引与证据规则。
- 关闭 Exp01 Nsight Compute Gate：完成 blocks 32/128/256/1024 profiling，H1–H4 最终 SUPPORTED，128 vs 256 微架构成因保持 INCONCLUSIVE，Overall Gate PASS。
- 初始化 Exp02 CUDA Reduction V1-V7，并完成 correctness Gate A：3,087/3,087 执行通过；compute-sanitizer 在 Jetson 上不可用，记录为 N/A；Gate B/C 未开始。
- 完成 Exp02.2 Benchmark/Stability Gate：B1/B2/B3 全部完成，所有 timed configurations correctness PASS，V5/B128 为 N=16,777,229 下最快稳定配置；Gate B PASS，Gate C 待 checkpoint 后开始。
- 完成 Exp02.3 Nsight Gate 与最终收口：V1-V7 common-block profile、V5 block sweep 和 H1-H9 分析完成；Gate A/B/C 均 PASS，Overall PASS，READY_FOR_EXP03；Exp03 未开始。
- 初始化 Exp03 CUDA Matrix Transpose（V0-V4）并完成 Correctness Gate：828/828 bitwise exact PASS，修复 V1/V2 非 tiled grid_y 边界 bug；Gate A PASS，Gate B/C 未开始，READY_FOR_EXP03_BENCHMARK。

- 完成 Exp03.2 两轮正式 Benchmark；correctness/timing PASS，但多项配置 CV 持续超过 5%，Gate B INCONCLUSIVE，Gate C 未开始。
- 完成 Exp03.2b stability diagnosis 与 Formal Benchmark V2：time-based warmup、adaptive CUDA Event window、7 trials 和只读 telemetry 使全部配置 CV <= 0.791%；Gate B PASS，Gate C 待执行。
- 完成 Exp03.3 Nsight Compute 与收口：V1-V4 4096x4096 profile 取得 global sector、shared bank-conflict、occupancy、warp-state 证据；H1-H4 SUPPORTED，Gate A/B/C PASS，Exp03 CLOSED，READY_FOR_EXP04。
- 完成 Phase 1.0 Qwen3-0.6B runtime feasibility audit：确认当前 JetPack 6.2/CUDA 12.6/TensorRT 10.3/PyTorch 环境，识别最新 TensorRT-LLM 的 SM87/软件栈断层与 `v0.12.0-jetson` 的 Qwen3 支持缺口；Gate P1.0 `PASS WITH CONSTRAINTS`，推荐 Phase 1.1 先建立 HF PyTorch BF16 reference baseline，未安装或运行任何 runtime/model。
- 完成 Phase 1.1 Qwen3-0.6B exact-revision Hugging Face BF16 reference：Gate A/B/C/D PASS；模型全参数 `cuda:0`、forward finite、3/3 bounded generations PASS，并记录 unified-memory/allocator/tegrastats 证据；Phase 1.2 未开始。
- 完成 Phase 1.2 Qwen3-0.6B BF16/eager formal benchmark：manual KV-cache loop 与 `generate()` 8/8 token 一致；ISL 32/128/512/1024 各 10 trials，全部 TTFT/TPOT CV <5%；Gate A/B/C/D PASS，Phase 1.2 CLOSED。
- Phase 1 overall formally closed as `PASS / CLOSED`; TensorRT-LLM backport is not a prerequisite for closure and remains a later runtime investigation.
- Completed Phase 2.1 TensorRT capability audit: direct synthetic FP16 and explicit Q/DQ INT8 Linear execution passed on Orin SM87; ONNX parser route blocked by missing frozen-venv package; INT4 weight-only construction path not identified. No Qwen3 quantization, formal performance claim, TensorRT-LLM or Phase 2.2 work started.
- Completed Phase 2.1.5 synthetic TensorRT graph pipeline enablement: isolated ONNX toolchain, three FP16 opset-17 exports/checks, and six dynamic-profile TensorRT parser/build/CUDA execution cases passed. No Qwen3 export, quantization, benchmark, or Phase 2.2 work started.
- Completed bounded Phase 2.1.8 Qwen3-like decoder-block feasibility audit: synthetic FP16 single-block ONNX export/check, TensorRT 10.3 parse/build, and dynamic batch 1/2 CUDA execution passed. FP16 normalization/default-stream warnings are retained; no full Qwen3 export, quantization, benchmark or Phase 2.2/3 work started.
- Completed bounded Phase 2.1.9 full Qwen3 TensorRT architecture audit: mapped frozen config, layer/weight layout, theoretical weight and KV-cache memory, dynamic prefill/decode contract, operator concerns and native TensorRT versus TensorRT-LLM routes. Gate D is `BLOCKED_NEEDS_RUNTIME_WORK`; no checkpoint load, full export, engine, benchmark or quantization was performed.
- Prepared Phase 2.2 Qwen3 TensorRT FP16 runtime prototype design: CPU-only KV-cache manager, prefill/decode request interfaces, theoretical batch-1/2 cache estimates, attention notes and future Gate A-D plan. Synthetic cache contract validation passed; no model, CUDA/TensorRT execution, engine, benchmark or quantization was run.
- Completed bounded Phase 2.2-A single-layer TensorRT FP16 KV-cache runtime integration: synthetic prefill/decode parser/build and CUDA execution passed, dynamic cache growth 8->12 and CUDA-resident direct bindings were validated. Numerical results are informational only; default-stream and FP16 normalization warnings remain. No full Qwen3 export, benchmark, profiler or quantization was run.
- Completed Phase 2.2-B1 bounded four-layer TensorRT runtime orchestration: ordered hidden-state handoff, independent per-layer KV caches, dynamic decode 8->12, CUDA residency and stream ownership all passed. No 28-layer deployment, benchmark, Nsight, quantization or full Qwen3 export was run.
- Audited the Phase 2.2-B2 environment bridge: phase1-hf has frozen Transformers/Safetensors while phase2-trt-tools has ONNX/TensorRT; pip lacks --dry-run, so no installation was attempted. Selected `SOLUTION_C_DUAL_ENV_REQUIRED`; B2 scientific execution remains not started.
- Completed bounded Phase 2.2-B2 real Qwen3-0.6B Layer 0 integration via dual environments: frozen identity/hash and 11-tensor mapping passed; HF BF16, portable BF16 cross-env reproduction, FP16 ONNX/TensorRT prefill and dynamic decode 8->12 passed with zero cache-prefix mutation. No full model or 28-layer runtime was built.
- Completed bounded Phase 2.2-B3 real Qwen3 layers 0-3 TensorRT FP16 stack audit: one prefill graph and one dynamic decode graph passed with per-layer KV isolation, 8->12 cache growth, and numerical/attention propagation evidence. No layers 4-27, full model, benchmark, profiler or quantization work was performed.
- Phase 2.2-B4 28-layer feasibility is blocked by Jetson oracle resource pressure: both the monolithic and predefined 7x4 HF attempts were killed with exit 137 before artifact completion. No TensorRT export/build/runtime or system modification was attempted.
- Recovered the Phase 2.2-B4 oracle prerequisite through B4.1: the exact Phase 1 model probe passed, a 28-file streaming handoff avoided all-layer state retention, and fresh-process 4/8/28-layer diagnostics passed through one 28-layer decode `8->9`. Root cause is `IMPLEMENTATION_MEMORY_LIFETIME_CONFIRMED`; no ONNX/TensorRT, benchmark, profiler or quantization work was run.
- Completed Phase 2.2-C1E layerwise decoder divergence localization: reused the existing 28-layer engine and canonical embedding, found first non-zero divergence at Layer 0, relative-L2 alert at Layer 21 and cosine alert at Layer 22; C1 remains `BLOCKED` and no engine rebuild or C2 work was performed.
- Completed Phase 2.2-C1 closeout repair: built new Jetson-local 28-layer prefill/decode engines with the C1K FP16 RoPE cache correction. Embedding boundary and decode 8->12 KV invariants passed, but Layer 27 relative-L2 remained `2.0046324720`; gate is `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`. No historical B4.2 artifact was modified and C2 was not started.
- Completed Phase 2.2-C1F Layer 0 component-attribution audit: confirmed Qwen3-specific RMSNorm/QK-norm/RoPE/GQA/SwiGLU structure, but the existing B4.2 artifact exposes no internal component bindings. Result is `COMPONENT_LOCALIZATION_BLOCKED`, first operator `UNKNOWN`; existing B3 `attention_l0` is retained as partial evidence only. No engine rebuild, repair or C2 work was performed.
- Completed Phase 2.2-C1G Layer 0 internal tensor probe: three independent Group A/B/C diagnostic engines passed probe-validity checks against B4.2. QK score relative-L2 `1.95e-7` rose to softmax `0.003432`, narrowing the first material amplification to softmax; root cause remains not confirmed. No precision A/B, repair or C2 work was performed.
- Completed Phase 2.2-C1H attention softmax semantics diagnostic: exact masked pre-softmax inputs were nearly identical, same-input native and explicit FP32 TensorRT softmax matched the portable result, and Layer 0 FP32-softmax-only A/B did not improve final divergence. Result `SOFTMAX_HYPOTHESIS_REJECTED`; C1 remains blocked and no repair/C1I/C2 work was performed.
- Completed Phase 2.2-C1I Q/K and RoPE numerical isolation: Q/K projection and norm stayed below 0.002 relative-L2, while same-input Q/K RoPE reached 0.092710741/0.014421386 and QK raw 0.043750536. Result `ROPE_MAJOR_SOURCE_CONFIRMED`; C1 remains blocked and no C1J/C2 or repair was started.
- Completed Phase 2.2-C1J Qwen3 Layer 0 RoPE numerical root-cause diagnostic: native TensorRT cache differs from portable FP16 cache, while FP32-cache semantics match exactly; positions/layout/GQA are confirmed and even/odd is rejected. Result `ROPE_PRECISION_SEMANTICS_CONFIRMED`; C1 remains blocked and no repair, C1K or C2 was started.
- Completed Phase 2.2-C1K RoPE cache precision validation: explicit portable FP16 `cos/sin` initializers reduced Q/K RoPE relative-L2 to `0.0001572046/0.0000268356` and Layer 0 final relative-L2 from `0.0347152092` to `0.0038153231` (`9.0989x`). Result `ROPE_CACHE_FIX_VALIDATED`; C1 remains blocked and no repair/C1L/C2/benchmark/Nsight/quantization/rebuild was started.
