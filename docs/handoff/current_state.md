# Current State

本文件是辅助交接。独立恢复工作应以 `AGENTS.md`、`docs/PROJECT_STATE.md`、`docs/experiment_index.md`、实验报告和 raw artifacts 为主。

## 当前 branch

`phase/02-qwen3-quantization`

## 当前 commit

本轮 starting HEAD 为 `e9fa64fba35b570f67b91e9a686fd91f6190308a`；Phase 2.2-A closeout commit 以 `git rev-parse HEAD` 为准。

## 本轮完成

- 重新核验 Windows、GitHub、Jetson 均从 clean 的 `exp/01-vector-add@e10f2c0` 开始。
- `ncu --version`、`sudo -n ncu --version`、`sudo -n ncu --list-sections` 全部 PASS；Nsight Compute `2024.3.1.0`。
- 新增最小 runner，沿用原 `N=16,777,216`、FP32 kernel、warmup 20、repetitions 200，只 profile blocks 32/128/256/1024 的第一个 measured launch。
- 四个 profile 均成功、correctness PASS、最大误差 0；`.ncu-rep` 仅保留在 Jetson `/tmp`，Git 只保存 TXT/CSV。
- 完成 occupancy、SM、memory、warp stall、coalescing 和 128/256 正式分析。
- Exp01 保持冻结；Exp02 Gate A/B/C 均为 `PASS`，Overall 为 `PASS`；Exp03 未开始。

## 关键实验结果

- Benchmark mean latency（既有稳定性数据）32/128/256/1024：`6.698908/2.207088/2.257612/3.331965 ms`。
- Theoretical occupancy：`33.33/100.00/100.00/66.67%`；achieved occupancy：`24.69/85.85/83.37/57.25%`。
- SM throughput：`6.65/21.21/21.14/15.07%`；memory/L2 throughput：`32.75/82.45/87.38/48.88%`。
- 主要 stall 全部为 long scoreboard；128/256 ratio `40.60/39.25`。
- 全部 block 的 global load/store 为 32 B/sector，L2 theoretical sectors 等于 ideal，excessive 为 0。
- 128/256 profile SM frequency 为 `509.98/407.99 MHz`；当前 metrics 无法可靠解释既有 2.238% benchmark 差异。
- 128 vs 256 final：`microarchitectural cause remains inconclusive`。
- H1/H2/H3/H4：`SUPPORTED`；Gate A `PASS / FROZEN`，Gate B `PASS`，Gate C `PASS`，Overall `PASS`。

## 本轮未完成 / 限制

- NCU 直接 `DRAM Throughput` 为 `N/A`；没有估算该值。
- Nsight Systems 未执行，本轮 Gate C 不要求该证据。
- Exp01 未 merge 到 `main`；Exp02 未开始。

## 新增证据

- `experiments/Exp01-vector-add/notes/exp01_2_nsight_compute.md`
- `experiments/Exp01-vector-add/benchmark/ncu_profile_summary_20260830T144903Z.csv`
- `experiments/Exp01-vector-add/benchmark/profiler/20260830T144618Z/`
- `experiments/Exp01-vector-add/benchmark/profiler/20260830T144903Z/`
- `experiments/Exp01-vector-add/scripts/run_ncu_profile.sh`

## 下一步建议

项目所有者审核并接受 Exp01 关闭证据。状态为 `READY_FOR_EXP02`，但只有收到明确新任务后才能设计 Exp02；不得自行开始。

## Push / 工作区状态

最终 commit、GitHub push、Jetson fast-forward 与三端 clean 状态必须由本轮结束时的 Git 命令验证，不能从本段预先推断。

## Phase 1.2 Formal Benchmark (2026-09-01)

- Starting branch/HEAD: `phase/01-qwen3-baseline@9a8f825cadcd2c8b9b30d1d58ec30c2cf9139e8f`.
- Qwen3-0.6B exact revision, BF16, eager attention, batch 1; Jetson Orin SM87, CUDA 12.6, NVIDIA PyTorch 2.5.0a0, Transformers 4.57.3.
- Manual KV-cache method validation: 8/8 tokens equal to `generate()`, DynamicCache lengths valid, Gate A PASS.
- Formal ISL 32/128/512/1024: 10 trials each, fixed OSL 32, all TTFT/TPOT CV <= 3.042%, no OOM/CUDA failure; Gates B/C/D PASS; Phase 1.2 PASS/CLOSED.
- Results: `experiments/Phase1-qwen3-baseline/artifacts/phase1_2_formal_20260901T075200Z/`; report: `experiments/Phase1-qwen3-baseline/docs/phase1_2_formal_benchmark.md`.
- Summary CSV: `phase1_2_summary_20260901T081200Z.csv`; all-trials CSV and board-level `VDD_IN`/thermal summary are in the same directory.
- Required Chinese functional regression returned `北京`; no model/runtime settings were changed. No profiler, quantization, TensorRT-LLM, TensorRT engine, Phase 2, or Exp05 work started.
- Phase 1 overall is now `PASS / CLOSED`; TensorRT-LLM backport is not a prerequisite for closure. Phase 2.0 quantization audit is the authorized next action.

## Exp02.0/Exp02.1 (2026-08-31)
- Branch: exp/02-reduction initialized from d3ee572 and synchronized to GitHub/Jetson.
- V1-V7 implemented; correctness matrix 3,087/3,087 PASS. Max absolute error 1.1754035949707031e-4 (V5 signed N=1048589 B=256); max normalized error 2.086155075380606e-8.
- Compute Sanitizer discovery: N/A, command not installed. Gate A/B/C PASS; Overall PASS; READY_FOR_EXP03.
- No Exp01 files or conclusions modified; no benchmark or Nsight run.

## Exp02.2 Benchmark Gate (2026-08-31)
- Gate B PASS after B1 block survey, B2 scaling and B3 five-round stability; Gate C not started at checkpoint.
- Final blocks V1/B512, V2-V7/B128. V5 fastest 1.626439 ms mean (N=16777229); V7 vs V6 paired CI [0.316896, 0.350628] ms.
- Artifacts: benchmark/raw/block_survey_20260830T172929Z.csv, scaling_20260830T172929Z.csv, stability_20260830T172929Z.csv, summaries, block_candidates and paired_comparisons.
- Next permitted action is Gate C only after checkpoint commit/push; Exp03 not started.

## Exp02 Final Closeout (2026-08-31)
- Exp02 Gate A/B/C all PASS; Overall PASS; READY_FOR_EXP03. Exp03 was then initialized on a new branch; its Gate A is PASS and Gates B/C are not started.
- Final benchmark winner V5/B128, 1.626439 ms mean at N=16777229; V7 vs V6 paired CI [0.316896, 0.350628] ms.
- NCU common profiles V1-V7 and V5 B64/B128/B256/B512 sweep saved under benchmark/profiler/20260831T020000Z; .ncu-rep remains Jetson /tmp only.
- H1 SUPPORTED, H2 PARTIALLY_SUPPORTED, H3 SUPPORTED, H4 SUPPORTED, H5 PARTIALLY_SUPPORTED, H6 SUPPORTED, H7 SUPPORTED, H8 SUPPORTED, H9 INCONCLUSIVE.

## Exp03.0/Exp03.1 (2026-08-31)
- Branch: exp/03-matrix-transpose; initialized from Exp02 closeout `8293d6a`.
- Final correctness artifact: `experiments/Exp03-matrix-transpose/benchmark/raw/correctness_20260831T044304Z.csv`; summary: `correctness_summary_20260831T044304Z.csv`.
- V1/V2/V3/V4 each passed 207/207; total 828/828 bitwise exact; guard and CUDA failures 0.
- Initial V1/V2 non-tiled grid_y bug was fixed and all cases rerun; failed first attempt retained on Jetson.
- Gate A PASS; Gate B/C NOT_STARTED; Overall IN_PROGRESS; READY_FOR_EXP03_BENCHMARK.
- Formal benchmark, Nsight Compute and Exp04 were NOT started.

## Phase 2.1 TensorRT Capability Audit (2026-09-01)

- Starting branch/HEAD: `phase/02-qwen3-quantization@be8a3e262ff39dc69c2f434b8d3b3d182a157b38`; frozen venv and Jetson checkout were verified clean before probe work.
- Jetson Orin SM87, CUDA 12.6, TensorRT 10.3.0, `trtexec` v100300 and NVIDIA PyTorch 2.5.0a0 were confirmed. No package/system changes were made.
- ONNX/Polygraphy/ONNX GraphSurgeon are absent; ONNX parser route is `BLOCKED_NO_ONNX_PACKAGE`. Direct TensorRT Python network fallback is clearly labeled in the report.
- Synthetic FP16 Linear built 3/3 and executed 6/6 (M=1/32). Explicit Q/DQ INT8 built 3/3 and executed 6/6 finite with error recorded; status `PARTIALLY SUPPORTED`. INT4 surface flags/types exist, but no public packed weight-only construction path was identified; Gate D `PARTIALLY_SUPPORTED / BLOCKED`.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_1_tensorrt_capability_audit.md`; artifacts: `experiments/Phase2-qwen3-quantization/artifacts/phase2_1_20260901/`. No Qwen3 quantization, TensorRT-LLM, Qwen engine, Phase 2.2 or Phase 3 work started.


## Exp03.2 Benchmark (2026-08-31)
- Two complete benchmark runs: benchmark_20260831T075957Z.csv and benchmark_20260831T080838Z.csv.
- Protocol: six dimensions, V1-V4, warmup 20, repetitions 100, five trials, CUDA Event kernel-only timing, deterministic rotation.
- Correctness sanity passed, but repeated CV >5% remained; Gate B INCONCLUSIVE; Gate C NOT_STARTED.
- No power/clocks changed; do not profile until stability is resolved.

## Exp03.2b Stability Diagnosis (2026-08-31)
- Valid Diagnostic A: `stability_diagnostic_A_20260831T112500Z.csv`; all 4096^2 CVs 0.111-1.047%, so Diagnostic B was not run.
- Formal Benchmark V2: `benchmark_v2_20260831T113200Z.csv`; 168/168 trials retained, all 24 CVs <= 0.791%, minimum actual timed window 793.641 ms.
- Gate B: B1/B2/B3 PASS. Fixed short measurement window is a SUPPORTED CONTRIBUTOR; DVFS/device-state contribution INCONCLUSIVE because frequency fields were unavailable.
- Gate C is the next permitted action. No kernels, power mode, clocks, sudoers, SSH/Git automation, Exp01 or Exp02 were modified.

## Exp03.3 Nsight / Final Closeout (2026-08-31)
- NCU raw/details/summary artifacts are under `experiments/Exp03-matrix-transpose/profiling/raw/20260831T121500Z/`; `.ncu-rep` remains Jetson-local `/tmp/jetson-qwen-exp03-ncu-20260831T121500Z/`.

## Exp04.0 Initial GEMM Foundation (2026-08-31)

- Host-created branch: `exp/04-gemm`, base `fab89aef074f35fac3e98d9ba82a4b08e8560841`; clean before work.
- Added `experiments/Exp04-gemm/` design, V0 CPU FP32 reference, V1 naive one-thread-per-output CUDA GEMM, correctness and adaptive benchmark scripts.
- Jetson execution was restored. V1 built with CUDA 12.6 on Orin CC 8.7; NCU 2024.3.1.0 confirmed. Correctness passed 13/13 with canaries intact and no CUDA failures. Adaptive V1 benchmark collected seven trials for four shapes; CV <= 0.505%. Raw/summary files were copied into the Windows worktree.
- Initialization commit `50689df` was pushed successfully. Result commit is pending this closeout.
- Exp04 V2/V3 execution, formal benchmark and NCU/HMMA SASS evidence are present. The dual-reference artifact `experiments/Exp04-gemm/benchmark/raw/wmma_correctness_dual_reference_20260831T165518Z.csv` records 8/8 aligned WMMA cases against an FP16-quantized reference (Track A) and original FP32 reference (Track B). A1/A2/A3/B/C and Overall are `PASS / CLOSED`. Track B measures end-to-end mixed-precision numerical impact including FP16 input quantization.
- Readiness is `READY_FOR_EXP05_DESIGN` only after explicit authorization. Double buffering/cp.async were not required; Exp05 was not started.
- V2 Shared Memory Tiling and V3 WMMA are complete; double buffering and `cp.async` are not started. Earlier V0/V1-only wording is historical and superseded by this closeout state.
- Gate C C1/C2/C3 PASS. H1-H4 SUPPORTED: V2 store sectors/request 32.105 with 14,735,372 excessive sectors; V3/V4 shared bank conflicts 16,349,795 vs 53,953; achieved occupancy V1/V2/V3/V4 79.20/79.26/94.28/94.47%.
- Exp03 Gate A/B/C PASS; Overall PASS/CLOSED; Readiness READY_FOR_EXP04. Exp04 has not started. Direct DRAM throughput is unavailable and not estimated.

## Phase 1.0 Qwen3 Runtime Feasibility Audit (2026-09-01)

- Branch `phase/01-qwen3-baseline`, created from Exp04 closeout `9ede5e03773d23194f06059c339f32d539f7b7be` after Windows/GitHub/Jetson clean synchronization was verified.
- Gate P1.0 is `PASS WITH CONSTRAINTS`. The recommended Phase 1.1 path is Hugging Face Transformers plus the existing NVIDIA Jetson PyTorch/CUDA stack, starting with a bounded BF16 reference baseline.
- Latest TensorRT-LLM natively supports Qwen3 but SM87 is absent from its tested hardware matrix and the current release dependencies do not match JetPack 6.2. The Jetson-specific 0.12 branch supports SM87/CUDA 12.6 but its code accepts only Qwen/Qwen2/Qwen2-MoE, not Qwen3.
- No packages, weights, source trees or containers were downloaded or installed; no engine, inference, power/clock or system configuration operation occurred. Phase 1.1, Phase 2 quantization and Exp05 Softmax are not started.
- Primary report: `experiments/Phase1-qwen3-baseline/docs/phase1_0_runtime_feasibility_audit.md`; environment evidence: `experiments/Phase1-qwen3-baseline/artifacts/environment_audit_20260901T125855+0800.txt`.

## Phase 1.1 Dependency Checkpoint (2026-09-01)

- Formal venv `/home/nvidia/.venvs/jetson-qwen-phase1-hf` is active with `system-site-packages=true`; NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08` and CUDA 12.6 were preserved.
- Transformers `4.57.3`, Accelerate `1.14.0`, HF Hub `0.36.2`, Safetensors `0.8.0`, Tokenizers `0.22.2`, Regex `2026.9.3`, and hf-xet `1.6.0` import; `pip check` passes.
- Gate A: `PASS`; Gates B/C/D: `NOT STARTED`. Ordinary resolver's PyPI Torch plan was avoided with wheel-only `--no-deps` installation.
- Failed partial `/home/nvidia/.venvs/jetson-qwen-phase1` was preserved untouched. Dependency evidence is under `experiments/Phase1-qwen3-baseline/artifacts/phase1_1_*`.
- Next action: review and perform Gate B exact Qwen3 revision acquisition. No inference, formal benchmark, TensorRT-LLM, Phase 1.2, Phase 2, or Exp05 started.

## Phase 1.1 BF16 Reference Closeout (2026-09-01)

- Model `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca` is stored outside Git; weight SHA256 is `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Gate A/B/C/D are `PASS`; Phase 1.1 is `PASS / CLOSED`. BF16 placement is `cuda:0` only, forward is finite, and three bounded deterministic generations pass.
- PyTorch 2.5/Transformers 4.57.3 requires eager attention because the installed SDPA lacks `enable_gqa`; the failed SDPA run is retained. This is not a performance conclusion.
- Minimum checkpoint MemAvailable was 2,746,023,936 bytes; successful-run swap delta was 68,157,440 bytes and stable. No OOM or offload occurred.
- Formal report: `experiments/Phase1-qwen3-baseline/docs/phase1_1_hf_bf16_reference.md`. Phase 1.2, Phase 2, TensorRT-LLM, TensorRT engine work, and Exp05 are not started.

## Phase 2.1.5 TensorRT Graph Pipeline Enablement (2026-09-01)

- Branch/starting HEAD: `phase/02-qwen3-quantization@def7e66b91e81cd6a42bc8975897ebfcdabf6fda`; only the new graph-pipeline scripts were untracked at start.
- Isolated tools venv `/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools` now imports ONNX 1.22.0, ONNX GraphSurgeon 0.6.1, Polygraphy 0.53.4, TensorRT 10.3.0 and the existing NVIDIA PyTorch 2.5.0a0/CUDA 12.6. NumPy 1.26.4, `ml_dtypes` 0.5.4 and `typing_extensions` 4.15.0 are local to this venv; `pip check` passes.
- Synthetic Linear, MLP/GELU and RMSNorm-like+Linear FP16 opset-17 graphs passed export/checker (3/3), TensorRT parser/build and CUDA execution at M=1/32 (6/6), with context-resolved output shape and `DataType.HALF` allocation. RMSNorm FP16 overflow warning is retained as a risk note.
- Gate A/B/C and Overall `PASS` for this bounded plumbing audit. No Qwen3 export/quantization, benchmark, memory/power measurement, Nsight profile, TensorRT-LLM or Phase 2.2/3 work was started.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_1_5_graph_enablement.md`; evidence: `experiments/Phase2-qwen3-quantization/artifacts/phase2_1_5_20260901/`. Engines/ONNX remain Jetson-local under `/tmp`.

## Phase 2.1.8 Qwen3-like Decoder Block Feasibility (2026-09-01)

- Starting branch/HEAD: `phase/02-qwen3-quantization@bb4dea8286fea60448e476dd82e789822cfd3f54`; Windows and Jetson were clean before the audit.
- Existing tools venv only: PyTorch `2.5.0a0+872d972e41.nv24.08`, CUDA `12.6`, TensorRT `10.3.0`, ONNX `1.22.0`; `pip check` passed. No package, CUDA, TensorRT, JetPack, power or clock setting was changed.
- Synthetic FP16 single decoder block (RMSNorm, RoPE, 16Q/8KV GQA, causal attention, SwiGLU) used the recorded Qwen3-0.6B dimensions but loaded no checkpoint. ONNX checker passed (201 nodes); TensorRT parser had zero errors; FP16 build and batch 1/2 dynamic-profile CUDA execution passed with finite FP16 `[B,8,1024]` output.
- Numerical comparisons are informational only: batch 1/2 max abs 0.005859375 and RMSE 0.000831708/0.000835744. TensorRT warned about FP16 Reduce/Pow overflow risk and default-stream synchronization; retain both limitations.
- Gate A/B/C PASS, Gate D SUPPORTED, Phase 2.1.8 PASS/BOUNDED. Evidence commit `fe0edb8e4239e2e91137f7a6862d59511ed3b74c` is pushed. This does not change Phase 2.1 INT8/INT4 `INCONCLUSIVE` status. No full Qwen3 export, INT8, INT4, TensorRT-LLM, benchmark, Phase 2.2 or Phase 3 was started; BF16 reference unchanged.

## Phase 2.1.9 Full Qwen3 TensorRT Architecture Audit (2026-09-02)

- Starting branch/HEAD: `phase/02-qwen3-quantization@6e2976c1e3b77d3f9f23775b90c1e4e36a2e4dc5`; Windows and Jetson were clean before the audit.
- Read-only architecture audit used frozen config and manifest metadata only. No checkpoint tensor load, full ONNX export, TensorRT engine build, inference or benchmark occurred.
- Qwen3-0.6B map: 28 layers, 1024 hidden, 16Q/8KV heads, head dim 128, MLP 3072, vocab 151936, max positions 40960, tied embeddings. Serialized FP16/BF16 weight lower bound 1.400 GiB; batch-1 KV cache 112 KiB/token, 448 MiB at 4096 and 4.375 GiB at 40960.
- Gate A/B/C PASS; Gate D `BLOCKED_NEEDS_RUNTIME_WORK`. Native TensorRT is partial pending full export and an explicit KV-cache/prefill/decode runtime; TensorRT-LLM remains blocked by current SM87/software-stack intersection; HF TensorRT backend is unknown.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_1_9_full_qwen3_tensorRT_architecture_audit.md`; evidence: `experiments/Phase2-qwen3-quantization/artifacts/phase2_1_9_20260901/`.

## Phase 2.2 Runtime Prototype Preparation (2026-09-02)

- Starting branch/HEAD: `phase/02-qwen3-quantization@0baf237ed9004a418e3cafa597a51d2cf024d738`; Windows and Jetson were clean before preparation.
- Added CPU-only `KVCacheManager`, prefill/decode request dataclasses, KV memory estimate script, attention/runtime design notes and Gate A-D plan under `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/`.
- Synthetic byte-layout validation passed for allocation, partial-layer visibility, all-layer sequence advancement, payload round-trip, decode position and reset. Batch 1/2 cache estimates are theoretical only.
- No Qwen3 checkpoint load, full ONNX export, TensorRT execution/engine, benchmark, INT8, INT4 or TensorRT-LLM work occurred. Runtime execution remains unstarted and requires explicit authorization.
## Phase 2.2-A checkpoint (2026-09-02)

Single-layer synthetic Qwen3-like TensorRT FP16 runtime integration is bounded-pass. Evidence is under `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/artifacts/phase2_2a_20260902/`; report is `phase2_2a_single_layer_trt_runtime.md`. Prefill and dynamic decode engines parse/build and execute on CUDA; cache grows 8->12 with zero prefix mutation. Numerical differences are informational, and TensorRT default-stream/FP16 normalization warnings remain. No full checkpoint/export/full-model engine, benchmark, profiler, INT8/INT4 or TensorRT-LLM work was run. Phase 2.2-B and Phase 3 are not started.

## Phase 2.2-B1 checkpoint (2026-09-02)

Four logical layers reuse the Phase 2.2-A synthetic FP16 prefill/decode engines. B1-1 through B1-5 are `PASS / BOUNDED`: ordered hidden handoff, independent per-layer `[B,8,L,128]` caches, prefill `B=1,S=8`, decode 8->12, CUDA residency and shared runtime stream. Evidence is under `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/artifacts/phase2_2b1_20260902/`; report is `phase2_2b1_multilayer_runtime.md`. No 28-layer deployment, full checkpoint/export, benchmark, Nsight, INT8/INT4 or TensorRT-LLM work was run. Phase 2.2-B2 and Phase 3 are not started.

## Phase 2.2-B2.0 checkpoint (2026-09-02)

Environment bridge audit selected `SOLUTION_C_DUAL_ENV_REQUIRED`. The frozen `phase1-hf` venv has Transformers 4.57.3 and Safetensors 0.8.0; `phase2-trt-tools` has ONNX 1.22, Polygraphy 0.53.4 and TensorRT 10.3 but lacks Transformers/Safetensors. Both preserve NVIDIA PyTorch and pass `pip check`. The installed pip lacks `--dry-run`, so no package installation was attempted. B2 scientific execution remains not started; evidence is under `phase2_2b2_env_20260902/`.
## Phase 2.2-B2 checkpoint (2026-09-02)

Real frozen Qwen3-0.6B Layer 0 integration completed through the dual-env bridge. Model revision and safetensors hash match the frozen identity; 11 Layer 0 tensors (15,730,944 params, 31,461,888 BF16 bytes) mapped exactly. HF Transformers BF16 and portable BF16 match at zero error for prefill and four decode steps; cross-env reproduction also matches exactly. Portable FP16 real layer exported and TensorRT 10.3 built/executed dynamic decode 8->12 with finite CUDA outputs, zero prefix mutation and bounded new-slot errors. B2-1 through B2-8 are `PASS / BOUNDED`; decision `READY_FOR_REAL_MULTILAYER_INTEGRATION`. No full model or 28-layer runtime was built.
## Phase 2.2-B3 checkpoint (2026-09-02)

Real Qwen3 layers 0-3 were mapped and reproduced exactly by HF/portable BF16. A single four-layer TensorRT 10.3 FP16 prefill graph and a single dynamic decode graph passed ONNX checker, parser/build and CUDA execution. Decode advanced 8->12 with independent per-layer KV pointers, zero prefix mutation and finite CUDA outputs. Numerical propagation and attention-output metrics are bounded and recorded under `phase2_2b3_20260902T055036Z/`; no extrapolation to 28 layers is made. B3-1 through B3-8 are `PASS / BOUNDED`; decision `READY_FOR_28_LAYER_DECODER_STACK_FEASIBILITY`. Layers 4-27, full model, benchmark, profiler and quantization remain unstarted.
## Phase 2.2-B4 checkpoint (2026-09-02)

B4 preflight verified the frozen model identity, then the monolithic 28-layer HF oracle was killed with exit 137 before artifact output. The predefined 7x4 partitioned fallback was attempted once and was also killed with exit 137. No handoff, ONNX export, TensorRT build or runtime validation was started. Status is `BLOCKED_BY_28L_ORACLE_MEMORY`; existing B3 evidence and the Jetson stash backup remain untouched.

## Phase 2.2-B4.1 checkpoint (2026-09-02)

The exact Phase 1 BF16 full-model load and short forward passed, excluding a current intrinsic model-load failure under the measured state. Static audit plus dynamic recovery confirm `IMPLEMENTATION_MEMORY_LIFETIME_CONFIRMED`: old B4 retained all 28 CPU layer states, overlapping CUDA copies and large reference/handoff trees. Streaming extraction produced 28 independently hashed `/tmp` layer files, while fresh-process 4-layer, 8-layer and one 28-layer oracle attempt showed fixed allocator reservation and expected KV-only growth. The 28-layer `S=8` prefill and one decode `8->9` passed without exit 137. B4.1 is `PASS / CLOSED`; decision `B4_ORACLE_MEMORY_PATH_RECOVERED`. No 28-layer ONNX/TensorRT, benchmark, profiler, quantization or later phase started. The pre-existing Jetson stash remains preserved.

## Phase 2.2-B4.2 checkpoint (2026-09-02)

The frozen 28-file B4.1 handoff mapped 28/28 real Qwen3 decoder layers. The primary one-stack design exported checker-valid FP16 opset-17 prefill/decode graphs and built two TensorRT 10.3 engines without OOM, exit 137, or fallback. B=1,S=8 prefill and four decode steps through 8->12 passed for all layers with finite CUDA tensors, exact old K/V prefixes and 28-way per-type pointer isolation. Selected Layer 0/3/7/15/27 propagation passed the predefined relative-L2/cosine bound. B4.2 is `PASS / BOUNDED`; decision `REAL_28_LAYER_TRT_DECODER_STACK_FEASIBLE`. Large ONNX/engine files remain Jetson-local under `/tmp/phase2_2b4_2_20260902T082326Z/`. TensorRT FP16 normalization and default-stream warnings remain. No embedding, final norm, LM head, sampling, token generation, benchmark, Nsight, quantization, TensorRT-LLM or later phase was started. The existing stash remains preserved.

## Phase 2.2-C1D checkpoint (2026-09-02)

Starting HEAD was `f2249dd353b85d849292b59ca08c2a418888b36a`. Added the non-destructive boundary diagnostic `src/phase2_2c1/c1d_decoder_boundary_diagnostic.py` and raw artifact `phase2_2c1d_20260902T180000Z_c1d_diagnostic.json`. D0 B4.2 current control passed (relative-L2 `0.0201395`, cosine `0.9997966`). D1 canonical embedding was byte-identical (SHA256 `da04f533...`); D2 still diverged progressively to Layer 27 (relative-L2 `2.004247`, cosine `0.534268`). D3 host-staged and direct-device paths failed identically. Both streams were pointer `0` with explicit synchronization; prefill has only `hidden_states` FP16 and `position_ids` INT64 inputs. No stream/pointer/lifetime/binding defect was confirmed. C1 remains `BLOCKED`; C2 must not start. No engine or historical artifact was changed; no OOM/exit 137 occurred.

## Phase 2.2-C1E checkpoint (2026-09-02)

Added `src/phase2_2c1/c1e_layerwise_localization.py`, reusing the existing B4.2 engine and C1 artifacts. The run completed with `FIRST_DIVERGENCE_FOUND`: all 28 hidden outputs were `[1,8,1024]` FP16 and finite; first non-zero divergence was Layer 0, first relative-L2 alert (>0.10) Layer 21, and first cosine alert (<0.99) Layer 22. D0 remained PASS. Mean absolute error is recorded per layer in `phase2_2c1e_layerwise_20260902T190000Z.json`. Component attribution is `NOT PERFORMED` because no internal intermediate bindings exist. C1 remains BLOCKED and C2 must not start; no OOM, rebuild, benchmark, profiler or quantization work occurred.

## Phase 2.2-C1F checkpoint (2026-09-02)

Starting HEAD was `6e48add7acdb40a64f13920efd7bb27b615ec3bb`. Added the non-destructive `src/phase2_2c1/c1f_layer0_component_attribution.py` diagnostic and report. The run reused the existing B4.2 prefill engine, canonical C1 embedding and Layer 0 handoff. Qwen3 Layer 0 structure was confirmed as input/post RMSNorm, Q/K normalization, RoPE (rotary 128, theta 1e6), 16Q/8KV GQA (repeat 2), attention/output projection, residuals and SwiGLU MLP. B4.2 has 84 outputs but no internal component bindings, so all requested component metrics are `NOT_AVAILABLE` and first divergent operator is `UNKNOWN`; result is `COMPONENT_LOCALIZATION_BLOCKED`. Existing B3 `attention_l0` partial evidence is max abs `0.01171875`, relative-L2 `0.00316544`, cosine `0.99999416`, and cannot establish 28-layer attribution. C1 remains BLOCKED; C2 must not start. Raw artifact: `phase2_2c1f_layer0_component_20260902T193000Z.json`.

## Phase 2.2-C1G checkpoint (2026-09-02)

Starting HEAD was `b8c9bc8edbb123e989a8a734eac6627077608336` on `phase/02-qwen3-quantization`. Added `src/phase2_2c1/c1g_layer0_internal_probe.py`, the internal-probe report and three raw Group A/B/C JSON artifacts. Three independent diagnostic engines passed probe validity against B4.2; B4.2 remained read-only. The first non-zero component difference was input RMSNorm (relative-L2 `0.000491762`), while QK scores were nearly identical (`1.95e-7`) and softmax increased to `0.003432333`, the first material amplification. Result is `FIRST_MATERIAL_OPERATOR_FOUND`, with root cause `NARROWED / NOT CONFIRMED`; no precision A/B or repair was performed. C1 remains `BLOCKED`; C2 must not start. ONNX/engine binaries remain ignored Jetson-local diagnostic artifacts; raw JSON is the repository evidence. Report: `phase2_2c1_layer0_internal_probe.md`.
