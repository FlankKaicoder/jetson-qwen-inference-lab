# Resume Evidence Extraction Report

Project: Jetson Qwen Transformer AI Infra Optimization Lab
Repository: `FlankKaicoder/jetson-qwen-inference-lab`
Audit date: 2026-09-09 (Asia/Shanghai)

This report records repository evidence only. It is not a resume narrative. Values are copied from reports and raw artifacts; timing boundaries and gate limitations are preserved.

## Current Project Status

| Field | Evidence-backed value |
| --- | --- |
| Working directory | `E:\nvidia-qwen` |
| Branch | `phase/08-rmsnorm-optimization` |
| Starting HEAD | `6fb18773014a43f51c82b91338f2972131a4ce86` |
| Remote | `origin` = `git@github.com:FlankKaicoder/jetson-qwen-inference-lab.git` |
| Current phase | Phase 8.4-A, Qwen3 TensorRT RMSNorm plugin impact evaluation |
| Current gate | `BOUNDED / NO_END_TO_END_SPEEDUP` |
| Interpretation | Layer-0 plugin node correctness passed; full-model numerical equivalence is `INCONCLUSIVE`; plugin latency was slower in both modes |
| Working tree at audit start | Existing unrelated untracked experiment/audit paths were present and preserved; no existing file was modified or deleted |
| Next action recorded by project state | Stop after Phase 8.4-A; do not replace all 113 RMSNorm nodes without authorization |

Repository structure inspected: `docs/`, `results/`, `experiments/`, Phase 5/6/8 source trees, CUDA kernel directories, TensorRT plugin directories, benchmark/profiler artifacts, and scripts. The source-of-truth documents were `README.md`, `ROADMAP.md`, `AGENTS.md`, `docs/project_management.md`, `docs/PROJECT_STATE.md`, `docs/experiment_index.md`, and `docs/handoff/current_state.md`.

## Phase Evidence Table

| Phase | Experiment goal | Actually completed | Key technology | Quantitative result | Evidence files |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | CUDA execution model: thread/block/grid, warp, shared memory, reduction, GEMM, memory movement | Vector Add, Reduction V1-V7, Matrix Transpose V1-V4, and GEMM V0-V3 were implemented, correctness-tested, benchmarked, and partially profiled | CUDA C++, shared-memory reduction/tiling, WMMA, Nsight Compute | Vector Add: `N=2^24`, B128 `2.171583 ms`, effective bandwidth `92.710 GB/s`; correctness `77/77`. Reduction: V5/B128 `1.626439 ms` for `N=16,777,229`; V3/V4 shared conflicts `6,952,266/48,955`. Transpose 4096x4096: V1/V2/V3/V4 `1.961324/10.018314/2.702256/1.425565 ms`; V4/V1 `1.376x` latency ratio; V4 achieved occupancy `94.47%`. GEMM WMMA has Tensor pipe activity and `HMMA.16816.F32` SASS evidence | `experiments/Exp01-vector-add/`; `experiments/Exp02-reduction/`; `experiments/Exp03-matrix-transpose/`; `experiments/Exp04-gemm/` |
| Phase 1 | Qwen3 deployment baseline | Qwen3 0.6B BF16 reference loaded on Jetson; 28-layer TensorRT decoder stack and HF eager baseline were measured | Hugging Face Transformers, TensorRT 10.3, BF16, GQA, KV cache, Prefill/Decode | Qwen/Qwen3-0.6B, 28 layers, hidden 1024, intermediate 3072, Q/KV heads `16/8`, head dim 128. Jetson Orin Nano Super, 25 W. HF prefill medians S32/128/512/1024 `128.8281097/128.3848572/300.9336548/614.4094543 ms`; TPOT medians `125.2064855/121.1677301/118.4400550/115.5420662 ms`. Throughput, memory peak, and Phase 1 kernel attribution: `UNKNOWN` | `experiments/Phase1-qwen3-baseline/docs/phase1_1_hf_bf16_reference.md`; `experiments/Phase1-qwen3-baseline/docs/phase1_2_formal_benchmark.md`; `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/docs/phase2_2b4_2_real_28layer_trt.md` |
| Phase 2 | FP16, INT8, mixed precision and calibration | TensorRT synthetic FP16/INT8 capability; 28-layer FP16 runtime; explicit Q/DQ mixed policy; calibration and held-out evaluation | TensorRT Q/DQ, PT-W8A8, calibration, KV cache, final RMSNorm, LM head, greedy decode | 196 Linear targets (28 x 7), policy `63 FP16 + 133 PT-W8A8`, 24 calibration prompts and 12 held-out prompts, selected `BOUNDED_MSE_CLIP`, scale `0.0243602362`. Activation-only P95 relative-L2 `0.020111`. Engine size about `-27%`; mixed prefill about `+48%`, decode about `+35%` slower; throughput about `0.51 -> 0.38 tokens/s`. Runtime gate `MIXED_RUNTIME_SLOWER`; HF/TRT decoder drift unresolved (`C1 CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`) | `experiments/Phase2-qwen3-quantization/docs/phase2_3b_calibration_activation_range_audit.md`; `.../phase2_3d_mixed_precision_policy.md`; `.../phase2_3f_accuracy_memory_performance_comparison.md`; `experiments/Phase2-qwen3-quantization/artifacts/phase2_3f_20260904T050000Z/` |
| Phase 3 | TensorRT runtime and kernel analysis | Mixed slowdown was attributed to runtime lifetime first, then persistent contexts and steady-state kernel contributions were measured; CUDA Graph feasibility was tested; small GEMM microarchitecture was profiled | Engine Inspector, Nsight Systems, Nsight Compute, persistent `IExecutionContext`, CUDA Graph | Legacy-to-persistent mixed prefill `2259.919 -> 47.627 ms`, TPOT `2662.990 -> 43.362 ms`, gap recovery `99.7%-100.3%`; steady-state FP16/Mixed kernel time `219.709696/147.830560 ms`; largest single-kernel share `35.31%/26.78%`. CUDA Graph replay failed functionally: `3376` persistent kernels vs `64` FillFunctor kernels. Rank-1 h16816: memory/L2 `97.06%`, SM `39.57%`, HMMA active `39.673148%`, achieved occupancy `24.78%`; direct DRAM throughput `N/A` | `docs/phase3a_runtime_bottleneck_attribution.md`; `docs/phase3b_runtime_object_lifetime.md`; `docs/phase3c_residual_runtime_profiling.md`; `docs/phase3d0_cuda_graph_feasibility.md`; `docs/phase3e_kernel_attribution.md`; `results/phase3*/` |
| Phase 4 | Generate and test optimization hypotheses | 250 GEMM candidates inventoried; 7 projection families mapped across 28 layers; `up_proj` selected, benchmarked, correlated to kernels, and closed without a proven replacement | Operator attribution, Engine Inspector, NSYS NVTX/kernel correlation, hypothesis gating | `up_proj`: `C[1,3072] = A[1,1024] * B[1024,3072]`. Standalone PyTorch median `0.086364701 ms`; TensorRT IProfiler per-layer medians `153.280005-159.759998 us`; 28/28 layer mapping; 196 launches, 7/range, h16816 `84/196`, `sm80_xmma_gemm_*` `112/196`. H1 memory/L2 direction was proposed, but CUDA readiness `NOT READY`; no replacement accepted | `results/phase4a_operator_attribution/20260904T132820Z/`; `results/phase4e_up_proj_baseline/20260904T161320Z/`; `results/phase4f_kernel_attribution/20260904T164404Z/`; `results/phase4g_optimization_hypothesis/20260905T040918Z/` |
| Phase 5 | GEMM feasibility: cuBLASLt, CUTLASS, TensorRT comparison | Correctness-passing standalone cuBLASLt and CUTLASS candidates were measured; TensorRT xmma was compared under matched NCU sections; no custom CUDA target was authorized | cuBLASLt, CUTLASS v3.5.1, TensorRT tactics, NCU | cuBLASLt FP32-accumulate algorithm 21 median `0.080077961 ms`, CUTLASS best `0.083619133 ms` (`4.42%` slower), TensorRT IProfiler per-layer `0.153280005-0.159759998 ms`. Matched NCU: cuBLASLt `242.912 us`, TensorRT xmma `244.160 us`; memory/L2 `76.06%/76.92%`. Gate `PASS / BOUNDED / NO_PROVEN_CUDA_GEMM_OPTIMIZATION_TARGET`; TensorRT backend identity and numeric tactic ID `UNKNOWN/NOT_AVAILABLE` | `results/phase5a_cuda_feasibility_baseline/20260905T063059Z/`; `results/phase5b_tensorrt_gemm_path_investigation/20260905T112513Z/`; `results/phase5_closeout_and_target_reselection/20260905T115247Z/phase5_final_closeout_report.md` |
| Phase 6 | Attention structure, GQA, KV cache, decode, and FlashAttention feasibility | QK and Attention x V semantic/runtime attribution was completed with bounded conclusions; no FlashAttention or replacement implementation was authorized | GQA, KV cache shape tracing, NVTX/runtime/kernel correlation, xmma GEMM | QK direct cache progression: past KV lengths `8-11`, key lengths `9-12`, 16 Q heads, 8 KV heads, repeat factor 2; four QK xmma kernels had no proven workload ratio. Attention x V: 28 layers, 112 correlated kernels, 112 `cuLaunchKernelEx`, one exact xmma family; representative steady contribution `4,237,568 ns / 2.866503%`, all-trace share `1.834842%`. Exact kernel arguments, workload shapes, tactic/backend identity and direct AV NCU metrics remain `UNKNOWN`; FlashAttention `NOT AUTHORIZED`; no optimization conclusion | `results/phase6c_qk_dynamic_path_attribution/20260906T061115Z/`; `results/phase6e_qk_runtime_shape_invocation_attribution/20260906T083753Z/`; `results/phase6g_attention_v_runtime_attribution/phase6g_20260906T151718Z/`; `results/phase6h_attention_v_feasibility_boundary/20260906T160259Z/` |
| Phase 7 | Reassess all remaining optimization targets | Repository-only synthesis of Phase 3-6 evidence; candidate ranking updated, no new run or implementation | Evidence matrix, candidate scoring, gate discipline | 11 candidates inventoried. Attention x V status `NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY`; Fused QKV and gate_proj `ATTRIBUTION_ONLY`; RMSNorm/RoPE `NO_CURRENT_ACTION`. Overall gate `PASS / BOUNDED / NO_PROVEN_CUSTOM_KERNEL_OPTIMIZATION_TARGET` | `results/phase7_global_optimization_target_reassessment/20260907T080402Z/` |
| Phase 8 | RMSNorm CUDA kernel and TensorRT plugin path | PyTorch baseline, CUDA V0/V1/V2, root NCU profiles, synthetic TensorRT plugin, one real Qwen3 node integration, and one-node full 28-layer impact evaluation completed | Warp reduction, `half2`/`bfloat162`, NCU, TensorRT `IPluginV3`, Qwen3 node replacement | CUDA FP16 prefill V0/V1/V2 `0.02369286405/0.02021680002/0.01891695993 ms`; decode `0.02214268798/0.01916262398/0.01759222411 ms`; all 12 correctness cases passed. NCU prefill V0/V1/V2 `25.664/22.656/20.448 us`, achieved active warp `65.43/65.32/29.78%`, direct DRAM `N/A`. Real Layer-0 plugin node relative-L2 `0.0003934488`, plugin `0.0147100797 ms` vs original `0.0131521601 ms` (`+11.84%`). Full graph Layer-0 relative-L2 `0.0006377380` prefill / `0.0008134234` decode; final hidden-layer relative-L2 `0.0221897/0.0143852`, full-model equivalence `INCONCLUSIVE`; end-to-end latency `38.5183 -> 39.6601 ms` prefill (`+2.96%`) and `41.7803 -> 42.6711 ms` decode (`+2.13%`) | `experiments/Phase8-rmsnorm-optimization/docs/`; `experiments/Phase8-rmsnorm-optimization/artifacts/phase8_0_*/`; `phase8_1_*/`; `phase8_2_*/`; `phase8_3B_20260908T_repro/`; `phase8_4A_20260909T/` |

## Resume Quantitative Evidence

### Hardware and software

| Item | Evidence |
| --- | --- |
| Jetson | Jetson Orin Nano Super; SM 8.7; 8 SM; 25 W mode recorded for Phase 1 |
| CUDA | 12.6 / 12.6.68 in Phase 8 environment records |
| TensorRT | 10.3.0 (v100300; one environment record reports `10.3.0.30-1+cuda12.5`) |
| Nsight Compute | 2024.3.1.0 |
| Nsight Systems | 2024.5.4.34 in Phase 8 audit; Phase 3 reports use the recorded environment |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| GPU memory | `7,989,940,224` bytes total in Phase 8.4-A process snapshots; peak allocation is `UNKNOWN` |

### Model and architecture

| Item | Evidence |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Parameter/checkpoint evidence | 751,632,384 serialized elements; BF16 checkpoint; SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Layers / hidden / intermediate | `28 / 1024 / 3072` |
| Attention heads | Q heads `16`; KV heads `8`; GQA repeat factor `2`; head dim `128` |
| Vocabulary / normalization | vocab `151936`; RMSNorm epsilon `1e-6`; RoPE theta `1,000,000` |
| RMSNorm inventory | `113` tensors: 56 input/post-attention `[1024]`, 56 Q/K norm `[128]`, 1 final `[1024]` |

### Performance and profiler numbers

| Metric | Measured evidence |
| --- | --- |
| HF baseline latency | Prefill medians `128.8281097-614.4094543 ms` over S32-S1024; TPOT medians `115.5420662-125.2064855 ms` |
| Quantized runtime | Mixed prefill about `48%` slower, decode about `35%` slower; throughput `0.51 -> 0.38 tokens/s`; engine size about `27%` smaller |
| Runtime lifetime fix | Mixed prefill `2259.919 -> 47.627 ms`; TPOT `2662.990 -> 43.362 ms`; `99.7%-100.3%` recovery |
| GEMM | cuBLASLt `0.080077961 ms`; CUTLASS `0.083619133 ms`; matched NCU `242.912/244.160 us` cuBLASLt/TensorRT |
| Attention x V | `112` attributed kernels; `2.866503%` representative steady contribution; all-trace share `1.834842%` |
| RMSNorm CUDA | V2 FP16 prefill/decode `0.01891695993/0.01759222411 ms`; V2 vs V0 approximately `20.2%/20.6%` lower event mean |
| RMSNorm NCU | V0/V1/V2 prefill `25.664/22.656/20.448 us`; achieved active warp `65.43/65.32/29.78%`; global sectors `2048/2048/1536`; excessive sectors `0/0/0` |
| RMSNorm plugin | Isolated real node `0.0131521601 -> 0.0147100797 ms` (`+11.84%`); full graph prefill/decode `+2.96%/+2.13%` |
| Kernel occupancy and memory | Only reported achieved occupancy values are the cited NCU measurements. Theoretical occupancy is not substituted. Direct DRAM throughput is `N/A`; effective bandwidth is not treated as a DRAM counter |

### Code, experiment, benchmark, and profiler inventory

These are repository file counts from the audit, not claims of unique experiment runs:

| Inventory | Count / status |
| --- | --- |
| CUDA source files (`.cu`/`.cuh`) | `17` |
| Benchmark-named source files | `13` |
| Benchmark-named result/report files | `183` |
| Nsight Compute-named evidence files | `56` |
| Nsight Systems-named evidence files | `176` |
| Major phase coverage | `Phase 0` through `Phase 8` (`9` phases) |
| Unique benchmark run count across heterogeneous raw formats | `UNKNOWN`; repository does not define one normalized run registry |
| CUDA kernel count in the full TensorRT engines | `UNKNOWN`; Phase 3/6 report launch counts are workload-specific, not a static engine kernel count |

## Technical Capability Mapping

| Capability | Evidence |
| --- | --- |
| CUDA Programming | Vector Add, Reduction V1-V7, Transpose V1-V4, GEMM V0-V3, RMSNorm V0-V2; 17 `.cu/.cuh` files; correctness and CUDA Event benchmarks |
| GPU Architecture | SM 8.7, warp/block/shared-memory analysis, Tensor Core WMMA, NCU achieved active warp, instruction and sector metrics |
| TensorRT | 28-layer FP16 decoder stack, Engine Inspector, persistent execution contexts, TensorRT 10.3 build/deserialize/execute paths |
| Nsight Profiling | NCU artifacts for CUDA primitives, GEMM, runtime GEMMs, and RMSNorm; NSYS NVTX/runtime/kernel correlation for Phases 3-6 |
| LLM Quantization | Explicit Q/DQ INT8 capability, PT-W8A8 mixed policy, 196 targets, calibration and held-out prompts, measured mixed slowdown |
| Transformer Analysis | Qwen3 28-layer architecture, projection-family inventory, operator attribution, decoder numerical drift localization |
| Attention/GQA | 16 Q heads / 8 KV heads / repeat 2; QK and Attention x V attribution reports; direct runtime cache-shape progression |
| KV Cache | Prefill/decode engines, per-layer K/V outputs, decode key lengths and past lengths, exact-prefix checks in Phase 2 and Phase 6 |
| CUDA Kernel Optimization | Warp-shuffle reduction and vectorized RMSNorm V2; NCU explains lower synchronization/shared work; full-model speedup not proven |
| RMSNorm | 113-tensor inventory, explicit FP32 reference, V0/V1/V2 CUDA kernels, TensorRT `IPluginV3`, Qwen3 Layer-0 and full-graph bounded tests |
| GEMM/Tensor Core | Exp04 WMMA `HMMA.16816.F32`; cuBLASLt/CUTLASS/TensorRT comparison; xmma Tensor Core tactic strings and NCU evidence |

## Evidence Limitations

- Phase 1 profiler attribution, model peak memory, and direct full-model throughput are `UNKNOWN` where no committed evidence exists.
- Phase 2 decoder numerical drift remains unresolved; mixed precision runtime slowdown is measured but is not an accuracy-pass claim.
- Phase 3-7 use strict boundary labels. Theoretical occupancy, effective bandwidth, or device utilization is not substituted for achieved occupancy or DRAM counters.
- Phase 5 TensorRT versus standalone GEMM timing boundaries differ; the historical gap is `EVENT_TIME_GAP_INCONCLUSIVE`.
- Phase 6 QK/Attention x V exact runtime GEMM shapes, arguments, tactic IDs, backend identity, and direct AV NCU metrics remain `UNKNOWN`; FlashAttention was not authorized.
- Phase 8 V2 causal decomposition is `INCONCLUSIVE` because vectorization, block size, registers, and shared-memory shape changed together. The plugin accepted the graph but produced no end-to-end speedup, and full-model numerical equivalence is `INCONCLUSIVE`.

All referenced raw result directories and reports were read during this audit. No existing experiment record or result artifact was modified.
