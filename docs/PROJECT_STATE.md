# Project State

本文件是无历史上下文的新会话恢复项目状态的入口。完整实验方法与分析以实验报告和 raw artifacts 为准；若本页与更高优先级证据冲突，以 Git 和 raw artifacts 为准。

## Current State

| Field | Verified value |
| --- | --- |
| Project | `jetson-qwen-inference-lab` / Jetson Qwen Transformer AI Infra Optimization Lab |
| Current date | `2026-09-05` |
| Repository | `FlankKaicoder/jetson-qwen-inference-lab` |
| Windows path | `E:\nvidia-qwen` |
| Jetson path | `/home/nvidia/projects/jetson-qwen-inference-lab` |
| GitHub | `https://github.com/FlankKaicoder/jetson-qwen-inference-lab` |
| Current phase | Phase 5 — GEMM Investigation Freeze + Next Target Re-selection |
| Current experiment | Phase 5 Closeout |
| Current branch | `phase/05a-cuda-feasibility-baseline-study` |
| Current HEAD | Verify with `git rev-parse HEAD` |
| Main HEAD | `d42ab4aeabc751723a4a2c1036b93a5ed16d3d01` |
| Last completed experiment | Phase 5 Closeout — GEMM feasibility freeze and next target re-selection |
| Experiment status | Prior Phase 1-4, Phase 5-A and Phase 5-B Step 2 statuses are unchanged. Phase 5 was closed from committed evidence without new execution or profiling. |
| Current Gate | Phase 5 Closeout is `PASS / BOUNDED / NO_PROVEN_OPTIMIZATION_TARGET`. The next decision is `NEXT_TARGET_BOUNDED`; `unknown_attention_matmul` is an attribution-only candidate. |
| Readiness | Stop after Phase 5 closeout. No CUDA kernel, TensorRT Plugin, engine rebuild, tactic forcing, implementation, or new operator benchmark is authorized. |

## Confirmed Findings

- Baseline is FP32 `C[i] = A[i] + B[i]`, one element per thread, with bounds checking.
- Original correctness sweep: 11 sizes × 7 supported block sizes = 77/77 `PASS`; maximum absolute error `0`.
- Original benchmark: `N = 2^20, 2^22, 2^24`; block sizes `16, 32, 64, 128, 256, 512, 1024`; warmup `20`; repetitions `200`; CUDA Event kernel-only timing.
- Exp01.1 stability: `N=16,777,216`, six block sizes, five independent rounds; 30/30 correctness PASS. Block 128 was fastest in 5/5 rounds with mean `2.207088 ms`; block 256 mean was `2.257612 ms`.
- Nsight Compute 2024.3.1 successfully profiled blocks 32, 128, 256 and 1024 using the original workload. All four target runs passed correctness.
- Theoretical occupancy for 32/128/256/1024 is `33.33/100.00/100.00/66.67%`; achieved occupancy is `24.69/85.85/83.37/57.25%`.
- SM throughput is `6.65/21.21/21.14/15.07%`; memory/L2 throughput is `32.75/82.45/87.38/48.88%`.
- Long scoreboard is the main reported warp stall for every profiled block.
- Global loads/stores use 32 B/sector; L2 theoretical global sectors equal ideal sectors and excessive sectors are zero for all four blocks.
- H1, H2, H3 and H4 are `SUPPORTED` under the final evidence set.

## Rejected / Disproven Hypotheses

- “Larger block size is necessarily faster” is disproven; block 128 outperformed 256, 512 and 1024 under the recorded benchmark conditions.
- “Equal or higher occupancy guarantees equal or better latency” is disproven; blocks 128 and 256 both have 100% theoretical occupancy and close achieved occupancy, but different stable benchmark means.

No repository evidence records a formally `REJECT`-status experiment.

## Remaining Inconclusive / N/A Items

- 128 vs 256 precise cause: `microarchitectural cause remains inconclusive`. Occupancy, SM, stall and coalescing metrics are close; 256's normalized memory throughput is higher, not lower.
- Profile SM frequency differed for 128/256 (`509.98/407.99 MHz`), so profiler duration and raw traffic rate cannot provide a clean causal comparison of the independent benchmark gap.
- Direct NCU `DRAM Throughput`: `N/A` on this integrated platform; it was not estimated.
- Nsight Systems: `UNKNOWN` and not required for the completed Gate C definition.

## Known Blockers

- No blocker remains for Exp03 Gate A.
- Exp03 is not merged to `main`; merging requires explicit direction.

## Current Artifacts

- Main report: `experiments/Exp01-vector-add/README.md`
- Stability audit: `experiments/Exp01-vector-add/notes/exp01_1_stability_and_profile.md`
- Nsight closure report: `experiments/Exp01-vector-add/notes/exp01_2_nsight_compute.md`
- Original correctness/benchmark: `experiments/Exp01-vector-add/benchmark/correctness_results.csv`, `experiments/Exp01-vector-add/benchmark/vector_add_benchmark.csv`
- Stability raw/summary: `experiments/Exp01-vector-add/benchmark/stability_raw_20260827T075423Z.csv`, `experiments/Exp01-vector-add/benchmark/stability_summary.csv`
- NCU summary: `experiments/Exp01-vector-add/benchmark/ncu_profile_summary_20260830T144903Z.csv`
- NCU TXT/CSV evidence: `experiments/Exp01-vector-add/benchmark/profiler/20260830T144618Z/`, `experiments/Exp01-vector-add/benchmark/profiler/20260830T144903Z/`
- Reproducible runner: `experiments/Exp01-vector-add/scripts/run_ncu_profile.sh`
- `.ncu-rep` files: Jetson-local `/tmp/jetson-qwen-exp01-ncu/20260830T144618Z/` and `/tmp/jetson-qwen-exp01-ncu/20260830T144903Z/`; intentionally outside Git.
- Important prior commits: `249ddfb0a6873765bc391922111acfdd489e6d5c`, `74087eddc3815c45bae655978b57e99279dd4bd8`, `e10f2c06c7d90844cbf425e5ef6c32a413e314ec`.

## Phase 1.1 Dependency Checkpoint (2026-09-01)

- Formal venv: `/home/nvidia/.venvs/jetson-qwen-phase1-hf`, created with `--system-site-packages`; system NVIDIA PyTorch remains unchanged at `/usr/local/lib/python3.10/dist-packages/torch`.
- Gate A is `PASS`: pinned HF packages import, `pip check` passes, CUDA remains available on Orin SM87, and BF16 is supported.
- Ordinary pip resolution attempted PyPI `torch-2.13.0`; installation was blocked and replaced by wheel-only `--no-deps` installation. No PyPI Torch was installed.
- `/home/nvidia/.venvs/jetson-qwen-phase1` is preserved untouched as `FAILED_PARTIAL_ENV`.
- Evidence: `experiments/Phase1-qwen3-baseline/artifacts/phase1_1_*`.

## Phase 1.1 BF16 Reference Closeout (2026-09-01)

- Exact model: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`; `model.safetensors` SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Qwen3 loaded with BF16 parameters on `cuda:0` only, with no CPU/disk offload. Forward logits were finite and all three deterministic bounded generations passed semantic sanity.
- Current PyTorch 2.5 SDPA lacks the `enable_gqa` argument required by Transformers 4.57.3's SDPA path; the functional reference explicitly uses built-in eager attention. No performance conclusion is attached to this choice.
- Model-load allocator delta was 1,192,638,976 bytes; minimum checkpoint MemAvailable was 2,746,023,936 bytes. Successful-run swap increased 68,157,440 bytes and remained stable; no OOM occurred.
- Gate A/B/C/D: `PASS`; Phase 1.1: `PASS / CLOSED`. Report: `experiments/Phase1-qwen3-baseline/docs/phase1_1_hf_bf16_reference.md`.

## Phase 2.0 Quantization Backend Audit (2026-09-01)

- Isolated venv `/home/nvidia/.venvs/jetson-qwen-phase2-quant` preserved NVIDIA Torch 2.5.0a0 and CUDA 12.6; Phase 1 venv remains untouched.
- TorchAO 0.12.0 installed with a safe `--dry-run --no-deps` plan, but import failed on missing `torch._C._distributed_c10d` from eagerly imported Float8/distributed support. W8A16, A8W8 and W4A16 probes were not claimed as executable.
- Native operator names `_weight_int4pack_mm` and `_weight_int8pack_mm` are present, but operator presence is not runtime quantization evidence. No full model, formal benchmark, TensorRT engine or bitsandbytes build was run.
- Gate A `PASS`; Gate B/C `BLOCKED`; Gate D `INCONCLUSIVE`; Phase 2.0 is `BLOCKED` at the TorchAO candidate pending an explicitly authorized compatible backend path.

## Phase 2.1 TensorRT Capability Audit (2026-09-01)

- Frozen Jetson venv verified with Orin SM87, CUDA 12.6, TensorRT Python 10.3.0, `trtexec` v100300 and NVIDIA PyTorch 2.5.0a0. No package/system changes were made.
- ONNX tooling is absent (`onnx`, Polygraphy and ONNX GraphSurgeon); the requested parser path is `BLOCKED_NO_ONNX_PACKAGE`.
- Direct TensorRT fallback built/executed synthetic `1024 -> {1024,2048,3072}` Linear: FP16 3/3 engines and 6/6 M=1/32 executions PASS; explicit Q/DQ INT8 3/3 engines and 6/6 executions finite with numerical error recorded, status PARTIALLY SUPPORTED.
- INT4 flag/DataType/API surface exists, but no public packed weight-only construction path was identified; Gate D `PARTIALLY_SUPPORTED / BLOCKED`. Overall Phase 2.1 `INCONCLUSIVE`; no performance, memory or power claim.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_1_tensorrt_capability_audit.md`; evidence: `experiments/Phase2-qwen3-quantization/artifacts/phase2_1_20260901/`.

## Phase 2.1.5 TensorRT Graph Pipeline Enablement (2026-09-01)

- Isolated tools venv `/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools` uses NumPy 1.26.4, `ml_dtypes` 0.5.4 and `typing_extensions` 4.15.0; `pip check` passes. Existing NVIDIA PyTorch 2.5.0a0, CUDA 12.6, TensorRT 10.3.0 and NCU 2024.3.1.0 remain unchanged.
- Synthetic FP16 opset-17 Linear, MLP/GELU and RMSNorm-like+Linear graphs passed export and ONNX checker (3/3), TensorRT parser/build and CUDA `execute_async_v3` for M=1/32 (6/6); output buffers use context-resolved shapes and declared `DataType.HALF`.
- Gate A/B/C and Overall are `PASS` for graph-pipeline enablement only. No Qwen3 export, quantization, benchmark, memory/power measurement, or Nsight profile was run. Phase 2.2 and Phase 3 remain unstarted.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_1_5_graph_enablement.md`; evidence: `experiments/Phase2-qwen3-quantization/artifacts/phase2_1_5_20260901/`.

## Phase 2.1.8 Qwen3-like Decoder Block Feasibility (2026-09-01)

- Existing isolated tools venv: NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`, CUDA `12.6`, TensorRT `10.3.0`, ONNX `1.22.0`, Orin and Nsight Compute `2024.3.1.0`; `pip check` passed. No package or system change was made.
- A synthetic FP16 Qwen3-like single decoder block uses recorded Qwen3-0.6B dimensions (1024 hidden, 16 Q heads, 8 KV heads, 128 head dimension, 3072 intermediate), RMSNorm, RoPE, GQA, causal softmax attention and SwiGLU. No checkpoint was loaded.
- Opset-17 export and ONNX checker passed (201 nodes); TensorRT parser had zero errors, FP16 engine build passed, and dynamic batch 1/2 CUDA executions were finite with expected shapes and FP16 output. PyTorch comparison is informational only; its absolute/RMSE values are recorded without a formal accuracy claim.
- TensorRT FP16 Reduce/Pow normalization-overflow and default-stream synchronization warnings remain explicit limitations. This establishes only the bounded synthetic decoder-block feasibility path, not full-model or production readiness.
- Gate A/B/C `PASS`; Gate D `SUPPORTED`; Phase 2.1.8 `PASS / BOUNDED`. Underlying Phase 2.1 INT8/INT4 result remains `INCONCLUSIVE`.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_1_8_qwen3_block_feasibility.md`; evidence: `experiments/Phase2-qwen3-quantization/artifacts/phase2_1_8_20260901/`; ONNX and engine stay Jetson-local under `/tmp/phase2_1_8_qwen3_block/`.

## Phase 2.1.9 Full Qwen3 TensorRT Architecture Audit (2026-09-02)

- Read-only audit used the frozen Qwen3-0.6B config/revision and Phase 1 manifest; no checkpoint tensor was loaded, no full ONNX was exported and no engine/inference/benchmark was run.
- Config: 28 layers, hidden 1024, 16 Q heads, 8 KV heads, head dimension 128, intermediate 3072, vocabulary 151,936, max positions 40,960, RoPE theta 1,000,000, RMSNorm epsilon 1e-6, tied embeddings. Full map: `experiments/Phase2-qwen3-quantization/docs/qwen3_full_architecture.md`.
- Theoretical serialized weight lower bounds: 2.800 GiB FP32 and 1.400 GiB FP16/BF16; tied unique loaded lower bound 1,192,099,840 bytes. KV cache is 114,688 bytes/token per batch-1 sequence in FP16/BF16, reaching 448 MiB at 4096 and 4.375 GiB at 40960. These are planning estimates, not capacity measurements.
- Native TensorRT is `PARTIAL` because the synthetic block path works but full export, persistent KV cache, prefill/decode scheduling, sampling and runtime memory ownership are unimplemented. Current TensorRT-LLM is `BLOCKED_NEEDS_RUNTIME_WORK` by the documented SM87/software-stack intersection; HF TensorRT backend remains `UNKNOWN`.
- Gate A/B/C `PASS`; Gate D `BLOCKED_NEEDS_RUNTIME_WORK`; Phase 2.1.9 `PASS / BOUNDED`. Underlying Phase 2.1 INT8/INT4 status remains `INCONCLUSIVE`.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_1_9_full_qwen3_tensorRT_architecture_audit.md`; evidence: `experiments/Phase2-qwen3-quantization/artifacts/phase2_1_9_20260901/`.

## Required Next Action

Stop after Phase 5 closeout and next-target re-selection. Owner/ChatGPT review
is required before any follow-up. Do not implement CUDA kernels, CUTLASS
optimization kernels, TensorRT Plugins, engine rebuilds, ONNX changes, tactic
forcing, runtime redesign, or the bounded `unknown_attention_matmul`
attribution study without explicit authorization.

## Do-not-repeat Work

- Do not rerun or overwrite Original Exp01 correctness/benchmark, Exp01.1 stability, or Exp01.2 profiler artifacts merely to reconstruct context.
- Do not repeat the completed sudo/NCU permission setup or modify sudoers.
- Do not claim a precise 128/256 microarchitectural cause; the final evidence-backed status is `INCONCLUSIVE`.
- Do not estimate direct DRAM throughput where NCU reports `N/A`.
- Do not rerun Phase 4-A through 4-G or relabel shared h16816 evidence as pure
  `up_proj` evidence.
- Do not treat Phase 4-D PyTorch timing as an exact cuBLASLt algorithm identity
  or directly compare it to Phase 4-E without the boundary caveat.
- Do not reinterpret Phase 4-F's 196 correlated launches as seven kernels per
  NVTX instance; the Step 3 read-only requery records one correlated launch per
  observed NVTX instance.
- Do not start Exp02, merge `main`, change the roadmap, or modify device power/clock state without explicit direction.

## Last Verified Git State

Before Phase 3-A execution, the canonical Phase 2 checkpoint was `b2083895b1199edf8a3245013912f82aec1b942b`. Phase 3-A execution started on `phase/03-runtime-bottleneck-attribution`; A1 ran at `7067d7c8d6f42c24a54c8f774c9d530a105cd309` and A2/A3 analysis was committed through `903f7c1ee86a77d9a1dccb9b52f4839fba32e034`. The final closeout commit and push/pull state must be verified from Git at session end.

## Exp02.0/Exp02.1 Update (2026-08-31)

- Branch: exp/02-reduction; starting HEAD: d3ee5725374d780fcd3ae84fd9aa57e4d238ffb1.
- Exp02 Gate A: PASS. V1-V7 correctness: 3,087/3,087 executions PASS; max absolute error 1.1754035949707031e-4; max normalized error 2.086155075380606e-8.
- Gate B: PASS. Gate C: PASS. Overall: PASS.
- Compute Sanitizer discovery: N/A because compute-sanitizer is not installed on Jetson.
- Exp01 frozen conclusions unchanged, including 128 vs 256 microarchitectural cause remains inconclusive.

## Exp02.2 Benchmark Update (2026-08-31)

- Gate A PASS; Gate B PASS; Gate C PASS; Overall PASS; READY_FOR_EXP03.
- B1/B2/B3 raw and summary artifacts saved under experiments/Exp02-reduction/benchmark/raw/.
- Final B3 candidates: V1/B512, V2-V7/B128. V5 is fastest at 1.626439 ms mean for N=16777229.
- V6 vs V7 paired delta (V6-V7) = 0.333762 ms, 95% CI [0.316896, 0.350628] ms; significant in favor of V7.
- No Exp03 started; no Exp01 content changed.

## Exp02.3 Nsight / Final Update (2026-08-31)

- Gate A PASS; Gate B PASS; Gate C PASS; Overall PASS; READY_FOR_EXP03.
- NCU 2024.3.1 first-stage common profiles V1-V7 at B256 plus V5 B64/B128/B256/B512 sweep completed. Direct DRAM throughput remains N/A where unsupported; no estimate made.
- H2 is PARTIALLY_SUPPORTED due low measured divergence despite modulo source; H3 SUPPORTED by V3/V4 bank-conflict counters; H5 PARTIALLY_SUPPORTED; H6 SUPPORTED; H7 SUPPORTED; H8 SUPPORTED; H9 INCONCLUSIVE.
- Exp03 correctness was completed without benchmark or Nsight; Exp01 remains frozen unchanged.

## Exp03.0/Exp03.1 Update (2026-08-31)

- Branch: exp/03-matrix-transpose; initialized from Exp02 closeout `8293d6a`.
- Fixed geometry: TILE_DIM=32, BLOCK_ROWS=8, block(32,8); V0 CPU, V1 copy, V2 naive, V3 tiled, V4 padded.
- Final correctness: 828/828 bitwise PASS (V1-V4 each 207/207); guard failures 0; CUDA failures 0.
- Initial V1/V2 grid_y boundary bug was fixed and the complete matrix rerun; failed raw evidence was retained on Jetson.
- Gate A PASS; Gate B/C NOT_STARTED; Overall IN_PROGRESS; READY_FOR_EXP03_BENCHMARK.
- Formal benchmark, Nsight Compute, Exp04 and performance ranking were NOT started.

## Exp03.2 Benchmark Update (2026-08-31)

- Two complete benchmark runs used six dimensions, V1-V4, warmup 20, 100 repetitions and five trials with deterministic rotation.
- CUDA Event timing and correctness sanity passed, but CV >5% persisted; Gate B is INCONCLUSIVE.
- Gate C was not started because benchmark stability is unresolved. No power mode or clocks were changed.

## Exp03.2b Stability Update (2026-08-31)

- Adaptive Diagnostic A used per-configuration calibration, >=1 s warmup, >=500 ms actual CUDA Event window, deterministic rotation, seven trials and read-only telemetry.
- All 4096^2 versions passed CV <=5%; Diagnostic B was not run. Formal Benchmark V2 retained 168 trials and all 24 configurations had CV <=0.791%.
- Gate B is PASS. Fixed short measurement window is a supported contributor; DVFS contribution remains inconclusive. Gate C is permitted but not yet started.

## Exp03.3 Nsight / Final Closeout (2026-08-31)

- NCU 2024.3.1 profiled isolated 4096x4096 V1-V4 launches with installed sections and existing `sudo -n` permission. All pre/post profiling sanity checks passed and raw CSV/TXT artifacts are committed; `.ncu-rep` files remain Jetson-local under `/tmp/jetson-qwen-exp03-ncu-20260831T121500Z/`.
- V2 global stores: 32.105 sectors/request and 14,735,372 excessive sectors versus 4.000 sectors/request and zero excessive sectors for V1/V3/V4. V3 shared conflicts: 16,349,795 total; V4: 53,953 total and zero excessive wavefronts.
- Achieved occupancy: V1 79.20%, V2 79.26%, V3 94.28%, V4 94.47%. Gate C C1/C2/C3 PASS; H1-H4 SUPPORTED. Direct DRAM throughput remains unavailable and is not estimated.
- Exp03 Overall PASS / CLOSED; READY_FOR_EXP04. Exp04 was not started.

## Exp04.0 Initial GEMM Foundation (2026-08-31)

- Branch `exp/04-gemm` was created by the host at `fab89aef074f35fac3e98d9ba82a4b08e8560841`; working tree was clean before initialization.
- Added V0 CPU FP32 reference, V1 one-thread-per-output naive FP32 CUDA kernel, correctness and adaptive benchmark scripts, and design documents under `experiments/Exp04-gemm/`.
- Jetson execution was restored: V1 built with CUDA 12.6 on Orin CC 8.7; NCU 2024.3.1.0 was confirmed. Correctness passed 13/13 with intact canaries and no CUDA failures. Adaptive V1 benchmark raw/summary artifacts were collected for four shapes with seven trials each; CV stayed below 0.51%. The 256^3 actual event window was 341 ms and is recorded as a calibration observation.
- At the initial V0/V1 checkpoint, Exp04 Gate A1 was `PASS`; final Gate B/C, V2 Shared Memory Tiling, V3 WMMA, double buffering and `cp.async` were not started. That historical checkpoint was superseded by the later Exp04 closure evidence below.
- Exp04 V2/V3 execution, formal benchmark and NCU/SASS evidence are present. The dual-reference raw artifact `experiments/Exp04-gemm/benchmark/raw/wmma_correctness_dual_reference_20260831T165518Z.csv` closes the original-FP32 precision-impact evidence gap: Track A passes implementation correctness, while Track B measures end-to-end mixed-precision numerical impact including FP16 input quantization. A1/A2/A3/B/C and Overall are `PASS / CLOSED`. V2-T16 was selected by bounded survey; SASS contains HMMA.16816.F32. Double buffering/cp.async were not required and Exp05 was not started.

## Phase 1.0 Runtime Feasibility Audit (2026-09-01)

- Started from clean, synchronized Exp04 closeout `9ede5e03773d23194f06059c339f32d539f7b7be`; branch `phase/01-qwen3-baseline`.
- Jetson: Orin Nano Super, aarch64, Ubuntu 22.04.5, L4T 36.4.3 / JetPack 6.2 mapping, 7.4 GiB RAM, CUDA 12.6, TensorRT 10.3, NVIDIA PyTorch 2.5.0a0 with CUDA and BF16 available.
- Transformers, TensorRT-LLM, Docker, NVIDIA Container CLI and model assets are absent. No package/model/runtime was installed or downloaded.
- Latest TensorRT-LLM natively supports Qwen3 but does not list SM87 in the official tested hardware matrix and requires a newer CUDA/TensorRT/PyTorch stack. `v0.12.0-jetson` targets SM87/CUDA 12.6 but supports Qwen through Qwen2, not Qwen3.
- Gate P1.0: `PASS WITH CONSTRAINTS`. Recommended Phase 1.1 path is an isolated Hugging Face Transformers BF16 reference baseline using the existing NVIDIA Jetson PyTorch stack. Phase 1.1, Phase 2 and Exp05 are not started.
- Report: `experiments/Phase1-qwen3-baseline/docs/phase1_0_runtime_feasibility_audit.md`.
- No Qwen3 checkpoint load, full ONNX export, TensorRT execution/engine, benchmark, INT8, INT4 or TensorRT-LLM work occurred. Runtime execution remains unstarted and requires explicit authorization.

## Phase 2.2-A Single-Layer TensorRT Runtime (2026-09-02)

- Bounded synthetic Qwen3-like single-layer FP16 prefill/decode ONNX graphs parsed and built with TensorRT 10.3; serialized engines remain Jetson-local under `/tmp` and are not tracked.
- CUDA execution passed for prefill `B=1,S=8` and dynamic decode cache lengths 8->9->10->11->12. Outputs were finite, shape-correct and `cuda:0` resident. Independent reference/TRT cache chains show zero prefix mutation in all decode steps.
- Runtime binds CUDA `data_ptr()` addresses and uses `torch.cuda.current_stream()` with explicit synchronization. TensorRT default-stream and FP16 Reduce/Pow normalization warnings remain limitations.
- Phase 2.2-A is `PARTIAL / BOUNDED PASS` (C1-C5 bounded PASS). No production accuracy, capacity, performance or full-model readiness claim is made; Phase 2.2-B remains not started.

## Phase 2.2-B4.1 Oracle Memory Recovery (2026-09-02)

- The bounded Phase 1 regression probe loaded the exact BF16 model and completed one short forward; current intrinsic full-model loading remains viable.
- Static audit confirmed 880,932,864 bytes of all-layer CPU BF16 state plus overlapping CUDA copies, retained reference trees and giant handoff construction in the failed B4 lifecycle.
- Streaming extraction produced 28 independently hashed Jetson-local layer files totaling 881,044,080 bytes without an all-layer Python state dictionary.
- Fresh-process 4-layer and 8-layer diagnostics showed fixed allocator reservation and expected KV-only growth. The single recovered 28-layer attempt passed `S=8` prefill and one decode `8->9` with no exit 137.
- Phase 2.2-B4.1 is `PASS / CLOSED`; decision `B4_ORACLE_MEMORY_PATH_RECOVERED`; root cause `IMPLEMENTATION_MEMORY_LIFETIME_CONFIRMED`. No ONNX/TensorRT, benchmark, profiler or quantization work was run.

## Phase 2.2-B4.2 Real 28-Layer TensorRT Decoder (2026-09-02)

- Frozen model identity and 28/28 B4.1 streaming handoff files passed hash validation; raw FP16 decoder tensors total 880,932,864 bytes.
- The primary one-stack architecture succeeded: checker-valid prefill/decode ONNX graphs (881,567,403/881,548,681 bytes) parsed with zero errors and built as TensorRT 10.3 FP16 engines (892,483,860/890,363,012 bytes). The 7x4 fallback was not used.
- B=1,S=8 prefill and decode 8->9->10->11->12 passed for all 28 layers. Hidden/K/V tensors were finite and CUDA resident; all K/V prefixes were bitwise invariant and all 28 layer caches remained pointer-isolated.
- Selected Layer 0/3/7/15/27 numerical propagation passed the predeclared relative-L2 <=0.10 and cosine >=0.99 engineering bound; decision `ACCEPTABLE_FOR_FULL_FP16_RUNTIME_STEP`. This is not a full-model quality claim.
- B4.2-1..B4.2-8 `PASS`; Overall `PASS / BOUNDED`; decision `REAL_28_LAYER_TRT_DECODER_STACK_FEASIBLE`. TensorRT normalization/default-stream warnings remain. No benchmark, Nsight, quantization, full token generation or later phase was started.

## Phase 2.2-C0 Runtime Architecture Audit (2026-09-02)

- Added `docs/phase2_2c_runtime_architecture.md`, documenting the full text-to-token pipeline and the boundary between the verified 28-layer decoder runtime and the not-yet-implemented full generation runtime.
- C0 records the planned C1 Embedding, C2 Final RMSNorm, C3 LM Head, C4 Greedy Sampling and C5 End-to-End Token Agreement stages with validation and memory-ownership constraints.
- C0 is `PASS / DESIGN ONLY`; no code, weights, engine, benchmark, profiler, quantization or generation loop was executed or modified. C1-C5 remain `NOT STARTED`.

## Phase 2.2-C1 Embedding Integration (2026-09-02)

- Real `model.embed_tokens.weight` (`[151936,1024]`, BF16) audit, one-node Gather ONNX export, TensorRT 10.3 FP16 build and embedding-only numerical validation passed. Evidence: `phase2_2c1_20260902T090000Z`.
- Passing embedding output was handed once to the unchanged B4.2 28-layer prefill engine. Outputs were finite and shape-correct, but final Layer 27 hidden comparison failed: relative-L2 `2.00425`, cosine `0.534268`.
- C1 is `BLOCKED`; likely next step is a narrow pre-decoder embedding-output comparison. No B4.2 code/engine, benchmark, profiler, quantization or C2 work was modified or started.

## Phase 2.2-C1D Decoder Boundary Diagnostic (2026-09-02)

- D0 reproduced the current B4.2 random-hidden control (`relative-L2 0.02014`, cosine `0.999797`, PASS).
- D1 proved the canonical FP16 embedding input byte-identical between portable and TensorRT (`sha256 da04f533...`, zero error). D2 still diverged progressively through the decoder to Layer 27 (`relative-L2 2.004247`, cosine `0.534268`).
- D3 host-staged and direct-device paths failed identically. Both runtimes used stream pointer 0 with explicit synchronization; the prefill engine exposes only `hidden_states` FP16 and `position_ids` INT64 inputs. No pointer, stream, lifetime, binding, or auxiliary-input defect was confirmed.
- C1 remains `BLOCKED`; no fix, engine rebuild, B4.2 overwrite, OOM or exit 137 occurred. Report: `phase2_2c1_decoder_boundary_diagnostic.md`; raw evidence: `phase2_2c1d_20260902T180000Z_c1d_diagnostic.json`.

## Phase 2.2-C1E Layerwise Divergence Localization (2026-09-02)

- Reused the existing B4.2 prefill engine, C1 canonical embedding reference, and 28-file handoff; no engine rebuild or decoder/runtime modification occurred.
- All exposed `hidden_l0`..`hidden_l27` outputs are `[1,8,1024]` FP16 and finite. D0 reproduced PASS (`relative-L2 0.0201395`, cosine `0.9997966`).
- First non-zero difference is Layer 0 (`max_abs 0.01171875`, relative-L2 `0.00378768`, cosine `0.9999921`). First relative-L2 > 0.10 is Layer 21; first cosine < 0.99 is Layer 22; Layer 27 reaches relative-L2 `2.004247`, cosine `0.5342684`.
- Component attribution is `NOT PERFORMED` because the engine exposes no RMSNorm/attention/MLP intermediates. C1 remains `BLOCKED`; C2 must not start. No OOM, exit 137, benchmark, profiler or quantization work occurred.
- Report: `phase2_2c1_layerwise_divergence_report.md`; raw evidence: `phase2_2c1e_layerwise_20260902T190000Z.json`.

## Phase 2.2-C1F Layer 0 Component Attribution (2026-09-02)

- Starting HEAD was `6e48add7acdb40a64f13920efd7bb27b615ec3bb`; the existing B4.2 prefill engine and canonical C1 input were reused without rebuild or runtime modification.
- The actual Qwen3 Layer 0 structure was recorded: input/post-attention RMSNorm, 16Q/8KV GQA, per-head Q/K normalization, rotary dimension 128 with theta 1,000,000, output projection, residuals and SwiGLU MLP.
- B4.2 exposes 84 outputs (hidden_l0..hidden_l27, present_k0..present_k27, present_v0..present_v27) and no internal component bindings. All requested component metrics are therefore `NOT_AVAILABLE`; first divergent operator is `UNKNOWN`.
- Existing B3 `attention_l0` partial evidence measured max abs `0.01171875`, relative-L2 `0.00316544`, cosine `0.99999416`; this is not 28-layer operator attribution. C1 remains `BLOCKED`; C2 must not start.
- Result: `COMPONENT_LOCALIZATION_BLOCKED`. Raw evidence: `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/artifacts/phase2_2c1f_layer0_component_20260902T193000Z.json`; report: `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/docs/phase2_2c1_layer0_component_attribution.md`.

## Phase 2.2-C1G Layer 0 Internal Tensor Probe (2026-09-02)

- Starting HEAD was `b8c9bc8edbb123e989a8a734eac6627077608336`. Three new independent Group A/B/C Layer 0 FP16 TensorRT diagnostic engines were built; B4.2 remained read-only and was not rebuilt.
- All probe validity checks passed against original B4.2 `hidden_l0`; Group A/C were exact and Group B was relative-L2 `0.00250531`, cosine `0.99999619`. B4.2 control remained relative-L2 `0.00378768`, cosine `0.99999213`.
- First non-zero component difference is input RMSNorm (relative-L2 `0.000491762`). QK score difference is `1.95e-7`, then softmax rises to `0.003432333`; this is the first material amplification, reported as `FIRST_MATERIAL_OPERATOR_FOUND` with status `NARROWED / NOT CONFIRMED`.
- MLP gate×up and down reach relative-L2 `0.005109561` and `0.005827574`, but are downstream amplification. No precision A/B or fix was performed. C1 remains `BLOCKED`; C2 must not start.
- Report: `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/docs/phase2_2c1_layer0_internal_probe.md`; raw JSON: `c1g_layer0_group_a/b/c_20260902T200000Z.json`.

## Phase 2.2-C1H Attention Softmax Semantics (2026-09-02)

- Starting HEAD was `64d8d5eefb832a51edf22e119001d0cfb709010a`. A new diagnostic-only script exposed raw/scaled/masked QK scores, same-input native and explicit FP32 TensorRT softmax micro engines, and an independent Layer 0 FP32-softmax-only A/B.
- Final masked pre-softmax inputs were nearly identical (relative-L2 `1.9496e-7`), with FP16 causal sentinel `-65504.0` and no infinities. Native and explicit FP32 TensorRT softmax both matched the portable output exactly and had oracle relative-L2 `0.000154608`.
- Layer 0 explicit FP32-softmax-only final relative-L2 was `0.003787680`, identical to unchanged B4.2 control. Result: `SOFTMAX_HYPOTHESIS_REJECTED`; C1 remains `BLOCKED`, no repair/C1I/C2 started.
- Minimum `MemAvailable` was `2,617,298,944` bytes, maximum RSS `3,623,332` KiB, with no OOM/exit 137. Report: `phase2_2c1_softmax_semantics.md`; raw JSON: `artifacts/c1h_20260902T210000Z/c1h_softmax_semantics_20260902T210000Z.json`.

## Phase 2.2-C1I Q/K and RoPE Numerics (2026-09-02)

- Added the diagnostic-only `c1i_qk_rope_numerics.py` and ran fresh Layer 0 probe/micro engines against the unchanged B4.2 control and canonical embedding input.
- Input RMSNorm, Q projection, K projection, Q norm and K norm remained below `0.002` relative-L2. Same-input Q RoPE reached `0.092710741` and K RoPE `0.014421386`; full probe QK raw reached `0.043750536`.
- Result: `ROPE_MAJOR_SOURCE_CONFIRMED` as the major upstream source in this diagnostic. Exact cast/cache mechanism is not yet isolated; C1 remains `BLOCKED`, C1J/C2 must not start.
- Raw evidence: `phase2_2_runtime_prototype/artifacts/c1i_20260902T223000Z.json`; report: `phase2_2c1_qk_rope_numerics.md`. Valid FP32 A/B variants all remained at final relative-L2 `0.034715209`; no OOM/exit 137 or B4.2 regression occurred.

## Phase 2.2-C1J Qwen3 Layer 0 RoPE Numerics (2026-09-03)

- Starting HEAD was `b8a9d828257d4a8a0bfd0cb5805f089e34b71095`; the new diagnostic used only fresh timestamped engines and loaded B4.2 read-only.
- Native TensorRT cache differs from the portable FP16 cache (Q cos/sin relative-L2 `0.0978644267/0.3146529794`), while the explicit FP32-cache variant matches cache values exactly. Positions `0..7`, half-split layout, rotary dimension `128`, and GQA repeat `2` are confirmed; even/odd is a negative control.
- Result: `ROPE_PRECISION_SEMANTICS_CONFIRMED`; C1 remains `BLOCKED`, C1K/C2 must not start without authorization. B4.2 control remains `0.0037876803`; no OOM/exit 137.
- Raw evidence: `phase2_2_runtime_prototype/artifacts/c1j_20260902T230000Z/`; report: `phase2_2c1_rope_numerics.md`.

## Phase 2.2-C1K RoPE Cache Precision Validation (2026-09-03)

- Starting HEAD was `4967feb895a5d2675db2571dadedcca0a8d32ff9`; only a new diagnostic script and timestamped evidence/report were added.
- Fixed portable FP16 `cos/sin` cache initializers reduced same-input Q/K RoPE relative-L2 from `0.0927107111/0.0144213885` to `0.0001572046/0.0000268356`.
- Layer 0 final relative-L2 improved from `0.0347152092` to `0.0038153231` (`9.0989x` reduction); corrected-vs-unchanged-B4.2 control is `0.0018616138`.
- Result is `ROPE_CACHE_FIX_VALIDATED`. C1 remains `BLOCKED`; no repair, C1L, C2, benchmark, Nsight, quantization or production rebuild started. No OOM/exit 137.
- Raw evidence: `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/artifacts/c1k_20260902T163005Z/c1k_rope_cache_precision_20260902T163005Z.json`; report: `phase2_2c1_rope_cache_precision_validation.md`.

## Phase 2.2-C3 LM Head Integration (2026-09-03)

- Starting HEAD was `92e8857a25c974c61ac18901424fabefaba319d4` on
  `phase/02-qwen3-quantization`. C2 final RMSNorm remains `PASS / BOUNDED`.
- The pinned `lm_head.weight` is `[151936,1024]`, BF16, bias-free, and exactly
  equal to `model.embed_tokens.weight` under tied embeddings. Independent
  TensorRT 10.3 FP16 MatMul validation passed synthetic prefill/decode and
  last-token paths with relative-L2 `0.00169018/0.00164510`, all 9 argmax
  results equal, and full top-5 overlap.
- Read-only embedding -> corrected decoder -> C2 norm -> LM Head produced
  finite `[1,8,151936]` logits. Relative-L2 `0.04533300` is explicitly
  `END_TO_END_DIAGNOSTIC_ONLY` because C1 decoder drift remains unresolved.
  No OOM or exit 137 occurred. C3 is `PASS / BOUNDED`; C4 and generation
  work were not started.
- Report: `experiments/Phase2-qwen3-quantization/phase2_2_runtime_prototype/docs/phase2_2c3_lm_head.md`; raw evidence: `artifacts/phase2_2c3_20260903T/`.

## Phase 2.2-C4 Greedy Sampling Integration (2026-09-03)

- Starting HEAD was `3a1bf6cf59be27dd0e07e415b2fe5eaeb009852c` on
  `phase/02-qwen3-quantization`. Added only the C4 sampler, report and raw
  JSON under `artifacts/phase2_2c4_20260903T/`.
- `SAMPLER_BACKEND = CPU_NUMPY_ARGMAX`; PyTorch reference and NumPy sampler
  agree exactly on clear-winner, near-tie and exact-tie logits. C3 decode
  token `2629` and last-token `57133` also agree exactly.
- Read-only `input_ids [0..7] -> embedding -> 28-layer decoder -> Final
  RMSNorm -> LM Head -> greedy` produced valid token `42` on both portable and
  TensorRT paths. Top1-top2 margin was `0.421875` on both. Full path remains
  `END_TO_END_DIAGNOSTIC_ONLY` due to C1 drift; no OOM or exit 137 occurred.
- C4 is `PASS / BOUNDED`. C5 autoregressive generation, tokenizer loop and
  sampling policy work were not started.

## Phase 2.2-C5 checkpoint (2026-09-03)

- Starting HEAD was `2bd7925ff0be23919f1ec06440941228e0dec01e`. Added only the
  orchestration script, C5 report, Phase 2.2 closeout and timestamped raw
  traces under `artifacts/phase2_2c5_20260903T/`.
- Pinned Qwen3 tokenizer prompt `Hello` -> input IDs `[9707]`. HF greedy
  reference generated `[21806, 0, 358, 2776]`; the TensorRT runtime generated
  `[0, 46309, 46309, 46309]`, diverging at step 0. This is recorded as
  `FULL_RUNTIME_TOKEN_MISMATCH_DUE_TO_UPSTREAM_NUMERICAL_LIMITATION`; C1 was
  not reopened.
- TensorRT embedding -> 28-layer prefill/decode -> Final RMSNorm -> LM Head
  completed all 3 decode steps with valid tokens, finite outputs, exact K/V
  prefixes, cache lengths `1 -> 2 -> 3 -> 4`, and 28-way pointer isolation.
  Pinned tokenizer decode produced `Hello!geoisgeoisgeois`. No OOM/exit 137.
- C5 and aggregate Phase 2.2 are `PASS / BOUNDED`. Phase 2.3 was not started.

## Phase 2.3-A Explicit INT8 Q/DQ Feasibility (2026-09-03)

- Selected real Qwen3 `model.layers.0.self_attn.q_proj` from the B2 Layer 0 handoff; frozen model revision and checkpoint hash were verified.
- One shared real `[1,8,1024]` FP16 canonical input was used for FP16, W8-QDQ and W8A8-QDQ. Weight quantization is symmetric per-tensor INT8; activation scale is a feasibility-only absmax scale, not final calibration.
- Minimal Q/DQ sanity and all three target graphs passed ONNX checker, TensorRT 10.3 parser/build, CUDA execution and finite-output checks. Non-zero zero-point parsing is unsupported; per-channel capability probes passed parser/build for tested axes.
- W8-QDQ is `QDQ_EXECUTION_PROVEN` with `INT8_COMPUTE_NOT_PROVEN`; W8A8-QDQ EngineInspector exposes Int8/Int8 fusion and an `i8` GEMM tactic, classified `INT8_COMPUTE_PROVEN` for this graph/profile.
- Component deltas versus TRT FP16 are W8 relative-L2 `0.04692697` and W8A8 `0.04787614`; no final accuracy threshold, calibration sweep, performance benchmark, Nsight, full 28-layer quantized runtime or INT4 work was run.
- Gate: `PASS`. Phase 2.3 overall is `IN PROGRESS`; Phase 2.3-B is the next authorized boundary.

## Phase 2.3-B Calibration / Activation Range Audit (2026-09-03)

- Exact provenance was proven for 36 real tokenizer samples: the captured tensor is post-`model.layers.0.input_layernorm` and pre-`model.layers.0.self_attn.q_proj`, with hook/direct norm equality for every sample.
- The deterministic corpus contains 24 calibration and 12 disjoint evaluation prompts across four length groups and English, Chinese, numerical, structured, code-like and mixed categories. No random formal input was used.
- Calibration candidates were GLOBAL_ABSMAX, P99.9, P99.99 and a bounded MSE clip grid. Held-out primary `W8A8 vs W8` activation-only relative-L2 median/P95/max were `0.019526/0.020111/0.020117` for selected `BOUNDED_MSE_CLIP`; total `W8A8 vs FP16` median/P95/max were `0.033015/0.033980/0.034272`.
- Selected-policy detailed EngineInspector showed Int8 activation, Int8 weight and an `sm80_xmma_gemm_i8i8...` tactic; `INT8_COMPUTE_PROVEN` is limited to this graph/profile. Frozen W8-vs-FP16 median/P95 relative-L2 `0.026780/0.027547` exceeded selected A8-only median/P95 `0.019526/0.020111`, supporting weight-quantization dominance for this target/corpus.
- Gate: `Phase 2.3-B = PASS`. Phase 2.3 remains `IN PROGRESS`; Phase 2.3-C was not started. No full-model calibration, 28-layer INT8, benchmark, Nsight, INT4, mixed precision or C1 reopening occurred.

## Phase 2.3-E 28-Layer Mixed-Precision Quantized Runtime (2026-09-04)

- Starting Windows HEAD was `77ca1d004abdc3d8a50ab8ecc0232d9ce82e4ed7` on
  `phase/02-qwen3-quantization`; Jetson execution used the historical
  `a1317a06f83634406bfb732a61f57a698e6aee2d` checkout and preserved its
  pre-existing untracked diagnostics.
- The frozen Phase 2.3-D `P2_FAMILY_GUARD_REFINED` policy (63 FP16 + 133
  PT-W8A8) was applied to the existing B4.2 FP16 prefill/decode ONNX graphs by
  injecting explicit weight/activation Q/DQ for every PT-W8A8 MatMul and
  preserving FP16 targets verbatim. All 133 required activation scales were
  present (139 scale entries, 6 refined-to-FP16 entries unused).
- 28/28 mixed prefill and decode engines built. EngineInspector shows 133
  `sm80_xmma_gemm_i8...` INT8 tensor-core GEMM tactics per engine, matching
  the 133 PT-W8A8 targets (`INT8_COMPUTE_PROVEN` for this deployable path).
- Full runtime (embedding -> 28-layer mixed decoder -> Final RMSNorm -> LM
  Head -> greedy) passed B=1,S=10 prefill, 4-step decode, KV cache invariants
  (length 1->5, exact prefixes, 28-way pointer isolation), Final RMSNorm/LM
  Head, and a 4-step `Hello` generation. No OOM/exit 137.
- Same-prefix mixed-vs-FP16: prefill last hidden relative-L2 `0.008568` and
  cosine `0.9999999`; forced-decode hidden relative-L2 `0.542..0.603` and
  logits relative-L2 `0.324..0.371`; top-1 agreement `0/4`, top-5 overlap
  `1..2/5`. Mixed and FP16 `Hello` tokens diverge after the shared degenerate
  first token.
- Gate: `Phase 2.3-E = PASS / BOUNDED`; runtime classification
  `PRIMARY_POLICY_RUNTIME` (no fallback). Mixed engines are ~27% smaller than
  FP16 (storage context only). C1 remains `CLOSED /
  NUMERICAL_LIMITATION_UNRESOLVED` and was not re-opened.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_3e_28layer_mixed_precision_runtime.md`;
  evidence: `experiments/Phase2-qwen3-quantization/artifacts/phase2_3e_20260904T034300Z/`;
  mixed ONNX/engines remain Jetson-local under `/tmp/phase2_3e_20260904T020000Z/`.

## Phase 2.3-F Accuracy / Memory / Performance Comparison (2026-09-04)

- Reused the 12 disjoint Phase 2.3-B evaluation prompts (truncated to the
  first 8 tokens to stay within the B4.2-derived engine 16-token context
  limit). Comparison is `TRT MIXED vs TRT FP16`; HF is semantic context only.
- Prefill last-token logits mixed-vs-FP16: relative-L2 median `0.3311`, cosine
  median `0.9086`, top-1 agreement `4/12`, top-5 overlap `2.92/5`.
- Same-prefix forced decode (96 steps): logits cosine median `0.8879`, top-1
  agreement `47.9%`, top-5 overlap `2.26/5`; some reference logits degenerate
  to zero from the documented C1 drift.
- Free-run greedy trajectories diverge at step 0 for 7/12 prompts; token
  mismatch is treated as a behavioral result, not a runtime failure.
- Same-session benchmark (warmup 5, repeats 10): mixed is ~48% slower prefill
  and ~35% slower decode than FP16 (tokens/s `0.51 -> 0.38`). S=32/128 are
  blocked by the 16-token engine profile, not memory.
- Engine storage is 27% smaller (892.5/890.4 MB FP16 -> 651.7/650.3 MB mixed).
  No OOM/exit137; peak RSS ~1.74 GB. Conclusion: `MIXED_RUNTIME_SLOWER`.
- Gate: `Phase 2.3-F = PASS / BOUNDED`; Phase 2.3 aggregate is
  `CLOSED / PASS / BOUNDED`. C1 remains `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`.
- Report: `experiments/Phase2-qwen3-quantization/docs/phase2_3f_accuracy_memory_performance_comparison.md`;
  evidence: `experiments/Phase2-qwen3-quantization/artifacts/phase2_3f_20260904T050000Z/`.

## Phase 3-A Runtime Bottleneck Attribution (2026-09-04)

- Starting canonical checkpoint was `b2083895b1199edf8a3245013912f82aec1b942b`;
  the new branch is `phase/03-runtime-bottleneck-attribution`. A1 evidence was
  produced at `7067d7c`, the Mixed profile-name fix at `2451800`, and A2/A3
  analysis at `903f7c1`. No Phase 3-B optimization was implemented.
- Controlled A1 benchmark (warmup 5, repeats 10, window repeats 3, 8-step
  window, batch 1, deterministic first evaluation prompt): FP16 prefill
  1517.606/1521.247 ms versus Mixed 2263.352/2265.980 ms at S=8/S=16 (+49.14%/
  +48.96%); FP16 TPOT 1959.902/1963.507 ms versus Mixed 2665.612/2664.299 ms
  (+36.01%/+35.69%). `MIXED_RUNTIME_SLOWER` reproduced.
- NSYS S=8 prefill and decode-step ranges showed GPU kernel time of 47.2/48.9
  ms FP16 versus 42.7/43.0 ms Mixed, only 3.00/2.29% and 1.85/1.46% of wall.
  Mixed CUDA API time was 907.2/975.4 ms versus FP16 357.1/295.3 ms.
- Mixed wall increases were +740.4 ms prefill and +813.4 ms decode. CUDA API
  increases were +550.1/+680.1 ms; `cuModuleLoadData+cuModuleUnload` accounted
  for +565.3/+693.8 ms. Sync and DtoH copy time were minor.
- The reused Phase 2.3-F runtime creates a TensorRT execution context per
  `TRT.run`; this mechanism is `SUPPORTED` as the module-load storm source, but
  not proven by a cached-context A/B. High-level mapping of `__myl_*` kernels
  remains `UNKNOWN`.
- Root cause ranking: P1 runtime scheduling/per-call execution-context
  creation and module load/unload; P2 mixed-engine structural/host complexity;
  P3 minor DtoH and extra kernel effects. `Q/DQ as the direct dominant GPU
  cost` is `DISPROVEN` and INT8 GEMM was not slower in this shape.
- Gate: `PASS / BOUNDED`. Recommended Phase 3-B target is persistent TensorRT
  execution contexts, followed by a post-fix runtime scheduling profile.
- Report: `docs/phase3a_runtime_bottleneck_attribution.md`; evidence:
  `results/phase3a_runtime_attribution/20260904T062712Z/` and
  `results/phase3a_runtime_attribution/20260904T063649Z_nsys/`. Raw NSYS
  reports remain Jetson-local under `/tmp/phase3a_nsys_20260904T063649Z/`.

## Phase 3-B Runtime Object Lifetime Optimization (2026-09-04)

- Starting canonical checkpoint was `87ad72f1d87150a720c6e5316620ed9fd8767001`;
  the new branch is `phase/03b-runtime-object-lifetime`. B0 was committed at
  `40eb2ee`, the persistent-context implementation at `9b61f4e`, and B2/B3/B4
  evidence records HEAD `1e4d8888b651b550f2359f6e8530553ee126e008`.
- B0 proved that Runtime, engines and stream were already persistent, while
  `TRT.run` created a function-local ExecutionContext on every call. Legacy
  mode made 2 contexts for a decoder-only call and 4 for a decoder+norm+LM
  Head call; the Phase 3-A profile sequence made 28 contexts.
- B1 added the `legacy_context_lifetime` / `persistent_context_lifetime`
  feature flag. Persistent mode creates one context per engine wrapper, then
  continues to set every input shape/address and output address per call. No
  temporary tensor pointer is cached and no scheduler/pool was added.
- B2 Mixed legacy-vs-persistent was exactly equal and finite for S=8 plus 8
  decode steps and S=16 prefill. All KV prefixes, lengths and 28-layer K/V
  pointer isolation passed. Legacy S=8 made 36 contexts; persistent made 5 and
  reused 36. No OOM/exit137 occurred.
- B3 same-session benchmark (warmup 5, repeats 10, 8-step windows): Mixed
  prefill improved from 2259.919/2255.619 ms to 47.627/42.406 ms at S=8/S=16;
  Mixed TPOT improved from 2662.990/2662.266 ms to 43.362/43.580 ms. Mixed is
  now roughly equal to FP16. Gap recovery was prefill 100.348%/99.721% and
  decode 100.009%/99.780%; values above 100% reflect small negative gaps and
  are noise-bounded.
- B4 steady-state NSYS showed Mixed module load/unload collapse from 832 calls
  at prefill and 566 calls at decode step 0 to zero; wall fell from 2310.857 to
  33.217 ms and 2881.660 to 65.848 ms. CUDA API time fell from 896.733 to
  18.957 ms and 926.188 to 22.452 ms. B2 exact equality prevents interpreting
  differing kernel totals as a semantic change.
- B5 initialization increased from 3.349467 to 6.779457 s for FP16 and
  2.636420 to 6.655416 s for Mixed. Persistent mode created 5 contexts, one
  per engine wrapper. Observational loaded `MemAvailable` decreased by
  388,288,512 B for FP16 and 245,555,200 B for Mixed; exact retained-context
  cost is `INCONCLUSIVE`. No deployment memory blocker was observed.
- Gate: `PASS / CLOSED / PROVEN`. H3B-1 is `PROVEN` for this single-request
  runtime and bounded workload. Phase 3-C was not started.
- Report: `docs/phase3b_runtime_object_lifetime.md`; primary evidence:
  `results/phase3b_runtime_lifetime/b0_lifecycle_audit.json`,
  `results/phase3b_runtime_lifetime/20260904T081621Z/`,
  `results/phase3b_runtime_lifetime/20260904T081727Z/`, and
  `results/phase3b_runtime_lifetime/20260904T082912Z_nsys/`. Raw NSYS remains
  Jetson-local under `/tmp/phase3b_nsys_20260904T082912Z/`.

## Phase 3-C Persistent Runtime Residual Profiling (2026-09-04)

- Starting checkpoint was `aa2545edf6670ef3e0e711735da24393f3aa5b13`; the new
  branch is `phase/03c-residual-runtime-profiling`. The C0/C1 preparation
  commit is `51f41c2`. C1 and C2/C3 evidence used Jetson HEAD `51f41c2`.
- C1 tested only persistent FP16 versus persistent Mixed. Both runtimes used
  5 context creations and 448 reuses; minimum `MemAvailable` was
  `3,191,648,256` bytes with no OOM or exit 137. The two runtimes were near
  parity; the S=8 FP16 prefill ordering is noise, not a semantic result.
- C2 collected separate FP16 and Mixed NSYS traces for one representative
  prefill and four decode steps. Steady-state module load/unload was `0/0` in
  both runtimes. Aggregate kernel time was 219.710 ms FP16 and 147.831 ms
  Mixed; aggregate CUDA API time was 186.374 ms FP16 and 116.749 ms Mixed.
- C3 name-based attribution found GEMM dominant (202.919 ms FP16 and
  133.532 ms Mixed), but the largest single kernel was only 35.31% (FP16) and
  26.78% (Mixed) of kernel time. No single kernel exceeded 50%.
- Decision gate is `Case C`: Mixed/FP16 are already near parity and no
  justified CUDA operator target was found. `NSIGHT COMPUTE NOT REQUIRED`.
  Myelin/internal kernels remain `UNKNOWN / TRT INTERNAL`; no operator identity
  was inferred from names.
- Gate: `PASS / BOUNDED`. Phase 3-D was not started.
- Report: `docs/phase3c_residual_runtime_profiling.md`; primary evidence:
  `results/phase3c_residual_runtime/20260904T090803Z_bench/` and
  `results/phase3c_residual_runtime/20260904T093500Z_nsys/`. Raw NSYS remains
  Jetson-local under `/tmp/phase3c_nsys_20260904T093500Z/`.

## Phase 3-D0 CUDA Graph Feasibility (2026-09-04)

- Starting checkpoint was `93e799fb2002f7b9884a9e3867ce93faf23cc173`; the
  branch is `phase/03d0-cuda-graph-feasibility`. D0-A passed the compatibility
  audit, identifying fixed decode-window addresses and excluding CPU sampling.
- Full-window PyTorch CUDA Graph capture completed for 1-step and 2-step
  smoke runs, but validation was `BLOCKED`: hidden/logits and KV differed from
  persistent stream despite fixed-address topology passing.
- A per-engine graph mode captured 4 graphs for 1 step, but validation also
  remained `BLOCKED` with the same failed outputs. Formal benchmark status is
  `BLOCKED_NO_VALID_GRAPH_PATH`; the faster graph wall time is invalid.
- Separate 8-step NSYS traces showed persistent stream with 3,376 kernels and
  217.708 ms kernel time, while per-engine graph replay had 64 kernels totaling
  0.298 ms, all `FillFunctor<long>`, and no TensorRT kernels.
- Final Gate is `BLOCKED / BOUNDED`. The current PyTorch capture path is
  `DISPROVEN`; a redesigned runtime or native TensorRT graph path remains
  `UNKNOWN`. Phase 3-D1 was not started.
- Report: `docs/phase3d0_cuda_graph_feasibility.md`; primary evidence:
  `results/phase3d0_cuda_graph/20260904T094952Z/`. Raw NSYS remains Jetson-local
  and hashed in `results/phase3d0_cuda_graph/20260904T094952Z/nsys/`.

## Phase 3-E TensorRT Kernel Attribution (2026-09-04)

- Starting checkpoint was `8b8a3530d9eed1e221419e41d24c9a4d8071cb02`; the
  branch is `phase/03e-tensorrt-kernel-attribution`. Code commits are `8e34c85`,
  `7b20282`, `b6ce658`, `b39bfec` and `ab65133`; the closeout commit must be
  verified with Git.
- EngineInspector audited four decoder engines without rebuilding them. FP16
  engines expose compact layers: 84 GEMM layers plus 336/529 UNKNOWN layers.
  Detailed Mixed metadata maps q/k/v/o/gate/up/down projections; 28 prefill
  attention-GEMM layers and 194 prefill / 250 decode GEMM layers are recorded.
- Joining Inspector tactics with Phase 3-C NSYS triggered NCU: FP16 rank-2
  kernel 35.31%, rank-1 h16816 20.28% FP16 and 26.78% Mixed.
- NCU used narrow kernel filters, one launch per target and
  `--clock-control none`. A prefill-only rank-2 attempt found no matching
  kernel and is retained; rank-2b used one decode step. Raw `.ncu-rep` files
  remain Jetson-local.
- Results: rank-1 h16816 97.06% memory/L2, 39.57% SM, 39.673148% HMMA active,
  24.78% achieved occupancy. Rank-2/3 decode GEMMs: 71.92-76.88% memory/L2,
  25.62-29.60% SM, 17.167166-17.697509% HMMA active, and achieved occupancy
  24.25-24.68%. Direct DRAM remains `N/A`.
- Final Gate is `PASS / BOUNDED / NO_PROVEN_CUDA_OPTIMIZATION_TARGET`. The
  low HMMA signal on ranks 2/3 is potential, not proof; compact FP16 metadata
  leaves operator/tactic attribution UNKNOWN. Phase 4 was not started.
- Report: `docs/phase3e_kernel_attribution.md`; primary evidence:
  `results/phase3e_kernel_attribution/20260904T121007Z/`.

## Phase 4 Evidence Checkpoint (2026-09-05)

- Branch is `phase/04a-tensorrt-operator-attribution-recovery`; the checkpoint
  is based on Phase 3-E HEAD `bf7abc67eb58662a68316045e166aa9f611330d7`.
  Verify the checkpoint commit with Git.
- Phase 4-A established 250 GEMM candidates and the TensorRT Layer -> ONNX
  Node -> Transformer Operator projection mapping. Phase 4-B selected
  decode-only `up_proj` as the first optimization-research target with score
  `9.0`.
- Phase 4-C froze `C[1,3072] = A[1,1024] * B[1024,3072]`, FP16 operands/output,
  28 decoder layers, correctness oracle, benchmark protocol, and success
  criteria. Phase 4-D added a standalone cuBLAS-backed `torch.matmul`
  reference: median `0.086364701 ms`, CV `0.0454%`; exact backend kernel
  remains `UNKNOWN`.
- Phase 4-E timed `28/28` existing-engine `up_proj` layers with TensorRT
  `IProfiler`; per-layer medians were `153.28-159.76 us` and the 28-layer
  aggregate was mean `4682.253873 us` / median `4339.215986 us`. Phase 4-F
  correlated `28/28` observed NVTX ranges to `28/28` layers and `196` kernel
  launches: h16816 family `84/196`, `sm80_xmma_gemm_*` `112/196`.
- Phase 4-G found no proven optimization hypothesis. H1, reduce or reshape
  memory/L2 traffic in the shared h16816 path, is only
  `MEDIUM_FOR_BOTTLENECK_DIRECTION`; CUDA readiness is `NOT READY`.
- Reports and raw artifacts are under `results/phase4a_operator_attribution/`,
  `results/phase4b_target_selection/`, `results/phase4c_up_proj_baseline/`,
  `results/phase4d_up_proj_gemm_baseline/`,
  `results/phase4e_up_proj_baseline/`, `results/phase4f_kernel_attribution/`,
  and `results/phase4g_optimization_hypothesis/`. No historical artifact was
  modified or deleted.

## Phase 5-A Environment Freeze And Harness Design (2026-09-05)

- Starting HEAD was `4979469d82e39910fe54de8275de442054e85b04`, the Phase 4
  evidence checkpoint. The new branch is
  `phase/05a-cuda-feasibility-baseline-study`.
- Read-only Jetson audit froze CUDA `12.6.68`, cuBLAS/cuBLASLt `12.6.1.4-1`
  (`cublasLtGetVersion() = 120601`), PyTorch `2.5.0a0+872d972e41.nv24.08`,
  TensorRT `10.3.0`, NCU `2024.3.1.0`, NSYS
  `2024.5.4.34-245434855735v0`, SM `8.7`, and no standard CUTLASS install.
- `nvidia-smi` did not expose useful integrated-platform power, clock, memory,
  p-state, or utilization fields; these are recorded as `N/A` or `UNKNOWN`,
  not inferred.
- The existing Mixed Decode engine remains read-only at 650,285,868 bytes,
  SHA-256 `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c`.
- The harness design freezes Phase 4-E as TensorRT Baseline 1 without rerun,
  defines a future direct cuBLASLt Baseline 2, and defines a future CUTLASS
  library-generated Baseline 3. No benchmark or implementation was run.
- Gate is `PASS / READ_ONLY_ENVIRONMENT_FREEZE` and harness design is
  `DESIGN_ONLY`; Phase 5-A benchmark execution is pending review.
- Evidence: `results/phase5a_cuda_feasibility_baseline/20260905T060916Z/`.

## Phase 5-A Direct cuBLASLt And CUTLASS Benchmark (2026-09-05)

- Branch `phase/05a-cuda-feasibility-baseline-study`; starting benchmark HEAD
  was `b300d1e9f7381d20fddec1649c7ca1a7a22309aa`. The Jetson checkout remained
  read-only on `phase/03e-tensorrt-kernel-attribution` at `bf7abc6`.
- Direct cuBLASLt primary result: FP16 input/output with FP32 accumulate,
  algorithm ID 21, heuristic index 4, workspace 0 bytes, median
  `0.080077961 ms`, mean `0.080076141 ms`, CV `0.000073666`, correctness PASS.
- All eight FP16-accumulate cuBLASLt records failed the frozen absolute-error
  gate and were excluded from performance ranking.
- CUTLASS `v3.5.1` (`f7b19de32c5d1f3cedfc735c2849f12b537522ee`) best candidate
  used `tb32x32x64/warp32x32x64/stages4/split-K1`: median `0.083619133 ms`,
  CV `0.000104699`, correctness PASS. It was about 4.42% slower than cuBLASLt.
- Phase 4-E TensorRT evidence was not rerun. Its per-layer `IProfiler` medians
  remain `0.153280005-0.159759998 ms`, but the runtime/layer boundary is not
  directly comparable to standalone CUDA-event GEMM timing.
- Gate is `INCONCLUSIVE / BOUNDED`: Case A and B are not applicable due to the
  boundary mismatch, and Case C is not satisfied because CUTLASS did not win.
  No Phase 5-B CUDA kernel discussion or implementation is authorized.
- Evidence: `results/phase5a_cuda_feasibility_baseline/20260905T063059Z/`;
  harnesses are under `experiments/Phase5-cuda-feasibility/src/`.

## Phase 5-A TensorRT GEMM Boundary Reconciliation (2026-09-05)

- Starting HEAD was `175018b312353c00ff3979c802907b828491631e`. The existing
  Phase 4-F Mixed persistent NSYS SQLite trace was re-read in read-only mode;
  no new profiling, TensorRT execution, benchmark, or historical artifact
  modification occurred.
- The trace contains 196 `up_proj` NVTX range instances across 28 logical layer
  names and 7 trace invocations. Every instance has exactly one correlated CUDA
  kernel; there are no attributed non-GEMM kernels.
- Observed family totals are `h16816`: 84 launches, median `253.680 us`;
  `sm80_xmma_gemm`: 112 launches, median `118.800 us`; other: 0. Family
  assignment changes by trace invocation and its cause remains `UNKNOWN`.
- Excluding the first invocation, correlated kernel median is `147.424 us`,
  host NVTX range median is `42.496 us`, and launch API duration median is
  `16.240 us`. All 196 kernels end after their NVTX range ends, so the negative
  NVTX-minus-kernel residual is not a disjoint runtime overhead component.
- The direct cuBLASLt median remains `80.077961 us`. The steady-state kernel
  median is `1.841x` that value; even the faster invocation medians are about
  `42-43%` higher. This supports `CASE_2_SUPPORTED_BOUNDED`, but no tactic
  defect is proven and TensorRT backend identity remains `UNKNOWN`.
- No CUDA kernel, CUTLASS optimization kernel, TensorRT Plugin, engine change,
  tactic forcing, or runtime redesign is authorized.
- Evidence: `results/phase5a_cuda_feasibility_baseline/20260905T072100Z/`;
  analysis script is `experiments/Phase5-cuda-feasibility/scripts/phase5a_boundary_reconciliation.py`.

## Phase 5-B Step 1 TensorRT GEMM Path Investigation (2026-09-05)

- Starting HEAD was `480aef94c49e6e39183c4e6395d9c3a45519ea34` on
  `phase/05a-cuda-feasibility-baseline-study`. The work was repository-side
  read-only analysis of frozen Phase 4/5-A evidence; no engine was
  deserialized, executed, rebuilt, or modified.
- Joined 28/28 HIGH-confidence Phase 4-A.1 `up_proj` mappings to the frozen
  TensorRT 10.3 EngineInspector JSON and 196 Phase 5-A correlated runtime
  kernel observations.
- Recovered Inspector tactic strings for all 28 layers. Decoder layers 0-7 and
  9-26 use `sm80_xmma_gemm_f16f16_f16f16_f16...`; layers 8, 14, and 27 use
  `sm80_xmma_gemm_f16f16_f16f32_f32...`. All labels contain
  `tensor16x8x16`.
- Numeric tactic ID, runtime workspace, backend identity, alpha/beta values,
  accumulator dtype, and physical layout semantics remain `NOT_AVAILABLE` or
  `UNKNOWN`. The `f16f32_f32` label is not treated as proven FP32 accumulator
  semantics.
- Runtime attribution remains bounded: each logical layer has three `h16816`
  and four `sm80_xmma_gemm` observations across seven historical invocations.
  Family assignment changes by invocation; the mapping is not proven
  one-to-one.
- Gate is `CASE_B_SUPPORTED_BOUNDED`, `NO_PROVEN_TACTIC_DEFECT`, and
  `NO_IMPLEMENTATION_AUTHORIZED`. Direct cuBLASLt algorithm 21 is comparison
  context only, not evidence of a TensorRT cuBLASLt backend.
- Evidence: `results/phase5b_tensorrt_gemm_path_investigation/20260905T103352Z/`;
  analysis script is `experiments/Phase5-cuda-feasibility/scripts/phase5b_tactic_attribution.py`.

## Phase 5-B Step 2 TensorRT vs cuBLASLt NCU Comparison (2026-09-05)

- Starting HEAD was `47ee7cd9ccd7ad551b5b5e2b44e3063f85ef4d45` on
  `phase/05a-cuda-feasibility-baseline-study`. The Jetson checkout remained
  read-only on `phase/03e-tensorrt-kernel-attribution` at `bf7abc6`.
- Profiled one post-warmup cuBLASLt algorithm-21 launch after `100` warmups and
  one Mixed Decode TensorRT
  `sm80_xmma_gemm_f16f16_f16f16_f16..._execute_kernel_trt` launch. NCU used
  `--clock-control none` and one launch per target.
- NCU duration was `242.912 us` for cuBLASLt and `244.160 us` for TensorRT;
  memory/L2 throughput was `76.06%` versus `76.92%`. The kernels had different
  resource shapes: cuBLASLt used more registers/shared memory and higher
  tensor-cycle activity, while TensorRT used higher occupancy and lower
  register/shared-memory pressure.
- The frozen CUDA-event cuBLASLt median remains `80.077961 us`; the historical
  TensorRT steady-state correlated kernel median remains `147.424 us`. NCU did
  not reproduce that event-time environment and the event-time gap remains
  `INCONCLUSIVE`.
- Gate is `CASE_A_SUPPORTED_BOUNDED` / `NO_PROVEN_OPTIMIZATION_TARGET`. No CUDA
  implementation is authorized.
- Evidence: `results/phase5b_tensorrt_gemm_path_investigation/20260905T112513Z/`;
  temporary cuBLASLt harness is
  `experiments/Phase5-cuda-feasibility/src/phase5b_step2_cublaslt_algo21_profile.cu`.

## Phase 5 Closeout And Next Target Re-selection (2026-09-05)

- Starting HEAD was `beee2ea4a3d364c0893d718285a4c60b4ad07bfa`. This was a
  repository-driven closeout over committed evidence; no Jetson execution,
  profiling, benchmark, engine rebuild, CUDA kernel, TensorRT Plugin, ONNX
  change, tactic forcing, or runtime modification occurred.
- Phase 5 final gate is `PASS / BOUNDED / NO_PROVEN_OPTIMIZATION_TARGET`,
  with `NO_PROVEN_CUDA_GEMM_OPTIMIZATION_TARGET`,
  `NO_PROVEN_TACTIC_DEFECT`, `EVENT_TIME_GAP_INCONCLUSIVE`,
  `TENSORRT_BACKEND_IDENTITY_UNKNOWN`, and
  `NO_CUDA_IMPLEMENTATION_AUTHORIZED`.
- `up_proj` is `CLOSED_FOR_NOW`. The historical steady-state `147.424 us`
  versus standalone cuBLASLt `80.077961 us` comparison retains its boundary
  caveat and was not reproduced under matched NCU.
- Next-target decision is `NEXT_TARGET_BOUNDED`. Rank 1 is the MEDIUM-confidence
  `unknown_attention_matmul` `/MatMul_*` chain, with `61.815776 ms` all-trace
  mapped time; it requires operator attribution and feasibility recovery, not
  implementation. Rank 2 is bounded `gate_proj`; Rank 3 is the single HIGH fused
  `q/k/v` range. No implementation is authorized.
- Evidence: `results/phase5_closeout_and_target_reselection/20260905T115247Z/`.
