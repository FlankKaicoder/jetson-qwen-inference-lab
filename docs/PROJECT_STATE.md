# Project State

本文件是无历史上下文的新会话恢复项目状态的入口。完整实验方法与分析以实验报告和 raw artifacts 为准；若本页与更高优先级证据冲突，以 Git 和 raw artifacts 为准。

## Current State

| Field | Verified value |
| --- | --- |
| Project | `jetson-qwen-inference-lab` / Jetson Qwen Transformer AI Infra Optimization Lab |
| Current date | `2026-09-02` |
| Repository | `FlankKaicoder/jetson-qwen-inference-lab` |
| Windows path | `E:\nvidia-qwen` |
| Jetson path | `/home/nvidia/projects/jetson-qwen-inference-lab` |
| GitHub | `https://github.com/FlankKaicoder/jetson-qwen-inference-lab` |
| Current phase | Phase 2 — LLM Quantization |
| Current experiment | Phase 2.2-C1 — Qwen3 embedding integration |
| Current branch | `phase/02-qwen3-quantization` |
| Current HEAD | Verify with `git rev-parse HEAD` |
| Main HEAD | `d42ab4aeabc751723a4a2c1036b93a5ed16d3d01` |
| Last completed experiment | Exp04 — CUDA GEMM tiling and WMMA |
| Experiment status | Phase 1 `PASS / CLOSED`; Phase 2.0 `BLOCKED`; Phase 2.1 `INCONCLUSIVE`; Phase 2.1.5 `PASS / CLOSED`; Phase 2.1.8/2.1.9 `PASS / BOUNDED`; Phase 2.2-A `PARTIAL / BOUNDED PASS`; Phase 2.2-B1/B2/B3 `PASS / BOUNDED`; Phase 2.2-B4.1 `PASS / CLOSED`; Phase 2.2-B4.2 `PASS / BOUNDED`; Phase 2.2-C0 `PASS / DESIGN ONLY`; Phase 2.2-C1 `BLOCKED` |
| Current Gate | Phase 2.2-C1 embedding-only gates PASS; C1D boundary diagnostic complete; decoder integration remains BLOCKED |
| Readiness | C1 blocked; C2 must not start. D0 PASS, D1 byte-identical, D3 host/direct identical failure; no stream/pointer/binding root cause confirmed |

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
- Gate A `PASS`; Gate B/C `BLOCKED`; Gate D `INCONCLUSIVE`; Phase 2.0 remains `RUNNING/BLOCKED` pending an explicitly authorized compatible backend path.

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

Owner review of Phase 2.2-B4.2 evidence. Do not begin embedding/final-norm/LM-head integration, full token generation, benchmarking, profiling, INT8/INT4, TensorRT-LLM, Phase 2.2-C/D, Phase 2.3 or Phase 3 without explicit authorization.

## Do-not-repeat Work

- Do not rerun or overwrite Original Exp01 correctness/benchmark, Exp01.1 stability, or Exp01.2 profiler artifacts merely to reconstruct context.
- Do not repeat the completed sudo/NCU permission setup or modify sudoers.
- Do not claim a precise 128/256 microarchitectural cause; the final evidence-backed status is `INCONCLUSIVE`.
- Do not estimate direct DRAM throughput where NCU reports `N/A`.
- Do not start Exp02, merge `main`, change the roadmap, or modify device power/clock state without explicit direction.

## Last Verified Git State

Before Exp01.2 profiling on `2026-08-30`, Windows, GitHub and Jetson were clean at `exp/01-vector-add@e10f2c06c7d90844cbf425e5ef6c32a413e314ec`; local and remote `main` were `d42ab4aeabc751723a4a2c1036b93a5ed16d3d01`. The final closure commit and push/pull state must be verified from Git at session end.

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
