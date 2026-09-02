# Changelog

本文件记录项目管理、实验和基础设施的重要变化。

## [Unreleased]

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
