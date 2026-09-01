# Phase 1.2 - Qwen3-0.6B BF16 Formal LLM Benchmark

## Purpose

Establish the exact-revision Hugging Face BF16/eager batch-1 performance reference used by later precision and runtime comparisons. The workload separates prefill from autoregressive decode and uses fixed exact shapes. It is a synthetic runtime benchmark, not a service benchmark or language-quality evaluation.

## Benchmark Identity

- Model: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
- Weight SHA256: `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Runtime: NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`, CUDA 12.6, Transformers 4.57.3, Accelerate 1.14.0, Orin SM87, BF16.
- Attention: Transformers built-in `eager`; batch 1; no quantization, compilation, offload, TensorRT, or external attention package.
- Platform: Ubuntu 22.04.5, L4T R36.4.3. Power mode remained 25W.
- Read-only clocks: CPU max 1.344 GHz, GPU max 918 MHz, EMC max 3.199 GHz with `FreqOverride=0`. Clocks were not locked.

## Method Validation

The manual KV-cache loop and `model.generate()` produced identical 8/8 greedy token IDs for the same non-thinking arithmetic prompt. The cache class was `DynamicCache`; observed lengths `[26..33]` exactly matched expectation. Timing-accounting error was 2.529 ms against a 10.382 ms tolerance. Exact ISL construction and fixed OSL behavior also passed.

Gate A - Method Validation: `PASS`.

## Formal Methodology

- Exact ISL: 32, 128, 512, and gated extension 1024. Exact valid token IDs were produced by repeating the tokenized `Hello world. ` sequence without padding.
- OSL: exactly 32 tokens even if greedy argmax selected EOS.
- Each ISL ran in a fresh process with one model load, five seconds loaded-model idle sampling, two complete warmups, and ten formal trials.
- The predefined extension to 15 trials applied only if TTFT or TPOT CV exceeded 5%; no shape triggered it.
- Prefill GPU latency used CUDA Events around the full-sequence model forward only.
- Inference TTFT, decode-step wall latency, TPOT, and E2E used `perf_counter_ns` with CUDA synchronization at timing boundaries. Tokenization, model load, disk I/O, and request/network overhead were excluded.
- Every formal trial retained all 31 decode-step GPU and wall timings, cache lengths, token IDs, allocator counters, CUDA memory info, and `/proc/meminfo` checkpoints.
- Tegrastats sampled every 200 ms. `VDD_IN` is treated only as board/module input power, not process-exclusive model power.
- The 1024 pilot passed its predefined memory gate before the formal run: minimum MemAvailable 2,343,194,624 bytes, swap growth 87,031,808 bytes, no OOM/CUDA error, and valid cache growth.

## Main Results

Values below are medians across ten formal trials. Memory is the minimum formal checkpoint; swap is the maximum formal checkpoint.

| ISL | OSL | Prefill GPU ms | Prefill tok/s | Inference TTFT ms | TPOT ms | Decode tok/s | E2E ms | Min MemAvailable | Peak swap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 128.828 | 248.393 | 129.067 | 125.206 | 7.987 | 4010.865 | 2.523 GiB | 361.0 MiB |
| 128 | 32 | 128.385 | 997.009 | 128.682 | 121.168 | 8.253 | 3885.192 | 2.389 GiB | 361.0 MiB |
| 512 | 32 | 300.934 | 1701.372 | 301.301 | 118.440 | 8.443 | 3971.578 | 2.369 GiB | 359.5 MiB |
| 1024 | 32 | 614.409 | 1666.644 | 614.607 | 115.542 | 8.655 | 4197.960 | 2.254 GiB | 513.2 MiB |

The summary CSV contains mean, median, standard deviation, CV, min, max, and P90 fields; the all-trials CSV retains every trial. The 1024 pilot timing was a safety observation and is not substituted for the formal statistics.

## Stability

| ISL | Trials | TTFT CV | TPOT CV | Result |
| ---: | ---: | ---: | ---: | --- |
| 32 | 10 | 0.305% | 0.187% | PASS |
| 128 | 10 | 1.050% | 0.122% | PASS |
| 512 | 10 | 3.042% | 0.187% | PASS |
| 1024 | 10 | 0.838% | 0.226% | PASS |

All 40 formal trials had exact OSL=32, valid cache growth, valid timing accounting, and no CUDA/OOM failure. Start/end power mode was identical for every process.

## Power And Thermal Evidence

| ISL | Loaded idle mean | Active mean | Active median | Active peak | Peak GPU/TJ temp |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4.767 W | 7.152 W | 7.147 W | 7.265 W | 52.031 C |
| 128 | 4.799 W | 7.419 W | 7.384 W | 7.976 W | 53.250 C |
| 512 | 4.768 W | 8.560 W | 8.081 W | 11.612 W | 55.093 C |
| 1024 | 4.781 W | 10.933 W | 10.374 W | 16.537 W | 58.312 C |

The active windows contain 189-204 samples per shape. The increasing observed `VDD_IN` and temperature are board-level observations under the fixed order 32 -> 128 -> 512 -> 1024; they are not process-exclusive attribution. No explicit throttle evidence was observed, and the recorded peak was 58.312 C. This experiment does not define a hardware throttling threshold.

## Context Scaling

- ISL 32 and 128 had effectively equal prefill latency, showing fixed/runtime overhead at short shapes; throughput rose because four times as many input tokens completed in the same time.
- Relative to ISL 32, median prefill latency was 2.34x at 512 and 4.77x at 1024. TTFT tracked prefill almost exactly: 2.33x and 4.76x.
- Median TPOT varied only from 125.206 to 115.542 ms, a 7.7% decrease rather than an increase. Decode was much less context-sensitive than prefill over this range.
- Maximum observed PyTorch allocated memory increased from 1,158.7 MiB at ISL 32 to 1,562.6 MiB at ISL 1024. Minimum MemAvailable declined by about 0.269 GiB, and the formal 1024 process reached 513.2 MiB swap used.
- The measurements establish scaling behavior for this exact HF/eager baseline. They do not establish a memory-bandwidth, Tensor Core, occupancy, or kernel-level cause.

## Hypotheses

- H1, prefill latency increases with ISL: `PARTIALLY SUPPORTED`. It is flat from 32 to 128 and clearly increases at 512 and 1024.
- H2, TTFT is increasingly dominated by prefill: `PARTIALLY SUPPORTED`. TTFT tracks prefill within roughly 0.2-0.4 ms at every shape, but the dominance fraction is already near-total and does not demonstrably increase.
- H3, decode TPOT is less sensitive to ISL but may increase with KV length: `PARTIALLY SUPPORTED`. Low sensitivity is supported; the observed TPOT decreased, so an increase is not supported here.
- H4, unified-memory use increases with sequence length: `SUPPORTED` by allocator growth and lower minimum MemAvailable, with system-level accounting caveats.
- H5, eager attention may make long-context prefill less efficient: `INCONCLUSIVE`. Long-shape latency increased, but no alternate attention path or profiler evidence exists to attribute a mechanism.
- H6, board power and thermal behavior may vary with ISL: `SUPPORTED` as a board-level observation, not process-exclusive attribution.

## Functional Regression

The required post-benchmark Chinese smoke prompt produced `北京` in two tokens with semantic sanity `true`. This confirms the benchmark work did not alter the established functional behavior; it is not a quality evaluation.

## Gates And Status

- Gate A - Method Validation: `PASS`.
- Gate B - Formal Benchmark: `PASS`; all core shapes and the gated 1024 extension completed with retained raw evidence.
- Gate C - Measurement Quality: `PASS`; all TTFT/TPOT CV values were below 5%, power mode was consistent, and memory/power/thermal evidence was retained.
- Gate D - Context Scaling: `PASS`; prefill, TTFT, decode, memory, and board-level power scaling can be answered from the evidence.
- Phase 1.2: `PASS / CLOSED`.
- Phase 1 remains `IN PROGRESS`; its next direction requires explicit owner review.

## Evidence And Limitations

- Raw and derived evidence: `artifacts/phase1_2_formal_20260901T075200Z/`.
- Formal summary: `phase1_2_summary_20260901T081200Z.csv`.
- All trials: `phase1_2_all_trials_20260901T081200Z.csv`.
- Power/thermal summary: `phase1_2_power_thermal_20260901T081200Z.csv`.
- Per-run JSON contains model/runtime/Git/power-mode and raw timing metadata. L4T and read-only clock-query evidence are retained in the experiment-level environment sidecar rather than duplicated into the already-captured run JSON.
- This is batch-1, fixed OSL=32, eager-attention, synthetic exact-token inference. It excludes service overhead and is not a maximum-context, concurrency, quality, or profiler study.
- The baseline is suitable for later quantization comparison if model revision, prompt shapes, timing boundaries, power state, and runtime identity remain controlled. No later phase is authorized by this result alone.

No profiler was run. No power mode, clocks, fan, swap, CUDA, PyTorch, or Transformers setting was changed. No TensorRT-LLM source or TensorRT engine was installed or built. No quantization, Phase 2, or Exp05 work was started. The failed partial venv remains preserved and untouched.
