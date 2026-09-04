# Phase 3-C Residual Runtime Profiling

- Date: 2026-09-04
- Branch: `phase/03c-residual-runtime-profiling`
- Starting checkpoint: `aa2545edf6670ef3e0e711735da24393f3aa5b13`
- Gate: `PASS / BOUNDED`

## 1. Question and Scope

Phase 3-B proved that persistent TensorRT execution contexts removed the
steady-state module-load storm and made Mixed roughly equal to FP16. Phase 3-C
therefore asked one question only: after that fix, where does the remaining
steady-state time go?

The authorized scope was profiling, attribution and bottleneck identification.
No CUDA kernel, TensorRT plugin, attention optimization, quantization redesign,
engine rebuild or Phase 3-D implementation was performed. Legacy runtime was
not profiled or benchmarked again.

## 2. Environment and Evidence State

- Device: Jetson Orin Nano Super, SM 8.7, batch 1.
- OS/kernel: Ubuntu 22.04 aarch64, Linux 5.15.148-tegra; L4T R36.4.3.
- Power mode: `NV Power Mode: 25W`; clocks were not modified.
- Software: CUDA 12.6, TensorRT 10.3.0, NVIDIA PyTorch 2.5.0a0 and Nsight
  Systems 2024.5.4.34-245434855735v0.
- FP16 engines: `/tmp/phase2_2b4_2_20260902T082326Z/{prefill,decode}_28layer.engine`.
- Mixed engines: `/tmp/phase2_3e_20260904T020000Z/work/mixed_{prefill,decode}_28layer.engine`.
- Auxiliary engines: the existing embedding, final RMSNorm and LM Head engines.
- Windows/Jetson execution branch was verified at `51f41c2` before C2.
- Phase 3-B artifacts remained untouched.

## 3. C1 Persistent Controlled Benchmark

Primary artifact:
`results/phase3c_residual_runtime/20260904T090803Z_bench/`.

Protocol: deterministic first evaluation prompt, batch 1, S=8/S=16, warmup 5,
repeats 10, 3 decode-window repeats, 8 decode steps per window, forced
deterministic continuation tokens and `time.perf_counter` with CUDA
synchronization before/after. Both runtimes used 5 context creations and 448
context reuses. Minimum checkpoint `MemAvailable` was `3,191,648,256` bytes; no
OOM or exit 137 occurred.

| Metric | FP16 Persistent | Mixed Persistent |
| --- | ---: | ---: |
| Prefill S=8 median, ms | 63.380289 | 46.981784 |
| Prefill S=16 median, ms | 40.003374 | 41.105138 |
| Decode TPOT S=8, ms | 46.070557 | 43.411762 |
| Decode TPOT S=16, ms | 41.645107 | 43.173353 |
| Throughput S=8, tokens/s | 22.663087 | 23.301708 |
| Throughput S=16, tokens/s | 23.943471 | 23.216906 |
| Decode window S=8, ms | 340.802388 | 365.059806 |
| Decode window S=16, ms | 341.495899 | 354.446278 |
| Initialization, s | 7.069036 | 6.662524 |

The S=8 FP16 prefill being slower than Mixed in this run is treated as noise
within the current benchmark, not as a semantic or optimization result. The
overall conclusion is that the two persistent runtimes are near parity at this
workload.

C1 gate: `PASS`.

## 4. C2 Steady-State Nsight Systems

Primary artifacts:
`results/phase3c_residual_runtime/20260904T093500Z_nsys/`.

Two separate NSYS traces were collected with
`--sample=none --cpuctxsw=none`:

1. `fp16_persistent`
2. `mixed_persistent`

Each trace reused the existing NVTX ranges for `PHASE3B_INIT`, `PHASE3B_WARMUP`,
`PHASE3B_STEADY_PREFILL_S8` and `PHASE3B_STEADY_DECODE_STEP_0..3`. The analysis
aggregates one representative prefill plus four representative decode steps.
Initialization and warmup are excluded from steady-state aggregates.

The CSV field called `host_gap_proxy_ms` is
`max(0, wall - max(CUDA API busy, GPU kernel busy))`. It is a bounded
wall-time residual, not a direct scheduler-level CPU gap measurement. API time
and kernel time can overlap.

| Metric, prefill + 4 decode steps | FP16 Persistent | Mixed Persistent |
| --- | ---: | ---: |
| Wall, ms | 319.333024 | 237.800576 |
| GPU kernel count | 2084 | 2452 |
| GPU kernel time, ms | 219.709696 | 147.830560 |
| GPU busy, % of wall | 68.802685 | 62.165770 |
| GPU idle, ms | 99.623328 | 89.970016 |
| CUDA API calls | 2364 | 3516 |
| CUDA API time, ms | 186.374432 | 116.749376 |
| CUDA API, % of wall | 58.363647 | 49.095498 |
| Synchronization calls | 38 | 38 |
| Synchronization time, ms | 143.809952 | 71.544608 |
| CUDA memcpy API time, ms | 0.517216 | 0.429440 |
| GPU memcpy/memset time, ms | 0.013376 | 0.010144 |
| Steady module load/unload, calls | 0 | 0 |
| Steady module load/unload, ms | 0 | 0 |

Representative S=8 prefill:

| Metric | FP16 Persistent | Mixed Persistent |
| --- | ---: | ---: |
| Wall, ms | 38.567424 | 33.651328 |
| GPU kernel time, ms | 22.677568 | 20.975776 |
| CUDA API time, ms | 21.160256 | 18.028736 |
| Synchronization time, ms | 14.211968 | 10.844128 |
| GPU busy, % | 58.799800 | 62.332684 |

The per-step decode times vary substantially in this small four-step window,
especially FP16 steps 2-3 and Mixed steps 2-3. Those values are evidence of
run-to-run and step-to-step variance in this short profile window; they are not
sufficient for a new performance optimization claim. Full per-range values are
in `analysis/c2_range_summary.csv` and `analysis/c2_fp16_vs_mixed.csv`.

Module-load checks:

| Range | FP16 load/unload | Mixed load/unload |
| --- | ---: | ---: |
| `PHASE3B_INIT` | 430 / 0 | 695 / 0 |
| `PHASE3B_WARMUP` | 0 / 0 | 0 / 0 |
| steady prefill + decode 0..3 | 0 / 0 | 0 / 0 |

These initialization loads are engine/context bring-up work and are excluded
from steady-state analysis. The required persistent steady-state check remains
zero for both runtimes.

C2 gate: `PASS`.

## 5. C3 Kernel / Operator Attribution

Primary artifacts: `analysis/c3_top_contributors.csv`,
`analysis/c3_range_categories.csv` and
`analysis/c3_top_kernels_by_name.csv`.

GEMM classification is name-based on explicit `xmma_gemm` / `h16816gemm`
tactic names. Names beginning `__myl_` are classified only as TensorRT
internal; no RMSNorm, RoPE, attention or other operator identity is inferred
from Myelin naming. Memory is the NVTX-filtered CUDA GPU memory-operation time.

| Category | FP16 time, ms | Mixed time, ms | Delta, Mixed-FP16, ms | Evidence |
| --- | ---: | ---: | ---: | --- |
| GEMM | 202.919104 | 133.532448 | -69.386656 | `NAME_BASED_GEMM_TACTIC` |
| Attention | 0.294816 | 0.366560 | +0.071744 | `NAME_BASED_MHA_GEMM_TACTIC_ONLY` |
| RMSNorm | 0 | 0 | 0 | `NOT_IDENTIFIED_BY_SAFE_KERNEL_NAME` |
| RoPE | 0 | 0 | 0 | `NOT_IDENTIFIED_BY_SAFE_KERNEL_NAME` |
| Elementwise | 0.004384 | 0.004160 | -0.000224 | `NAME_BASED_TORCH_STYLE_KERNEL` |
| Memory | 0.013376 | 0.010144 | -0.003232 | `NSYS_CUDA_GPU_MEM_TIME_SUM` |
| TensorRT internal | 16.491392 | 13.927392 | -2.564000 | `TRT_NAMESPACE_ONLY_OPERATOR_UNKNOWN` |
| Unknown | 0 | 0 | 0 | `NO_SAFE_NAME_MAPPING` |

GEMM is therefore the dominant named kernel category, but it is composed of
many tactics rather than one unavoidable kernel:

- FP16: the largest single kernel was 77.583328 ms, or 35.31% of FP16 kernel
  time; the second and third were 20.28% and 13.04%.
- Mixed: the largest single kernel was 39.590976 ms, or 26.78% of Mixed kernel
  time; the second and third were 17.67% and 12.57%.

No single GPU kernel exceeded half of the profiled steady-state kernel time in
either runtime. The precise tensor/operator ownership of the GEMM tactics
inside the full-model TensorRT graph remains `UNKNOWN`.

C3 gate: `PASS / BOUNDED`.

## 6. Bottleneck Classification

`PROVEN`:

- Persistent steady-state module load/unload is zero for both FP16 and Mixed.
- GEMM is the dominant named kernel category in this workload.
- No single GPU kernel dominates more than 50% of kernel time in this trace.

`SUPPORTED`:

- GPU kernel time is a substantial fraction of NVTX wall time.
- A material wall-time residual remains outside the simple max of CUDA API and
  kernel busy time: 99.623 ms for FP16 and 89.970 ms for Mixed.
- FP16 and Mixed persistent runtimes are already near parity in the C1
  benchmark.

`UNKNOWN` / `INCONCLUSIVE`:

- Exact scheduler-level CPU gaps beyond the recorded wall-time proxy.
- Exact DRAM bandwidth and achieved occupancy, because NCU was not run.
- Exact mapping from GEMM tactics or `__myl_` kernels to decoder operators.
- A causal explanation for the short-window per-step decode variance.

Decision-gate result: `Case C - Mixed/FP16 are already near parity, and there
is no justified single CUDA operator target from this evidence alone`.

## 7. NCU Decision

`NSIGHT COMPUTE NOT REQUIRED`.

The predefined C4 trigger was one dominant GPU kernel. The largest observed
single kernels were 35.31% and 26.78% of their respective kernel totals. That
does not satisfy the trigger. NCU was therefore not run.

## 8. Phase 3-D Recommendation

Do not enter Phase 3-D CUDA operator optimization on this evidence. There is no
single GPU kernel that justifies a focused CUDA replacement, and the persistent
runtimes are already near parity.

If later work is authorized, the evidence-backed directions are runtime or
scheduling work (for example host/runtime gap isolation or CUDA graph
feasibility), not immediate RMSNorm/RoPE/attention kernel implementation. No
such phase was started.

## 9. Limitations

- The benchmark is bounded to the current single-request runtime, batch 1,
  the first deterministic evaluation prompt and S=8/S=16.
- The profile uses one representative prefill and four decode steps; it is not
  a long-tail or concurrent-serving profile.
- API time, synchronization time and GPU kernel time can overlap. The report
  does not claim a non-overlapping CPU/runtime/GPU decomposition.
- Kernel attribution is name-based. TensorRT internal kernels and full-model
  GEMM operator ownership are not precisely mapped.
- The Mixed NSYS run happened after the FP16 run; run-to-run DVFS effects were
  not measured and cannot be excluded.
- The C1 S=8 prefill ordering differs from S=16 and is treated as noise rather
  than an optimization signal.

## 10. Reproduction Commands

From the Jetson repository root:

```bash
python3 experiments/Phase2-qwen3-quantization/src/phase3c/phase3c_residual_runtime.py \
  --mode bench \
  --out results/phase3c_residual_runtime/<new-benchmark-dir>

nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/phase3c_nsys_<unique>/fp16_persistent \
  python3 experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py \
  --mode profile \
  --out results/phase3c_residual_runtime/<new-profile-dir>/fp16_persistent_profile \
  --profile-runtime fp16 --profile-seq-len 8 --profile-decode-steps 4 \
  --lifetime persistent_context_lifetime

nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/phase3c_nsys_<unique>/mixed_persistent \
  python3 experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py \
  --mode profile \
  --out results/phase3c_residual_runtime/<new-profile-dir>/mixed_persistent_profile \
  --profile-runtime mixed --profile-seq-len 8 --profile-decode-steps 4 \
  --lifetime persistent_context_lifetime

python3 experiments/Phase2-qwen3-quantization/src/phase3c/analyze_nsys.py \
  --nsys-dir results/phase3c_residual_runtime/<new-profile-dir> \
  --out results/phase3c_residual_runtime/<new-profile-dir>/analysis
```
