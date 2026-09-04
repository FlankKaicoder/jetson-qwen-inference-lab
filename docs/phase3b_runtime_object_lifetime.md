# Phase 3-B Runtime Object Lifetime Optimization

- Date: 2026-09-04
- Branch: `phase/03b-runtime-object-lifetime`
- Starting Phase 3-A checkpoint: `87ad72f1d87150a720c6e5316620ed9fd8767001`
Gate: `PASS / CLOSED / PROVEN`

## 1. Question and Scope

Phase 3-A proved that the Mixed runtime slowdown was dominated by host CUDA API
time, especially `cuModuleLoadData` and `cuModuleUnload`. Phase 3-B tested the
remaining causal link:

```text
short-lived TensorRT execution context
  -> repeated CUDA module load/unload
  -> host-side runtime slowdown
```

The intervention changed only runtime object lifetime. Engine contents,
precision, TensorRT graphs, prompts, tensor shapes, CUDA computation, warmup,
repetition methodology and power mode were held fixed. No CUDA kernel, plugin,
attention change, quantization policy change or engine rebuild was performed.
The validation is bounded to the current single-request, single-threaded
runtime.

## 2. Environment and Evidence State

- Device: Jetson Orin Nano Super, SM 8.7, batch 1.
- OS/kernel: Ubuntu 22.04 aarch64, Linux 5.15.148-tegra; L4T R36.4.3.
- Power mode: `NV Power Mode: 25W`; clocks were not modified.
- Software: CUDA 12.6, TensorRT 10.3.0, NVIDIA PyTorch 2.5.0a0 and Nsight
  Systems 2024.5.4. Nsight Compute was not used.
- FP16 engines: `/tmp/phase2_2b4_2_20260902T082326Z/{prefill,decode}_28layer.engine`.
- Mixed engines: `/tmp/phase2_3e_20260904T020000Z/work/mixed_{prefill,decode}_28layer.engine`.
- Auxiliary engines: the existing embedding, final RMSNorm and LM Head engines.
- B0 evidence was committed at `40eb2ee`. The implementation was committed at
  `9b61f4e`. B2/B3/B4 evidence records Git HEAD `1e4d888`.

## 3. B0 Lifecycle Audit

Source: `results/phase3b_runtime_lifetime/b0_lifecycle_audit.json` and the
Phase 2.3-F runtime code path.

| Object | Lifetime | Re-created per inference? |
| --- | --- | ---: |
| `trt.Runtime` | Persistent for the Python `TRT` object | No |
| deserialized Engine | Persistent for the engine wrapper | No |
| ExecutionContext | Function-call local inside `TRT.run` | Yes |
| CUDA stream | Persistent wrapper around the current PyTorch stream | No |
| shapes/addresses | Reset on every call | Required for correctness |

The exact recreation site was `TRT.run`: `self.engine.create_execution_context()`.
The actual engine layout has five persistent wrappers: prefill, decode,
embedding, final RMSNorm and LM Head. Legacy mode therefore created:

- 2 contexts per decoder-only prefill or decode call;
- 4 contexts when the prefill/decode pipeline also executed norm and LM Head;
- 28 contexts for the Phase 3-A profile sequence of two warmup prefills, one
  measured prefill and four measured decode steps.

B0 gate: `PASS`. The short-lived object was proven from the code path to be the
ExecutionContext. Runtime and engine were already persistent, so only context
lifetime was changed.

## 4. B1 Minimal Persistent Implementation

The feature flag has two values: `legacy_context_lifetime` and
`persistent_context_lifetime`.

- Legacy mode subclasses the original `TRT.run` unchanged and increments a
  creation counter. It retains the per-call context creation/destruction
  behavior.
- Persistent mode creates one context per engine wrapper during initialization,
  then reuses it for every run.
- Persistent mode does not cache temporary tensor pointers. It sets every input
  shape and input/output tensor address on every call, allocates fresh output
  tensors on every call, executes on the same engine-owned stream and
  synchronizes.
- There is one context per engine; no context is shared across engines. No
  multi-thread context pool, continuous batching or request scheduler was added.

## 5. B2 Functional Validation

Primary artifact:
`results/phase3b_runtime_lifetime/20260904T081621Z/functional_result.json`.
The comparison was Mixed legacy versus Mixed persistent, not Mixed versus HF.
The deterministic first evaluation prompt was used at batch 1.

- S=8 prefill plus 8 decode steps: hidden, logits and all 28 K/V tensors were
  exactly equal. `all_exact=true`, `all_finite=true`, shapes passed, KV gate
  passed and invariants passed.
- S=16 prefill: hidden, logits and all K/V tensors were exactly equal with all
  gates passed.
- KV prefixes were preserved, cache lengths advanced exactly and the K/V
  pointers remained isolated across all 28 layers.
- Legacy S=8 made 36 context creations with 0 reuses; persistent made 5 and
  reused them 36 times. Legacy S=16 made 4; persistent made 5 and reused 4.
- Minimum checkpoint `MemAvailable` was `4,195,950,592` bytes. There was no
  OOM, exit 137 or new numerical/token regression.

B2 gate: `PASS`.

## 6. B5 Initialization and Memory Trade-off

Source: `results/phase3b_runtime_lifetime/20260904T081727Z/bench_result.json`.
Initialization is the wall time to deserialize the five engines and create the
contexts/pipeline. Memory rows are sequential same-process observations, not an
isolated context-cost experiment.

| Runtime | Legacy initialization s | Persistent initialization s | Delta s |
| --- | ---: | ---: | ---: |
| FP16 | 3.349467 | 6.779457 | +3.429990 |
| Mixed | 2.636420 | 6.655416 | +4.018996 |

| Runtime | Legacy loaded `MemAvailable` B | Persistent loaded `MemAvailable` B | Delta B |
| --- | ---: | ---: | ---: |
| FP16 | 3,428,831,232 | 3,040,542,720 | -388,288,512 |
| Mixed | 3,611,865,088 | 3,366,309,888 | -245,555,200 |

At the corresponding loaded rows, CUDA free memory was lower by
`395,350,016` bytes for FP16 and `235,732,992` bytes for Mixed. The minimum
benchmark checkpoint `MemAvailable` was `3,038,470,144` bytes. No OOM or exit
137 occurred. The exact retained-context-only cost is `INCONCLUSIVE` because
these are observational snapshots across sequentially loaded wrappers and the
allocator state is not solely attributable to contexts. No deployment memory
blocker was observed.

The one-time initialization cost is separate from the recurring per-call win.
At the measured workload, persistent mode removes roughly 2.212-2.213 seconds
of Mixed legacy prefill latency and 2.619-2.620 seconds of Mixed legacy TPOT
latency.

## 7. B3 Four-Way Benchmark

Primary artifact: `results/phase3b_runtime_lifetime/20260904T081727Z/bench_result.json`.
Protocol: deterministic first evaluation prompt, batch 1, S=8/S=16, warmup 5,
repeats 10, 3 decode-window repeats, 8 decode steps per window, forced
deterministic continuation tokens, `time.perf_counter` with CUDA
synchronization before/after. All four configurations ran in one session.

| Metric | FP16 Legacy | FP16 Persistent | Mixed Legacy | Mixed Persistent |
| --- | ---: | ---: | ---: | ---: |
| Prefill S=8 median, ms | 1512.489 | 50.226 | 2259.919 | 47.627 |
| Prefill S=16 median, ms | 1508.887 | 40.322 | 2255.619 | 42.406 |
| Decode TPOT S=8 median, ms | 1955.252 | 43.423 | 2662.990 | 43.362 |
| Decode TPOT S=16 median, ms | 1956.412 | 42.030 | 2662.266 | 43.580 |
| Decode throughput S=8, tokens/s | 0.510542 | 23.834265 | 0.375194 | 23.349145 |
| Decode throughput S=16, tokens/s | 0.509917 | 23.773820 | 0.374631 | 22.924270 |
| Loaded `MemAvailable`, B | 3,428,831,232 | 3,040,542,720 | 3,611,865,088 | 3,366,309,888 |

Context counters for the complete B3 workload were 448 creations/0 reuses for
legacy and 5 creations/448 reuses for persistent in both FP16 and Mixed.

### Gap Recovery

`GapRecovery = (OldGap - NewGap) / OldGap`, where OldGap compares Mixed legacy
to FP16 legacy and NewGap compares Mixed persistent to FP16 persistent.

| Workload | Old gap ms | New gap ms | Gap recovery |
| --- | ---: | ---: | ---: |
| Prefill S=8 | 747.430469 | -2.598402 | 100.348% |
| Prefill S=16 | 746.732776 | 2.084089 | 99.721% |
| Decode S=8 | 707.738169 | -0.061028 | 100.009% |
| Decode S=16 | 705.854476 | 1.549500 | 99.780% |

Values above 100% correspond to a small negative residual gap. They should be
read as near-complete recovery within measurement noise, not as extra
theoretical improvement. Mixed is now roughly equal to FP16; the remaining
milliseconds-scale S=16 deltas are small and noisy.

## 8. B4 Nsight Systems Causal A/B

Primary compact artifacts:
`results/phase3b_runtime_lifetime/20260904T082912Z_nsys/analysis/`.
All four configurations were profiled. The table uses the S=8 prefill range and
decode step 0 after the two-warmup initialization boundary. Times are ms.

| Runtime / range | Legacy wall | Persistent wall | Legacy module load/unload | Persistent module load/unload | Legacy API | Persistent API |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 prefill S=8 | 1559.613 | 42.003 | 310+310 / 304.112 | 0+0 / 0 | 352.114 | 21.887 |
| FP16 decode step 0 | 2133.301 | 73.985 | 122+126 / 262.041 | 0+0 / 0 | 311.266 | 33.008 |
| Mixed prefill S=8 | 2310.857 | 33.217 | 416+416 / 861.718 | 0+0 / 0 | 896.733 | 18.957 |
| Mixed decode step 0 | 2881.660 | 65.848 | 281+285 / 890.140 | 0+0 / 0 | 926.188 | 22.452 |

Module load/unload count and time reduction was `100%` for every profiled
steady-state range. CUDA API time fell by 87.8-97.9% across the FP16 and Mixed
ranges. Wall reduction for the primary Mixed ranges was 98.56% at prefill and
97.71% at decode step 0.

For the same ranges, GPU kernel counts remained identical in the persistent
A/B, while measured kernel-time totals varied:

| Runtime / range | Legacy kernel ms | Persistent kernel ms |
| --- | ---: | ---: |
| FP16 prefill S=8 | 47.209 | 22.690 |
| FP16 decode step 0 | 48.913 | 45.673 |
| Mixed prefill S=8 | 42.623 | 20.852 |
| Mixed decode step 0 | 43.070 | 35.629 |

B2 proved exact output equality, so kernel-time variation should not be
interpreted as a semantic change. It may reflect profiler timing/device-state
variation and requires separate attribution if it becomes the next bottleneck.

Sync and memcpy were small in legacy mode. Mixed prefill had 13
`cudaStreamSynchronize` calls totaling 14.970 ms and 164 DtoH calls totaling
8.356 ms. Mixed persistent had 5 sync calls totaling 12.360 ms and no DtoH
copies in the measured range. Mixed decode step 0 had 12.056 ms sync and 8.288
ms DtoH legacy versus 10.386 ms sync and 0 ms DtoH persistent.

## 9. H3B-1 Classification

`H3B-1` is `PROVEN` for this runtime and workload:

1. B0 proved the ExecutionContext was repeatedly recreated in `TRT.run`.
2. Persistence reduced repeated context creation: B2 used 5 creations/36 reuses
   and B3 used 5 creations/448 reuses instead of legacy all-creation.
3. Steady-state `cuModuleLoadData` and `cuModuleUnload` counts and time
   collapsed to zero for every profiled range.
4. End-to-end wall latency dropped in the expected direction by roughly
   97.7-98.6% in the profiled Mixed ranges.
5. Mixed legacy and persistent outputs were exactly equal and finite, with KV
   prefix and pointer invariants intact.

The exact root-cause conclusion is that per-call TensorRT execution-context
lifetime was the dominant cause of the earlier Mixed slowdown through repeated
CUDA module load/unload activity. Mixed engine structure was not the primary
cause; direct Q/DQ GPU cost had already been disproven in Phase 3-A.

## 10. Limitations

- The result is bounded to batch 1, S=8/S=16, a short deterministic decode and
  the current single-request runtime. It is not a multi-request or CUDA Graph
  claim.
- Exact retained-context memory cost is `INCONCLUSIVE`; only observed loaded
  and steady-state memory deltas are supported.
- The S=8 persistent prefill and TPOT CV values span 7.94-16.78% across FP16
  and Mixed, higher than the corresponding S=16 values; medians are used and
  the >100% recovery cases are treated as noise-bounded.
- NSYS initialization remains outside the steady-state ranges, so the report
  does not claim module loading disappeared from the process lifetime.
- Raw `.nsys-rep` reports remain Jetson-local under
  `/tmp/phase3b_nsys_20260904T082912Z/`; only compact exports are committed.
- High-level Myelin kernel-name mapping remains `UNKNOWN`.

## 11. Artifacts and Reproduction

Primary artifacts:

- B0 audit: `results/phase3b_runtime_lifetime/b0_lifecycle_audit.json`
- B2 functional PASS:
  `results/phase3b_runtime_lifetime/20260904T081621Z/`
- B3 benchmark and B5 memory:
  `results/phase3b_runtime_lifetime/20260904T081727Z/`
- B4 NSYS compact evidence:
  `results/phase3b_runtime_lifetime/20260904T082912Z_nsys/`
- Retained failed diagnostics:
  `results/phase3b_runtime_lifetime/20260904T080822Z/`,
  `20260904T081011Z/`, `20260904T081136Z/`, `20260904T081425Z/`

From the Jetson repository root:

```bash
python3 experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py \
  --mode functional \
  --out results/phase3b_runtime_lifetime/<new-timestamp>

python3 experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py \
  --mode bench --out results/phase3b_runtime_lifetime/<new-timestamp> \
  --prefill-lengths 8 16 --warmup 5 --repeats 10 \
  --window-repeats 3 --decode-steps 8

nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/phase3b_nsys_<unique> \
  python3 experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py \
  --mode profile --out /tmp/phase3b_profile_<unique> \
  --profile-runtime mixed --profile-seq-len 8 --profile-decode-steps 4 \
  --lifetime persistent_context_lifetime

python3 experiments/Phase2-qwen3-quantization/src/phase3b/analyze_nsys.py \
  --stats-dir <exported-stats-dir> --out <analysis-out-dir>
```

Repeat NSYS separately for `fp16`/`mixed` and
`legacy_context_lifetime`/`persistent_context_lifetime`, using a unique output
path for each profile. Export the NVTX kernel/range and range-filtered CUDA API
CSV layouts expected by `analyze_nsys.py`; the committed analyzer consumes
`stats/` and `stats3/` CSVs.

## 12. Final Gate

`PASS / CLOSED / PROVEN`

The mechanism, functional safety, benchmark effect, causal NSYS evidence and
memory trade-off are all present. Phase 3-C, CUDA operator work, TensorRT
plugin work, attention work and new quantization work were not started.
