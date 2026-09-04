# Phase 3-A Runtime Bottleneck Attribution

Date: 2026-09-04  
Branch: `phase/03-runtime-bottleneck-attribution`  
Starting Phase 2 checkpoint: `b2083895b1199edf8a3245013912f82aec1b942b`  
Gate: `PASS / BOUNDED`

## 1. Question and Scope

Phase 2.3-F proved that a deployable mixed-precision TensorRT runtime with
133 PT-W8A8 Linear targets was 48-49% slower at prefill and 35-36% slower at
decode than the FP16 runtime, despite 27% smaller serialized engines. The
Phase 3-A question was to attribute that slowdown without changing the
quantization policy, rebuilding engines, or starting optimization work.

The comparison was `TRT FP16` versus `TRT MIXED` only. Phase 0/1/2 artifacts
and conclusions were reused read-only. The memory-lifetime rule was preserved:
the runtime continued streaming layer engines rather than retaining all
HuggingFace weights.

## 2. Environment and Engines

- Device: Jetson Orin Nano Super, SM 8.7, batch 1.
- OS/kernel: Ubuntu 22.04 aarch64, Linux 5.15.148-tegra; L4T R36.4.3 /
  JetPack 6.2 mapping.
- Power mode: `NV Power Mode: 25W`; clocks were not modified.
- Software: CUDA 12.6, TensorRT 10.3.0, NVIDIA PyTorch 2.5.0a0, Nsight Systems
  2024.5.4. Nsight Compute was not used.
- FP16 engines: Jetson `/tmp/phase2_2b4_2_20260902T082326Z/{prefill,decode}_28layer.engine`.
- Mixed engines: Jetson `/tmp/phase2_3e_20260904T020000Z/work/mixed_{prefill,decode}_28layer.engine`.
- Embedding: `/tmp/phase2_2c1_20260902T090000Z/embedding_fp16.engine`.
- Final RMSNorm: `/tmp/phase2_2c2_20260903T/final_rmsnorm_fp32_reduce.engine`.
- LM Head: `/tmp/phase2_2c3_20260903T024500Z/lm_head_fp16.engine`.
- Git state during A1: `7067d7c8d6f42c24a54c8f774c9d530a105cd309`; during
  analysis: `903f7c1ee86a77d9a1dccb9b52f4839fba32e034`.

## 3. A1 Controlled Benchmark

Workload was the deterministic first Phase 2.3-B evaluation prompt, batch 1.
The engine profile limit is 16 tokens, so prefill used `S=8` and `S=16`; decode
TPOT measured one step anchored at the corresponding cache length. An
8-step decode window used an 8-token cache and positions 8..15 so it stayed
inside the 16-token profile. Both runtimes used the same prompt IDs, forced
deterministic continuation tokens, embedding/norm/LM-head engines, host timing
method, warmup, repetitions, process order and power mode.

Protocol: warmup 5, repeats 10 for prefill/TPOT, 3 decode-window repeats,
8 decode steps per window. Timing was `time.perf_counter` with
`torch.cuda.synchronize` immediately before and after the measured region.

| Metric | FP16 | Mixed | Mixed slower |
| --- | ---: | ---: | ---: |
| Prefill S=8 median | 1517.606 ms | 2263.352 ms | 49.140% |
| Prefill S=16 median | 1521.247 ms | 2265.980 ms | 48.955% |
| Decode TPOT at S=8 | 1959.902 ms | 2665.612 ms | 36.007% |
| Decode TPOT at S=16 | 1963.507 ms | 2664.299 ms | 35.691% |
| Decode tokens/s | 0.508-0.510 | 0.374-0.375 | 26.4-26.5% lower |

The slowdown reproduced for both sequence lengths and both prefill and decode.
`a1_gate.json` records `reproduced_mixed_slower = true`; A1 gate is `PASS`.

## 4. A2 Nsight Systems Method

Each runtime was profiled separately with:

```bash
nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none -o <unique-path> \
  python3 ... --mode profile --out <artifact-dir> \
  --profile-runtime {fp16|mixed} --profile-seq-len 8 --profile-decode-steps 4
```

After two warmup iterations, the measured NVTX ranges were one
`A1_prefill_S8` execution and four `A1_decode_step_N` executions. Only the
first decode step is used in the main attribution table; the other steps were
retained in exports for consistency inspection. Raw `.nsys-rep` and SQLite
files remain Jetson-local under `/tmp/phase3a_nsys_20260904T063649Z/` as
`fp16.nsys-rep`, `fp16.sqlite`, `mixed_full.nsys-rep` and
`mixed_full.sqlite`. A small stale `mixed.nsys-rep` diagnostic from the output
path collision is also preserved; the valid full report is `mixed_full`.
Git contains only compact exported CSV/JSON summaries.

The `nvtx_gpu_proj_sum` projection was not used as GPU busy time because it
reflected wall time in this runtime topology. Attribution uses
`nvtx_kern_sum` for kernels and `cuda_api_sum` filtered by NVTX range for CUDA
APIs. Host remainder is computed as NVTX wall minus kernel time minus CUDA API
time and is therefore a bounded accounting residual, not a direct system-wide
CPU counter.

## 5. A2 Range Summary

Source: `analysis/a2_range_summary.csv`.

| Runtime / range | Wall ms | GPU kernel ms (count) | GPU busy | CUDA API ms (calls) | Host remainder ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 prefill S8 | 1571.1 | 47.2 (396) | 3.00% | 357.1 (4068) | 1166.8 |
| Mixed prefill S8 | 2311.5 | 42.7 (448) | 1.85% | 907.2 (4528) | 1361.6 |
| FP16 decode step 0 | 2131.4 | 48.9 (422) | 2.29% | 295.3 (2514) | 1787.3 |
| Mixed decode step 0 | 2944.8 | 43.0 (501) | 1.46% | 975.4 (4516) | 1926.5 |

| Runtime / range | `cuModuleLoadData` ms | `cuModuleUnload` ms | Module load+unload ms | `cudaStreamSynchronize` ms | `cuMemcpyDtoH` ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 prefill | 194.3 (310) | 113.2 (310) | 307.5 | 34.6 (13) | 3.0 (56) |
| Mixed prefill | 551.5 (416) | 321.3 (416) | 872.8 | 14.5 (13) | 8.3 (164) |
| FP16 decode | 154.9 (122) | 91.5 (126) | 246.4 | 33.3 (14) | 2.9 (56) |
| Mixed decode | 615.4 (281) | 324.8 (285) | 940.2 | 12.3 (14) | 8.3 (165) |

The Mixed wall increase was +740.4 ms at prefill and +813.4 ms at decode. The
CUDA API time increase was +550.1 ms and +680.1 ms respectively. Within that
API delta, module load plus unload accounted for +565.3 ms and +693.8 ms.
Synchronization and DtoH memcpy time were small in both runtimes.

## 6. A3 Kernel / API Attribution

Source: `analysis/a3_kernel_attribution.csv`.

| Workload | Category | FP16 count/time | Mixed count/time | Delta |
| --- | --- | ---: | ---: | ---: |
| Prefill | FP16 TRT GEMM | 141 / 43.28 ms | 62 / 25.40 ms | -79 / -17.88 ms |
| Prefill | INT8 TRT GEMM | 0 / 0 ms | 133 / 13.22 ms | +133 / +13.22 ms |
| Prefill | FP16 attention GEMM | 28 / 0.62 ms | 28 / 0.73 ms | 0 / +0.11 ms |
| Prefill | Myelin fused, mapping unknown | 226 / 3.27 ms | 224 / 3.31 ms | -2 / +0.04 ms |
| Prefill | All CUDA APIs | 4068 / 357.1 ms | 4528 / 907.2 ms | +460 / +550.1 ms |
| Prefill | Module load+unload | 620 / 307.5 ms | 832 / 872.8 ms | +212 / +565.3 ms |
| Decode | FP16 TRT GEMM | 197 / 45.79 ms | 118 / 26.72 ms | -79 / -19.07 ms |
| Decode | INT8 TRT GEMM | 0 / 0 ms | 133 / 12.62 ms | +133 / +12.62 ms |
| Decode | Myelin fused, mapping unknown | 225 / 3.12 ms | 250 / 3.63 ms | +25 / +0.51 ms |
| Decode | All CUDA APIs | 2514 / 295.3 ms | 4516 / 975.4 ms | +2002 / +680.1 ms |
| Decode | Module load+unload | 248 / 246.4 ms | 566 / 940.2 ms | +318 / +693.8 ms |

Kernel names are aggregated by conservative prefix matching. `__myl_*` remains
`TRT_MYELIN_FUSED_UNKNOWN`; it is not mapped to Q/DQ, RMSNorm, Softmax, RoPE or
any other high-level operator.

## 7. A5 Attribution and Evidence Levels

- `Mixed slowdown reproduced`: `PROVEN` by A1 and prior Phase 2.3-F.
- `Runtime is host-bound under this workload`: `PROVEN` by measured GPU kernel
  time being only 1.46-3.00% of NVTX wall time in both runtimes.
- `Mixed GPU kernel time is not higher than FP16`: `PROVEN` for the measured
  ranges: Mixed kernel totals were 42.7 versus 47.2 ms at prefill and 43.0
  versus 48.9 ms at decode.
- `Mixed slowdown delta is dominated by host-side CUDA API cost`: `PROVEN` as
  measured attribution: API delta was +550.1/+680.1 ms while wall delta was
  +740.4/+813.4 ms.
- `Module load/unload is the dominant measured API contributor`: `PROVEN`
  within the profiler evidence: it explains +565.3/+693.8 ms of the measured
  API delta.
- `Mixed Q/DQ engines load more modules than FP16`: `SUPPORTED` by API counts
  and engine architecture; direct per-module ownership is not instrumented.
- `Per-call execution-context creation causes the module-load pattern`:
  `SUPPORTED`, not `PROVEN`. Code-path evidence shows the reused Phase 2.3-F
  `TRT.run` creates an execution context on every call. A dedicated A/B run
  with cached contexts was not performed because it would cross into Phase 3-B
  optimization.
- `Q/DQ boundary kernels are the direct dominant GPU cost`: `DISPROVEN` for
  the measured workload by the A2/A3 totals above.
- `INT8 GEMM itself is slower than FP16 GEMM`: `DISPROVEN` for this small
  batch-1 shape. INT8 GEMM added only 13.22/12.62 ms and total Mixed GEMM time
  was lower than FP16 GEMM time.
- `Exact high-level operator mapping of Myelin kernels`: `UNKNOWN`.
- `GPU idle-gap distribution and complete CPU scheduling decomposition`:
  `INCONCLUSIVE`; the host remainder is an accounting residual and this run did
  not collect OS/thread sampling.

## 8. Nsight Compute Decision

`NCU = NOT REQUIRED`, bounded by the Phase 3-A scope rule. Nsight Systems
already showed that GPU kernels occupied only 1.5-3.0% of wall and that the
Mixed slowdown delta was host-side API/module time, not GPU compute time.
Per-kernel NCU metrics such as achieved occupancy, SM throughput, memory
throughput, Tensor Core utilization or stalls would not change the dominant
runtime attribution. No NCU report was generated.

## 9. Root-Cause Ranking and Phase 3-B Recommendation

Priority 1: persistent TensorRT execution contexts / eliminate the per-call
module-load storm. Expected mechanism: create each engine's execution context
once and reuse it across prefill, decode and pipeline calls. Measurement
target: A1 median prefill/TPOT plus an NSYS check that per-range
`cuModuleLoadData/Unload` calls fall to near zero. Integration feasibility is
high because it changes runtime object lifetime rather than quantization
policy or kernels. End-to-end relevance is high because both prefill and
decode show the same structure.

Priority 2: characterize remaining host runtime scheduling after removing the
module-load storm. Even a perfect fix cannot explain all Mixed wall time; the
GPU remains mostly idle. A post-fix profile should determine whether remaining
gaps are enqueue scheduling, host logic, default-stream synchronization or
allocator effects. This is a measurement/control experiment, not kernel work.

Priority 3: mixed-engine structural cost, including additional modules and
DtoH copies. It is real but much smaller than the measured API delta.

Not justified yet: RMSNorm, Softmax, attention, small-GEMM or other CUDA kernel
optimization. GPU busy time is only 1.46-3.00%, Mixed kernel time is not higher
than FP16, and there is no NCU evidence that any high-level operator is the
end-to-end bottleneck. CUDA Graphs are also premature until the context
lifetime issue is fixed and remeasured.

## 10. Gate

`PASS / BOUNDED`

The slowdown was reproduced, comparable NSYS timelines were collected, the
dominant contributor is identified with strong evidence, and Phase 3-B has an
evidence-backed target. The boundary is that the Myelin kernel mapping remains
unknown, OS-level host sampling was not collected, and the causal link from
per-call context creation to module loading is code-path-supported rather than
proven by a context-cache A/B.

## 11. Reproduction

From the Jetson repository root, with the Phase 2.3-E engines present:

```bash
python3 experiments/Phase2-qwen3-quantization/src/phase3a/phase3a_runtime_profile.py \
  --mode bench --out results/phase3a_runtime_attribution/<new-timestamp> \
  --warmup 5 --repeats 10 --window-repeats 3 --decode-steps 8

nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/phase3a_nsys_fp16_<timestamp> \
  python3 experiments/Phase2-qwen3-quantization/src/phase3a/phase3a_runtime_profile.py \
  --mode profile --out /tmp/phase3a_profile_fp16_<timestamp> \
  --profile-runtime fp16 --profile-decode-steps 4

nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/phase3a_nsys_mixed_<timestamp> \
  python3 experiments/Phase2-qwen3-quantization/src/phase3a/phase3a_runtime_profile.py \
  --mode profile --out /tmp/phase3a_profile_mixed_<timestamp> \
  --profile-runtime mixed --profile-decode-steps 4
```

Export the NSYS reports, then run:

```bash
python3 experiments/Phase2-qwen3-quantization/src/phase3a/analyze_nsys.py \
  --nsys-dir <exported-directory> --out <exported-directory>/analysis
```

Use a unique `-o` path for each NSYS invocation. The Mixed run must not reuse
an existing output name.

## 12. Artifact and Failure Diagnostics

Primary A1 artifacts: `results/phase3a_runtime_attribution/20260904T062712Z/`.

Primary A2/A3 artifacts:
`results/phase3a_runtime_attribution/20260904T063649Z_nsys/`, including
`analysis/a2_range_summary.csv`, `analysis/a3_kernel_attribution.csv`,
`analysis/a2_a3_detail.json`, exported `stats/`, `stats2/` and `stats3/`
CSVs, environment snapshots and profile manifests.

Failed/diagnostic runs were retained Jetson-side under the untracked result
paths `results/phase3a_runtime_attribution/20260904T062257Z/` and
`results/phase3a_runtime_attribution/20260904T063612Z_nsys/`. The completed
Mixed NSYS report is
`/tmp/phase3a_nsys_20260904T063649Z/mixed_full.nsys-rep`; the small stale
`mixed.nsys-rep` from an output-path collision remains as a diagnostic.

During Jetson synchronization, Git protected two same-name CRLF working-tree
CSVs. Their numeric content was identical to the committed LF versions. The
original CRLF bytes were copied to
`/tmp/phase3a_e3db_pre_sync_20260904T070449Z/`, then only line endings of the
two working-tree CSVs were normalized to complete the fast-forward. No
measured value was changed.
