# Phase 3-E TensorRT Kernel Attribution & Compute Efficiency Audit

- Date: 2026-09-04
- Branch: `phase/03e-tensorrt-kernel-attribution`
- Starting checkpoint: `8b8a3530d9eed1e221419e41d24c9a4d8071cb02`
- Code commits: `8e34c85`, `7b20282`, `b6ce658`, `b39bfec`, `ab65133`
- Final Gate: `PASS / BOUNDED / NO_PROVEN_CUDA_OPTIMIZATION_TARGET`
- Classification: GEMM is the dominant named kernel category. Rank 1 is
  L2/memory-limited rather than SM-limited. Ranks 2/3 show low tensor-pipe
  activity on small decode GEMMs, but `PARTIALLY SUPPORTED` potential does not
  meet the phase's bar for a proven custom-kernel target.

## 1. Scope and Method

The authorized scope was kernel attribution and compute-efficiency audit only.
No CUDA kernel, TensorRT plugin, attention implementation, RMSNorm rewrite,
runtime redesign, or quantization change was made.

The audit reused the frozen Phase 2.2 FP16 engines and Phase 2.3-E mixed
engines and the Phase 3-C persistent-runtime NSYS evidence. TensorRT
EngineInspector output was joined with aggregate NSYS kernel totals. NCU was
triggered because one FP16 GEMM used 35.31% and the h16816 GEMM used 20.28% of
FP16 kernel time; the same h16816 GEMM used 26.78% of Mixed kernel time.

Primary artifact root:
`results/phase3e_kernel_attribution/20260904T121007Z/`.

## 2. Recovery Audit

- Starting branch: `phase/03d0-cuda-graph-feasibility`.
- Starting HEAD: `8b8a3530d9eed1e221419e41d24c9a4d8071cb02`, in sync with its
  origin branch.
- Windows tracked modifications: none.
- Preserved Windows untracked directory:
  `experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T203103Z/`.
- Jetson pre-existing untracked diagnostics were preserved.
- Required FP16 and Mixed engines existed and were not rebuilt.
- Nsight Compute was `/usr/local/cuda-12.6/bin/ncu`, version `2024.3.1.0`.

Gate E0: `PASS`.

## 3. Layer Attribution

Inspector output is canonical in `inspector_v2/`. Earlier `inspector/` remains
as historical evidence. Engine SHA-256 values and layer counts are recorded in
`inspector_v2/inspector_summary.json`.

| Engine | Detailed behavior | GEMM / attention-GEMM layers | Other layers |
| --- | --- | ---: | --- |
| FP16 prefill | Compact string layers only | 84 | 529 `UNKNOWN`; tactic/shape `UNKNOWN_NO_DETAILED_INSPECTOR` |
| FP16 decode | Compact string layers only | 84 | 336 `UNKNOWN`; tactic/shape `UNKNOWN_NO_DETAILED_INSPECTOR` |
| Mixed prefill | Detailed metadata | 194 GEMM + 28 attention GEMM | 205 `NO_KERNEL`, 222 TensorRT internal |
| Mixed decode | Detailed metadata | 250 GEMM | 201 `NO_KERNEL`, 248 TensorRT internal |

The detailed Mixed inspector maps decoder Linear layers to `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj`. Compact FP16
metadata permits operator-name matching where present, but not tactic or shape
recovery. Standalone RMSNorm and LM Head engines were outside this decoder
engine audit; their Phase 3-E kernel attribution remains `UNKNOWN`.

Gate E1: `PASS / BOUNDED`. Mapping is evidence-bounded; unknown mappings are
not inferred.

## 4. GEMM Shape Attribution

Representative detailed Mixed decode shapes are:

| Operator | M | N | K | Precision | Frequency |
| --- | ---: | ---: | ---: | --- | ---: |
| q_proj | 1 | 2048 | 1024 | INT8 | 27 |
| k_proj | 1 | 1024 | 1024 | INT8 | 27 |
| v_proj | 1 | 1024 | 1024 | INT8 | 27 |
| o_proj | 1 | 1024 | 2048 | INT8 / FP16 | 25 / 3 |
| gate_proj | 1 | 3072 | 1024 | INT8 / FP16 | 27 / 1 |
| up_proj | 1 | 3072 | 1024 | FP16 | 28 |
| down_proj | 1 | 1024 | 3072 | FP16 | 28 |

Representative detailed Mixed prefill shapes are:

| Operator | M | N | K | Precision | Frequency |
| --- | ---: | ---: | ---: | --- | ---: |
| q_proj | 8 | 2048 | 1024 | INT8 | 27 |
| k_proj | 8 | 1024 | 1024 | INT8 | 27 |
| v_proj | 8 | 1024 | 1024 | INT8 | 27 |
| gate_proj | 8 | 3072 | 1024 | INT8 / FP16 | 27 / 1 |
| up_proj | 8 | 3072 | 1024 | FP16 | 28 |
| down_proj | 8 | 1024 | 3072 | FP16 | 28 |
| o_proj | 8 | 1024 | 2048 | INT8 / FP16 | 25 / 3 |

For FP16 engines, M/N/K and tactic remain `UNKNOWN`; this is explicitly a
compact-inspector limitation, not an absent GEMM. Full rows and tactics are in
`analysis/e2_operator_shape_groups.csv`. Tactics are also shown in
`inspector_v2/gemm_shape_summary.csv`.

Gate E2: `PASS / BOUNDED`. Mixed shapes are supported; FP16 shapes remain
`UNKNOWN`.

## 5. NCU Selection and Decision

Selection used Phase 3-C aggregate kernel time:

| Rank | Kernel | FP16 ms | Mixed ms | Trigger evidence |
| --- | --- | ---: | ---: | --- |
| 1 | `trt_ampere_h16816gemm_128x64_ldg8_nn_v1` | 44.566752 | 39.590976 | 20.28% FP16 and 26.78% Mixed; mapped to `gate_proj;up_proj` in Mixed metadata |
| 2 | `sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt` | 77.583328 | 2.072032 | 35.31% FP16; operator mapping `UNKNOWN` |
| 3 | `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_fused` | 28.646528 | 2.437152 | 13.04% FP16; included as third unique GEMM; operator mapping `UNKNOWN` |

A prefill-only rank-2 filter found no matching kernel. That failed attempt is
retained as `ncu/rank2_f16f32.log`. A successful rank-2 retry used one decode
step. Raw `.ncu-rep` files remain Jetson-local under
`/tmp/phase3e_kernel_attribution_20260904T121007Z/ncu/`; compact CSV/log
evidence is committed.

NCU used `--clock-control none`, narrow kernel filters, and one launch per
target. Device clocks were not changed. The supplemental key-metrics file named
`three_gemm_key_metrics` profiles three rank-1 launches only; it does not cover
ranks 2/3 despite its file name. This scope is recorded in
`ncu/ncu_artifact_manifest.json`.

Gate E3 trigger: `PASS`.

## 6. NCU Results

The committed summary is
`analysis/e3_ncu_kernel_summary.csv` / `.json`.

| Rank | Duration | Memory/L2 | SM | Tensor/HMMA active | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7601.557 us (mean of 3) | 97.06% | 39.57% | 39.673148% | 24.78% |
| 2 | 184.032 us | 76.88% | 25.62% | 17.697509% | 24.25% |
| 3 | 155.648 us | 71.92% | 29.60% | 17.167166% | 24.68% |

All three kernels have 25% theoretical occupancy and achieved occupancy within
0.75 percentage points of that limit. Rank 1 uses 149 registers/thread and
24.73 waves/SM; ranks 2/3 use 54 registers/thread and 4 waves/SM. IMMA is zero
in all profiled FP16 GEMMs. Direct DRAM throughput is `N/A`; the reported
memory/L2 value must not be called DRAM throughput.

The highest recorded stall ratio for rank 1 is long scoreboard at
19.490065 per issue-active, followed by wait at 3.196035 and math-pipe
throttle at 0.831753. Ranks 2/3 also report long scoreboard as the highest of
these three captured ratios, but a full stall taxonomy is not available in the
compact section export and remains `UNKNOWN`.

Gate E3 result: `PASS / BOUNDED`.

## 7. Decision Analysis

Rank 1 is not a simple low-efficiency GEMM: L2/memory is 97.06% while SM is
39.57% and HMMA pipe active is about 39.67%. Its achieved occupancy is
essentially at the theoretical limit. A faster implementation would need to
reduce or reshape memory/L2 traffic rather than merely increase parallelism.

Ranks 2/3 are small M=1 decode GEMMs with low SM and HMMA utilization and only
71.92-76.88% memory/L2 throughput. This is consistent with a latency/shape
limited regime, but the FP16 inspector cannot map those kernels to specific
operators. Their 17.17-17.70% HMMA activity is therefore evidence of potential
optimization space, not a proven custom-kernel target.

The phase asks for large time plus low utilization plus a bad tactic/shape
before entering Phase 4. The first two conditions are observed for ranks 2/3;
the final condition is not fully established because tactic-to-operator and
runtime-cost attribution for compact FP16 metadata remains `UNKNOWN`.

## 8. Final Gate

`PASS / BOUNDED / NO_PROVEN_CUDA_OPTIMIZATION_TARGET`.

This is a bounded Outcome-A-like closeout, not a claim that every TensorRT
GEMM is optimal. The potential rank-2/rank-3 signal is retained, but it does
not justify an unscoped Phase 4 implementation by this evidence alone.

## 9. Remaining Bottleneck and Recommendation

The measured remaining kernel bottleneck is GEMM-dominated GPU work. Rank 1 is
L2/memory-limited; ranks 2/3 are underutilized small decode GEMMs. The
remaining non-GEMM work includes many TensorRT-internal `__myl_*` kernels whose
operator identity remains `UNKNOWN`.

If the owner later authorizes another study, the narrow next step would be a
Phase 4-A feasibility study only: recover rank-2/rank-3 operator/shape
attribution without changing kernels, then compare alternatives at the same
M/N/K. Do not start CUDA kernel implementation from this report alone.

## 10. Reproduction Commands

On Jetson:

```bash
cd /home/nvidia/projects/jetson-qwen-inference-lab
python3.10 experiments/Phase2-qwen3-quantization/src/phase3e/extract_engine_inspector.py \
  --engine fp16_prefill=/tmp/phase2_2b4_2_20260902T082326Z/prefill_28layer.engine \
  --engine fp16_decode=/tmp/phase2_2b4_2_20260902T082326Z/decode_28layer.engine \
  --engine mixed_prefill=/tmp/phase2_3e_20260904T020000Z/work/mixed_prefill_28layer.engine \
  --engine mixed_decode=/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine \
  --assume-m fp16_prefill=8 --assume-m mixed_prefill=8 \
  --assume-m fp16_decode=1 --assume-m mixed_decode=1 \
  --out results/phase3e_kernel_attribution/20260904T121007Z/inspector_v2

python3.10 experiments/Phase2-qwen3-quantization/src/phase3e/analyze_phase3e.py \
  --gemm-inventory results/phase3e_kernel_attribution/20260904T121007Z/inspector_v2/gemm_inventory.csv \
  --top-kernels results/phase3c_residual_runtime/20260904T093500Z_nsys/analysis/c3_top_kernels_by_name.csv \
  --aggregate-summary results/phase3c_residual_runtime/20260904T093500Z_nsys/analysis/c2_aggregate_summary.csv \
  --out results/phase3e_kernel_attribution/20260904T121007Z/analysis
```

NCU reproduction pattern:

```bash
/usr/local/cuda-12.6/bin/ncu \
  --clock-control none --target-processes all \
  --kernel-name regex:<exact-selected-kernel> \
  --launch-count 1 --csv --export <jetson-local-report> \
  python3.10 <persistent-runtime-profile-command>
```

The original profile launched the Phase 3-C persistent runtime workload with
batch 1, deterministic prompt IDs, S=8 prefill and forced deterministic decode
tokens. Reports were imported/exported to raw CSV after collection.
