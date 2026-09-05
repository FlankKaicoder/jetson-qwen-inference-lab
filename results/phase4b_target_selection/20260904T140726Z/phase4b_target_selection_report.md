# Phase 4-B CUDA Optimization Target Selection Report

## Environment

| Field | Value |
| --- | --- |
| Device evidence | Jetson Orin Nano Engineering Reference Developer Kit Super, SM 8.7 |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| Nsight Compute | 2024.3.1.0 |
| Batch / decode mode | batch 1, Mixed Decode engine |
| Branch | `phase/04a-tensorrt-operator-attribution-recovery` |
| Starting HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |
| Analysis host | Windows repository checkout |
| Analysis mode | Read-only metadata aggregation; no Jetson execution, benchmark, or profiling |

The Mixed Decode engine referenced by prior evidence is
`/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine`
(`650,285,868` bytes, SHA-256
`445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c`).
The Mixed Decode ONNX SHA-256 is
`8dccf65451c548762896280b498358069eaba100824b0acb92b7cdc9e4c51d9a`.
These identities were inherited from Phase 4-A evidence; this phase did not
load, rebuild, or execute the engine.

## Evidence Sources

| Source | Artifact | Evidence used |
| --- | --- | --- |
| Phase 3-C / 3-E steady-state kernel aggregation | `results/phase3e_kernel_attribution/20260904T121007Z/analysis/analysis_summary.json` | Mixed total kernel time `147.830560 ms`; h16816 top-kernel time `39.590976 ms`; share `26.781320%`; recorded `operator_match=gate_proj;up_proj` |
| Phase 3-E targeted NCU | `results/phase3e_kernel_attribution/20260904T121007Z/analysis/e3_ncu_summary.json` | h16816 rank1: memory/L2 `97.06%`, SM `39.57%`, HMMA active `39.673148%`, achieved occupancy `24.78%`, direct DRAM `N/A` |
| Phase 4-A.1 layer-to-operator mapping | `results/phase4a_operator_attribution/20260904T132820Z/gemm_candidates.csv` and report | 250 GEMM rows; `HIGH=194`, `MEDIUM=56`; all seven named projections represented by 28 HIGH rows each |
| Phase 4-A.2 runtime correlation | `results/phase4a_operator_attribution/20260904T134556Z/runtime_nvtx_kernel_mapping.csv` and report | 120 mapping rows across 86 ranges; `HIGH=63`, `MEDIUM=57`; 28 HIGH `up_proj` layers and one HIGH fused `k/q/v` range |
| Phase 4-B candidate summary | `candidate_operator_summary.json` | Machine-readable per-operator counts, precision, tactic family, shape, and mapped kernel summary |

No Phase 3-E, Phase 4-A.0, Phase 4-A.1, or Phase 4-A.2 artifact was regenerated
or modified. No new benchmark or profiler run was made.

## Candidate Operators

The complete table is in [candidate_ranking.csv](candidate_ranking.csv). The
scoring rule is in [scoring_rubric.json](scoring_rubric.json).

| Operator | Attribution | Shape | Phase 4-A.2 runtime evidence | Direct NCU evidence |
| --- | --- | --- | --- | --- |
| `up_proj` | 28/28 `HIGH` | `M=1,K=1024,N=3072` | 56 rows / 28 layers / `37.554112 ms` all-trace | Yes, through Phase 3-E h16816 rank1 |
| `gate_proj` | 28/28 `HIGH` | `M=1,K=1024,N=3072` | 1 row / 1 layer / `0.648512 ms` all-trace | Yes, through Phase 3-E h16816 rank1 |
| `q_proj` | 28/28 `HIGH` | `M=1,K=1024,N=2048` (27 rows), one `N=4096` | 2 rows / 1 fused layer / `1.480960 ms` all-trace shared with k/v | No |
| `k_proj` | 28/28 `HIGH` | `M=1,K=1024,N=1024` (27 rows), one `N=4096` | 2 rows / 1 fused layer / `1.480960 ms` all-trace shared with q/v | No |
| `v_proj` | 28/28 `HIGH` | `M=1,K=1024,N=1024` (27 rows), one `N=4096` | 2 rows / 1 fused layer / `1.480960 ms` all-trace shared with q/k | No |
| `o_proj` | 28/28 `HIGH` | `M=1,K=2048,N=1024` | None | No |
| `down_proj` | 28/28 `HIGH` | `M=1,K=3072,N=1024` | None | No |
| `unknown_attention_matmul` | 56/56 `MEDIUM`, operator `UNKNOWN` | Dynamic | 57 rows / 56 layers / `61.815776 ms` all-trace | No |

The seven projections all have Tensor Core tactic-family evidence in the
inspector record. That fact is not used to infer operator identity.

## Attribution Coverage

The known-operator evidence set has 8 candidates. Seven named projections have
28 `HIGH` layer-to-operator mappings each. The two unnamed attention MatMul
paths produce 56 `MEDIUM` rows whose Transformer operator remains `UNKNOWN`.

Runtime attribution is partial:

| Evidence class | Rows | Confidence | Interpretation |
| --- | ---: | ---: | --- |
| `up_proj` | 56 | `HIGH=56` | 28 execution layers, all with the full runtime-to-operator chain |
| `unknown_attention_matmul` | 57 | `MEDIUM=57` | Large all-trace time but operator identity remains `UNKNOWN` |
| fused `q/k/v` | 2 rows per operator | `HIGH` | One execution range viewed from three operators; wall-clock time is shared |
| `gate_proj` | 1 | `HIGH` | Only one mapped execution layer |
| `o_proj`, `down_proj` | 0 | Not available | No mapped runtime row in the committed Phase 4-A.2 subset |

The Phase 3-E mixed h16816 kernel is associated with
`gate_proj;up_proj`, but its `39.590976 ms` steady-state time cannot be split
between those two candidates.

## Ranking Methodology

Scoring is ordinal and evidence-conditioned, not a subjective continuous model:

1. **Evidence strength (0-2):** 2 requires at least one `HIGH`
   `kernel -> NVTX range -> ONNX node -> EngineInspector metadata -> TensorRT
   layer -> operator` row; 1 is HIGH layer-to-operator without runtime mapping;
   0.5 is MEDIUM only; 0 is no known operator.
2. **Runtime importance (0-3):** 3 requires both Phase 4-A.2 HIGH mapping and
   explicit membership in the Phase 3-E mixed top-kernel `operator_match` set;
   2 requires one of those two conditions; 0 requires neither.
3. **Optimization feasibility (0-4):** one point each for GEMM layer type,
   static decode shape, Tensor Core tactic family, and direct NCU metrics for a
   mapped or explicitly shared kernel.

The total is their sum. Tactic and kernel names were never used to infer an
operator. All-trace Phase 4-A.2 time is not treated as steady-state runtime
share, and shared or fused kernel time is not split or triple-counted.

## Target Ranking

| Rank | Operator | Evidence / runtime / feasibility | Total | Reason |
| --- | --- | --- | ---: | --- |
| 1 | `up_proj` | 2 / 3 / 4 | 9.0 | Strongest complete runtime-to-operator chain, plus static GEMM shape and rank1 NCU metrics |
| 2 | `gate_proj` | 2 / 2 / 4 | 8.0 | Same Phase 3-E GEMM family, but only one HIGH runtime range |
| 3 | `q_proj`, `k_proj`, `v_proj` | 2 / 2 / 3 | 7.0 | HIGH attribution and one fused HIGH runtime range, but only one covered layer and no direct NCU evidence |
| 5 | `unknown_attention_matmul` | 0.5 / 2 / 2 | 4.5 | Largest all-trace mapped time, but operator remains UNKNOWN and shapes are dynamic |
| 6 | `o_proj`, `down_proj` | 1 / 0 / 3 | 4.0 | HIGH layer attribution and static shapes, but no mapped runtime evidence |

Ranks 3 and 6 are ties. No tie-break beyond the stated score is introduced.

## Recommended First Optimization Target

The first optimization-research target is the **`up_proj` FP16 GEMM path**,
specifically the mapped GEMM family that includes the Phase 3-E
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1` kernel.

The supported shape is `M=1,K=1024,N=3072` in FP16 decode. The evidence chain
is:

```text
Phase 3-E NSYS top kernel and NCU metrics
    -> h16816 GEMM
    -> Phase 4-A.2 HIGH NVTX/ONNX/Inspector mapping
    -> 28 up_proj execution layers
    -> static M=1,K=1024,N=3072 GEMM
```

This is a **research target**, not a proven optimization. No speedup, kernel
improvement, or implementability claim is made. Phase 3-E already shows that
the h16816 kernel is memory/L2-heavy (`97.06%`) with relatively low HMMA
activity (`39.673148%`) and achieved occupancy (`24.78%`); direct DRAM
throughput is `N/A` and must not be inferred.

## Why Not Other Candidates

- `gate_proj` shares the strongest Phase 3-E kernel and has an identical
  static shape, but Phase 4-A.2 provides only one HIGH mapped range versus 28
  HIGH `up_proj` layers. The Phase 3-E time cannot be split between them.
- `unknown_attention_matmul` has the largest all-trace mapped time
  (`61.815776 ms`), but attribution is only `MEDIUM`, the Transformer operator
  is `UNKNOWN`, and shapes are dynamic. Selecting it would violate the
  evidence-first target rule.
- `q_proj`, `k_proj`, and `v_proj` have complete layer attribution but only one
  fused runtime range. Their `1.480960 ms` all-trace time is shared and cannot
  be triple-counted, and there are no direct NCU metrics for that path.
- `o_proj` and `down_proj` have strong layer-to-operator attribution and static
  GEMM shapes, but the committed runtime subset contains no mapped runtime rows
  or direct NCU evidence.

## Remaining Risks

- Phase 4-A.2 time is all-trace evidence, not a steady-state share.
- Phase 3-E h16816 time is shared by `gate_proj` and `up_proj`.
- The committed NSYS evidence is derived summaries, not raw timelines.
- Runtime ranges cover only a subset of the 699 EngineInspector layers.
- The h16816 path uses four mapped kernel families in Phase 4-A.2; target
  selection names a GEMM path, not one universally present kernel.
- The rank1 NCU result is memory/L2-limited rather than compute-limited, so a
  new compute kernel may not be the right intervention.
- No baseline correctness, power, memory, or latency experiment exists for a
  proposed replacement.
- Direct DRAM throughput is `N/A`; no DRAM utilization estimate is introduced.

## Gate

**Phase 4-B: `PASS / BOUNDED`**

The gate is `PASS` because the required evidence exists to select a first
research target. It is `BOUNDED` because mapping coverage is partial, shared
kernel time cannot be split, all-trace time is not steady-state time, and no
optimization has been attempted or proven.

## Recommendation

Stop after this selection. The next authorized step, if approved, should be a
narrow `up_proj` evidence-refinement and baseline-design study: define the
exact kernel subset, steady-state measurement boundary, correctness oracle,
and success criteria before any CUDA kernel or TensorRT Plugin implementation.
No implementation work should start from this report alone.
