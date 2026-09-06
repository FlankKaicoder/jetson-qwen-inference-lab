# Phase 6-F Post-Reproduction Target Ranking Report

## Executive Summary

Phase 6-F is an offline evidence synthesis. It re-ranks the remaining
optimization candidates after Phase 6-E and selects one bounded next
attribution target. It runs no inference, benchmark, Nsys, NCU, engine build,
ONNX export, precision change, or implementation.

The final gate is:

```text
NEXT_ATTRIBUTION_TARGET_RECOVERED
```

The selected target is **Attention x V across 28 layers** with classification
`ATTRIBUTION_ONLY`. The exact recommended next experiment is a read-only
representative-boundary AV runtime/kernel attribution query against the
existing Phase 3-C raw Nsys SQLite. No new profiling is recommended.

Layer-0 decode QK^T h16816 remains the numerically highest active candidate at
score 6, tied with AV, but its `NO_CURRENT_ACTION` classification overrides
eligibility. Its score falls from 8 to 6 because Phase 6-E did not reproduce
the historical h16816 path and left workload equivalence, runtime shape,
operation identity, and trigger unresolved.

## Evidence Boundary

The repository hierarchy remains Git, raw result artifacts, experiment report,
`docs/PROJECT_STATE.md`, then conversation. All performance values in this
report retain their original boundaries.

The key Phase 3-C denominator is `230,950,048 ns`, the Mixed persistent
all-trace GPU kernel aggregate. The direct AV and QK sums below are therefore
**all-trace** aggregates. They are not steady-state shares, CUDA Event times,
IProfiler layer times, or NCU durations. Phase 6-B's representative steady
layer-0 QK values use a separately stated boundary and must not be merged with
the all-trace AV sums.

From
`results/phase6a_unknown_attention_matmul_attribution/20260906T040500Z/contribution_reconciliation.csv`:

| Family | Rows | Calls | Direct all-trace total | Share of 230,950,048 ns |
| --- | ---: | ---: | ---: | ---: |
| Attention x V, layers 0-27 | 28 | 112 | `4,237,568 ns` | `1.834842%` |
| QK^T layers 1-27 | 27 | 108 | `2,518,016 ns` | `1.090286%` |

Layer-0 QK is excluded from the QK 1-27 sum. Its historical h16816 anomaly is
separately `54,984,352 ns` all-trace over seven launches and
`35,951,296 ns / 24.319258%` over five representative steady launches. These
numbers remain real but do not establish the current path or workload.

## Frozen Scoring Model

Phase 6-F reuses the Phase 6-D formula exactly:

```text
positive = R + A + S + B + F
final    = positive - U
```

All dimensions remain integers `0-3`. No weight or reproducibility dimension is
added. Phase 6-E evidence is incorporated through the existing dimensions.
For layer-0 QK, `F` decreases from `1` to `0` and `U` increases from `2` to
`3`. For AV and QK layers 1-27, newly checked direct-row aggregates make a
bounded offline attribution query plausible, so `F` increases from `1` to `2`.

The frozen tie-break is higher `R`, then higher `A`, then lower `U`, then lower
candidate ID. Classification overrides raw score.

## Phase 6-E Impact On Layer-0 QK

Phase 6-E is the decisive negative result:

1. The controlled frozen-engine run used the historical Mixed stack, persistent
   contexts, sample `eva_025`, prefill sequence length 8, and decode steps 0-3.
2. Direct cache progression was `[1,8,8,128]`, `[1,8,9,128]`,
   `[1,8,10,128]`, and `[1,8,11,128]` across decode steps 0-3.
3. All four controlled layer-0 `/MatMul` invocations correlated to one
   `sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_..._execute_kernel_trt` launch each.
4. Zero `trt_ampere_h16816gemm_128x64_ldg8_nn_v1` launches were observed.

The correct interpretation is bounded:

```text
historical h16816 behavior was real
but was not reproduced under the Phase 6-E controlled configuration
```

It is **not** proven that h16816 no longer exists, that the historical profiler
was wrong, that TensorRT automatically fixed a defect, or that xmma replaced
h16816 for the identical workload. Historical versus controlled workload is
`UNKNOWN`; the trigger is `UNKNOWN`; normalized performance is
`NOT_CALCULATED`.

Therefore layer-0 QK keeps its historical runtime significance and HIGH
semantic attribution, but loses actionability. It is downgraded from Phase 6-D
`ATTRIBUTION_ONLY` to `NO_CURRENT_ACTION`. The Phase 6-E stop-loss rule forbids
automatically creating another expanding Attention attribution phase for this
candidate.

## Before-vs-After Ranking

| Phase 6-D rank | Candidate | Old final | Phase 6-F rank | Candidate | New final | New classification |
| ---: | --- | ---: | ---: | --- | ---: | --- |
| 1 | Layer-0 decode QK^T h16816 | 8 | 1 | Layer-0 decode QK^T h16816 | 6 | `NO_CURRENT_ACTION` |
| 2 | Attention x V, 28 layers | 4 | 2 | Attention x V, 28 layers | 6 | `ATTRIBUTION_ONLY` |
| 3 | QK^T layers 1-27 static xmma | 4 | 3 | QK^T layers 1-27 static xmma | 5 | `ATTRIBUTION_ONLY` |
| 4 | Fused q/k/v | 4 | 4 | Fused q/k/v | 4 | `ATTRIBUTION_ONLY` |
| 5 | down_proj / o_proj | 4 | 6 | down_proj / o_proj | 4 | `NO_CURRENT_ACTION` |
| 6 | gate_proj shared-h16816 | 4 | 5 | gate_proj shared-h16816 | 4 | `ATTRIBUTION_ONLY` |
| 7 | TensorRT internal `__myl_*` | 1 | 7 | TensorRT internal `__myl_*` | 1 | `ATTRIBUTION_ONLY` |
| 8 | RMSNorm / RoPE | -1 | 8 | RMSNorm / RoPE | -1 | `NO_CURRENT_ACTION` |
| not ranked | CUDA Graph current prototype | not ranked | 9 | CUDA Graph current prototype | -2 | `NO_CURRENT_ACTION` |
| excluded | up_proj | 8 | `CLOSED_FOR_NOW` | up_proj | 8 | `CLOSED_FOR_NOW` |

Rank-change explanations:

- **Layer-0 QK, rank 1 to 1:** ordinal position is unchanged by the frozen
  tie-break, but the classification changes to `NO_CURRENT_ACTION`. This is the
  central Phase 6-F correction.
- **AV, rank 2 to 2:** ordinal rank is unchanged, but it becomes the highest
  eligible target once layer-0 QK is blocked. Direct all-trace AV evidence and
  HIGH semantic identity raise its score from 4 to 6.
- **QK layers 1-27, rank 3 to 3:** direct all-trace evidence raises the score
  from 4 to 5, but lower attribution confidence keeps it below AV.
- **Fused q/k/v, rank 4 to 4:** no new evidence; the shared single-layer range
  remains bounded.
- **gate_proj, rank 6 to 5:** it moves above down_proj/o_proj under the frozen
  tie-break because its runtime evidence is stronger, despite shared h16816
  attribution.
- **down_proj/o_proj, rank 5 to 6:** the reciprocal tie-break result of the
  gate_proj change; it still has no mapped runtime rows.
- **`__myl_*`, rank 7 to 7:** unchanged. The meaningful name-based aggregate
  has no semantic identity or clean surface.
- **RMSNorm/RoPE, rank 8 to 8:** unchanged. No positive performance evidence.
- **CUDA Graph current prototype, new rank 9:** retained for inventory
  completeness from Phase 3-D0. Its captured graph omitted TensorRT kernels and
  failed validation, so it is not a target.
- **up_proj:** remains excluded. Phase 5 closure overrides the raw score of 8.

## Mandatory Top-3 Review

### 1. Attention x V Across 28 Layers — Selected Eligible Target

- **Why investigate:** it is now the highest eligible candidate and has one
  bounded attribution question.
- **Runtime evidence:** direct AV rows total `4,237,568 ns`, or
  `1.834842%` of the Phase 3-C Mixed all-trace GPU kernel denominator. This is
  meaningful but bounded, not a steady-state claim.
- **Known semantics:** 28/28 decode `/MatMul_*` odd nodes are HIGH-confidence
  Attention x V. Prefill is fused into `_gemm_mha_v2_*`; decode is standalone
  metadata.
- **Optimization surface:** not clean. The exact runtime kernel mapping and
  steady-boundary contribution remain unresolved.
- **Strongest negative evidence:** prefill is already fused; one-to-one
  runtime kernel attribution is not proven; no headroom is proven.
- **Smallest question:** what are the representative-boundary AV runtime rows,
  immediate NVTX parents, correlated kernels, calls, and aggregate duration
  when AV is separated from QK and fusion boundaries?
- **Bounded next experiment:** one read-only attribution query on the existing
  Phase 3-C raw Nsys SQLite that correlates odd `/MatMul_*` AV ranges through
  runtime APIs to kernels, separates warmup from representative steady
  boundaries, and writes only derived CSV/Markdown evidence. No new profiling
  and no implementation.

### 2. QK^T Layers 1-27 Static Xmma Path — Attribution Only

- **Why review:** it has a bounded all-trace contribution and a sharply
  similar, but lower-confidence, attribution question.
- **Runtime evidence:** direct rows total `2,518,016 ns`, or `1.090286%`
  all-trace over 27 rows and 108 calls.
- **Known semantics:** static QK identity is HIGH for layers 1-27, with xmma
  tactic-class context.
- **Optimization surface:** not clean because runtime isolation and kernel
  mapping are not fully attributed.
- **Strongest negative evidence:** no defect evidence and no proven per-layer
  runtime equivalence or headroom.
- **Smallest question:** can layers 1-27 QK rows be separated into a
  representative-boundary runtime/kernel attribution using existing evidence?
- **Potential bounded experiment:** the same read-only SQLite query applied to
  even `/MatMul_*` rows excluding layer 0. It is not the selected next
  experiment because AV has stronger semantic attribution and a larger direct
  all-trace total.

### 3. Layer-0 Decode QK^T H16816 Path — No Current Action

- **Why review:** it remains historically significant and semantically well
  attributed, so its downgrade must be explicit rather than hidden.
- **Runtime evidence:** seven historical h16816 launches total
  `54,984,352 ns` all-trace; five steady launches total `35,951,296 ns`, or
  `24.319258%` of the representative steady boundary.
- **Known semantics:** layer-0 QK^T identity is HIGH and historical runtime
  ownership is HIGH.
- **Optimization surface:** not clean. Historical runtime shape, operation
  identity, replacement boundary, and trigger are unresolved.
- **Strongest negative evidence:** Phase 6-E observed four controlled xmma
  launches and zero h16816 launches, with same-workload classification
  `UNKNOWN`.
- **Smallest question:** what is the trigger or boundary condition that
  separates the historical h16816 path from the controlled xmma path?
- **Bounded next experiment:** `NONE` under the Phase 6-E stop-loss rule. An
  expanding trigger search is not authorized as the next phase.

## Gate Decision

The final gate is Gate B:

```text
NEXT_ATTRIBUTION_TARGET_RECOVERED
```

Gate A is not satisfied because AV still lacks clean runtime/kernel
attribution and a proven optimization surface. Gate C is not selected because
AV has a meaningful all-trace contribution, HIGH semantic identity, and one
bounded read-only question. Gate D is not selected because the evidence is
internally consistent once measurement boundaries are preserved.

The only recommended next experiment is the bounded AV attribution query
described above. It must receive explicit owner authorization and must not
begin during Phase 6-F.

## Authorization State

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
CUTLASS implementation: NOT AUTHORIZED
Engine rebuild: NOT AUTHORIZED
Precision change: NOT AUTHORIZED
Tactic forcing: NOT AUTHORIZED
```

No historical result was overwritten. The protected untracked directories
remain untouched.
