# Phase 6-D Evidence-Corrected Target Ranking Report

## Executive Summary

Phase 6-D is evidence synthesis only. It uses the frozen scoring model defined
in `phase6d_plan.md`: `final = R + A + S + B + F - U`, with every dimension
bounded to 0-3 and no weights. No new Jetson execution, benchmark, profiling,
engine rebuild, ONNX change, precision change, or tactic forcing occurred.

The final gate is:

```text
NEXT_ATTRIBUTION_TARGET_RECOVERED
```

The corrected top candidate is the layer-0 decode QK^T h16816 path. It has
strong runtime and semantic-attribution evidence, but its optimization surface
is not clean. Runtime Q/K/output shapes, h16816-versus-xmma workload identity,
same engine invocation, and path trigger remain unresolved. Therefore the next
step is attribution only, not CUDA, FlashAttention, a TensorRT Plugin, or any
implementation.

## Phase 5 Before Ranking

The Phase 5 ranking used a different scoring model and different evidence
state. Its numeric scores are not directly comparable to Phase 6-D scores.

| Old rank | Candidate | Old score | Phase 5 interpretation |
| ---: | --- | ---: | --- |
| 1 | `unknown_attention_matmul` (`/MatMul_*`) | 6.5 | Attribution and feasibility study only |
| 2 | `gate_proj` shared-h16816 | 6.0 | Not an implementation target |
| 3 | fused `q_proj;k_proj;v_proj` | 4.0 | Bounded follow-up only |
| 4 | TensorRT internal non-GEMM `__myl_*` | 3.5 | Attribution research only |
| 5 | `down_proj` / `o_proj` | 3.5 | Not selected |
| CLOSED_FOR_NOW | `up_proj` | not ranked | No proven CUDA GEMM target or tactic defect |

## Corrected Ranking

Active ranking follows the frozen score and tie-break order. The `up_proj`
score is shown in `candidate_scoring.csv` for transparency but is excluded
because Phase 5 closed it. RMSNorm / RoPE are also excluded from active
selection.

| New rank | Candidate | R | A | S | B | F | U | Final | Classification |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Layer-0 decode QK^T h16816 path | 3 | 3 | 1 | 2 | 1 | 2 | 8 | ATTRIBUTION_ONLY |
| 2 | Attention x V across 28 layers | 1 | 3 | 1 | 1 | 1 | 2 | 4 | ATTRIBUTION_ONLY |
| 3 | QK^T layers 1-27 static xmma path | 1 | 2 | 1 | 1 | 1 | 2 | 4 | ATTRIBUTION_ONLY |
| 4 | Fused `q_proj;k_proj;v_proj` | 1 | 2 | 1 | 1 | 1 | 2 | 4 | ATTRIBUTION_ONLY |
| 5 | `down_proj` / `o_proj` | 0 | 2 | 2 | 1 | 1 | 2 | 4 | NO_CURRENT_ACTION |
| 6 | `gate_proj` bounded shared-h16816 path | 1 | 2 | 1 | 2 | 1 | 3 | 4 | ATTRIBUTION_ONLY |
| 7 | TensorRT internal non-GEMM `__myl_*` | 2 | 0 | 0 | 1 | 1 | 3 | 1 | ATTRIBUTION_ONLY |
| 8 | RMSNorm / RoPE | 0 | 0 | 0 | 1 | 1 | 3 | -1 | NO_CURRENT_ACTION |
| CLOSED_FOR_NOW | `up_proj` | 3 | 3 | 3 | 0 | 0 | 1 | 8 | not active |

## Why The Top Ranking Changed

### `unknown_attention_matmul` To Layer-0 QK^T

Phase 5 only knew a medium-confidence `/MatMul_*` chain. Phase 6-A recovered
56 semantic candidates: 28 QK^T and 28 Attention x V, one pair per decoder
layer, with HIGH ONNX semantic identity. Phase 6-B then proved that the
suspicious layer-0 `/MatMul` row is not a duplicate aggregate: seven launches
sum to 54,984,352 ns all-trace, and five representative-steady launches sum to
35,951,296 ns, or 24.319258% of the 147,830,560 ns steady denominator.

This increases runtime significance and attribution confidence. It does not
make the candidate a feasibility winner. Phase 6-C found eleven exact `/MatMul`
launches: seven h16816 and four xmma. The immediate NVTX parents differ, while
same graph region, same engine invocation, and same mathematical workload are
`NOT_PROVEN`. Runtime shapes are `UNKNOWN`, normalized performance is
`NOT_CALCULATED`, and the path trigger is `UNKNOWN`. Consequently `S=1` and
`U=2`; the classification is `ATTRIBUTION_ONLY`.

### Attention x V And QK Layers 1-27

Semantic identity is now HIGH for the full 56-candidate family, but clean
runtime significance is weaker than the old all-trace family number suggested.
The tactic-consistent family subset is 6.831424 ms, or 2.957966%, on the
230.950048 ms all-trace denominator. AV-only and QK-only layer-isolated
contributions remain `UNKNOWN`. Prefill is fused into `_gemm_mha_v2_*`, while
decode metadata is standalone. These facts keep both candidates below the
direct layer-0 evidence and prevent a feasibility claim.

### `gate_proj`

`gate_proj` drops from old rank 2 because its runtime evidence remains one
direct row, 0.648512 ms all-trace, and the relevant h16816 family aggregate
cannot be cleanly assigned to `gate_proj`. Phase 5 also provides no proven GEMM
defect. A high Phase 3-E memory/L2 metric on a shared kernel context is not a
pure operator contribution.

### Fused Q/K/V

The fused projection remains one shared candidate with 4.442880 ms all-trace
for one observed layer. It is not triple-counted as three runtime candidates.
Coverage, per-projection share, and headroom remain `UNKNOWN`.

### TensorRT Internal `__myl_*`

The Mixed all-trace category remains large at 13.927392 ms, but operator
identity, kernel identity, workload, and isolation are unknown. Runtime size
therefore produces `R=2` only, not a target selection.

### `down_proj` / `o_proj`

Static mappings are useful, but the reviewed runtime artifact contains no
mapped rows. Static shape clarity cannot become runtime significance.

### `up_proj`

`up_proj` remains `CLOSED_FOR_NOW`. Phase 5 records cuBLASLt median
80.077961 us, CUTLASS best 83.619133 us, historical TensorRT steady median
147.424 us, and matched NCU durations of 242.912 us versus 244.160 us. There is
no proven tactic defect or CUDA target. Phase 6-D introduces no new evidence
that reopens it.

### RMSNorm / RoPE

Phase 3-C recorded 0 ms under the name-based categories. That gives no current
runtime optimization surface and is not overridden by numerical diagnostic
evidence from Phase 2.2.

## Qualitative Control

Although layer-0 QK^T and `up_proj` both score 8, the score does not select
`up_proj`. Phase 5 is a higher-priority experiment evidence chain: `B=0` and
`F=0` explicitly encode that the current implementation is competitive and the
CUTLASS replacement already lost. The active-ranking exclusion and qualitative
review therefore preserve the frozen Phase 5 closure.

Layer-0 QK^T is selected only as the next attribution target. Its exact runtime
surface is unresolved, and the raw duration cannot be interpreted as a defect.

## Next Attribution Study

One bounded follow-up is recommended, subject to explicit owner authorization:

**Phase 6-E: Frozen-Engine Layer-0 QK Runtime Shape And Invocation Attribution**

Objective:

1. Recover actual runtime Q, K, and output shapes for the eleven exact
   `/MatMul` launches.
2. Determine whether h16816 and xmma launches represent the same mathematical
   workload.
3. Resolve whether they occur in the same graph region or engine invocation.
4. Identify the observed path trigger without forcing a tactic or rebuilding an
   engine.

Hard limits: no implementation, no engine rebuild, no ONNX modification, no
precision change, and no tactic forcing. If runtime shape or workload identity
cannot be recovered, the result is `INCONCLUSIVE` and no optimization target is
opened.

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
