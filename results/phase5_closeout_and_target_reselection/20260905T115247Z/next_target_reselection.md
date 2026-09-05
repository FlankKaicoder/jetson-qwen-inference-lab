# Next Optimization Target Re-selection

## Conclusion

```text
NEXT_TARGET_BOUNDED
```

The next bounded candidate is `unknown_attention_matmul`, specifically the
existing `/MatMul_*` MEDIUM-confidence runtime chain. This is a candidate for
attribution and feasibility recovery, not an implementation target. The
operator identity is still `UNKNOWN`; therefore no CUDA kernel, TensorRT
Plugin, fusion, or runtime change is recommended yet.

## Inputs And Exclusions

This re-selection used only committed Phase 3-C, 3-E, 4-A, 4-B, 4-F, 4-G and
Phase 5 artifacts. No new profiling or operator benchmark was run.

It deliberately does not select RMSNorm, RoPE, or Attention merely because
they are roadmap topics. Phase 3-C identified `0 ms` under name-based
`RMSNorm` and `RoPE` categories; the smaller attention category was
`0.366560 ms` in Mixed. Their known Phase 2.2 numerical evidence cannot be
substituted for decode/prefill performance evidence. Conversely, the largest
remaining mapped runtime evidence is not ignored just because its operator is
not yet named.

## Scoring Rubric

Scores are bounded qualitative evidence scores, not predicted speedups:

| Dimension | Range |
| --- | ---: |
| Runtime contribution | `0-3` |
| Attribution confidence | `0-2` |
| Kernel / operator isolation feasibility | `0-2` |
| Optimization headroom evidence | `0-2` |
| Implementation feasibility | `0-1` |
| Integration value if faster | `0-1` |
| Evidence quality | `0-1` |
| Bounded / unresolvable evidence penalty | `0` to `-2` |

`up_proj` is excluded from the new ranking because of the Phase 5 gate. It is
shown as `CLOSED_FOR_NOW`.

## Candidate Ranking

| Rank | Candidate | Runtime evidence | Kernel / headroom evidence | Attribution confidence | Isolation feasibility | Possible mechanism | Risk | Score | Recommendation |
| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| 1 | `unknown_attention_matmul` (`/MatMul_*`) | Phase 4-A.2: 57 rows / 56 layers / `61.815776 ms` all-trace | `sm80_xmma_gemm_f16f16_f16f32...` tactic family; Phase 3-E rank-2 NCU: `76.88%` memory/L2, `17.697509%` HMMA, direct DRAM `N/A` | MEDIUM only; operator `UNKNOWN` | Bounded: dynamic shape, but exact NVTX/ONNX/Inspector chain exists | First recover operator identity and GEMM semantics; only then assess feasibility | Operator identity unknown; dynamic shape; no proven benefit | `6.5` | Attribution-only feasibility study |
| 2 | `gate_proj` (bounded, not implementation) | Phase 4-A.2: one HIGH runtime row / `0.648512 ms` all-trace | Shared Phase 3-E h16816 aggregate `39.590976 ms` in Mixed; rank-1 NCU `97.06%` memory/L2 and `39.673148%` HMMA | HIGH layer mapping, LOW runtime coverage | Low: h16816 time is shared with `up_proj` and cannot be split | Potential memory-traffic reshape only after pure attribution | Shared time; would risk reopening closed GEMM path | `6.0` | Do not rank as implementation target |
| 3 | fused `q_proj;k_proj;v_proj` | One HIGH fused layer: `4.442880 ms` all-trace; shared across q/k/v and not triple-counted | Runtime kernels recorded, but no operator-specific NCU evidence | HIGH for one layer | Low: fused range and single-layer coverage | Attribution recovery and coverage check only | Coverage limited; no headroom proof | `4.0` | Bounded follow-up only |
| 4 | TensorRT internal non-GEMM fused path (`__myl_*`) | Phase 3-C Mixed: `13.927392 ms`; FP16: `16.491392 ms` | No safe operator mapping; no targeted NCU evidence | UNKNOWN | Very low: TRT-internal and no operator chain | Fusion is a hypothesis, not an evidenced mechanism | TRT internal boundary; no isolation | `3.5` | Attribution research only |
| 5 | `down_proj` / `o_proj` | No Phase 4-A.2 mapped runtime rows | HIGH 28/28 layer mapping and static GEMM shapes only | Layer mapping HIGH, runtime coverage none | Medium for static GEMM, but no measured runtime share | No mechanism supported | No runtime/headroom evidence | `3.5` | Not selected |
| closed | `up_proj` | Phase 4-A.2: `37.554112 ms` all-trace; Phase 4-E `153-160 us` layer boundary | Complete Phase 5 feasibility chain | HIGH | High for isolated shape | No supported CUDA mechanism | Phase 5 gate closed | `CLOSED_FOR_NOW` | Not ranked unless new evidence |

All-trace mapped time is not a steady-state runtime share and must not be
interpreted as wall-clock percentage without a new controlled measurement.

## Why This Is Only Bounded

`unknown_attention_matmul` has the largest mapped time in the committed
runtime table, but three conditions prevent a strong target claim:

1. The 56-layer chain stops at `/MatMul_*` and does not identify a Transformer
   operator.
2. Its GEMM shapes are recorded as dynamic, unlike the frozen decode `up_proj`
   shape.
3. The Phase 3-E NCU metrics are associated with a tactic family, not a named
   operator, and direct DRAM remains `N/A`.

The only supported next step is therefore a narrowly scoped attribution and
feasibility study: recover `/MatMul_*` to Transformer-operator identity,
decode/prefill shape semantics, and isolation boundary before considering any
implementation. Do not implement anything in that follow-up without explicit
owner authorization.
