# Phase 7 Global Optimization Target Reassessment Report

## Executive Summary

Phase 7 is an offline repository-only reassessment of every remaining
optimization candidate after Phase 3 through Phase 6. It uses the frozen
`R + A + S + B + F - U` scoring model without changing weights or dimensions.

Final gate:

```text
PASS / BOUNDED
NO_PROVEN_CUSTOM_KERNEL_OPTIMIZATION_TARGET
```

The strongest active evidence surface is **Attention x V across 28 layers**.
Phase 6-G proves a complete decode runtime/kernel attribution chain, and
Phase 6-H correctly bounds what that proof does not establish. It does not
prove inefficiency, a tactic defect, headroom, replacement benefit, or an
accuracy-risk trade-off. Therefore it remains
`NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY`, not an authorized implementation
target.

## Scope And Method

No Jetson execution, benchmark, NSYS, NCU, engine build, ONNX change,
precision change, tactic forcing, kernel replacement, plugin, or
FlashAttention work occurred. Phase 7 only synthesized committed artifacts.

The source hierarchy is Git and raw artifacts first, then frozen experiment
reports, then `docs/PROJECT_STATE.md`. The preserved untracked directories in
the working tree were left unchanged and were not used to supersede committed
Phase 6 evidence.

## Candidate Inventory Result

The master inventory has eleven candidates:

| Candidate | Phase 7 status |
| --- | --- |
| Persistent ExecutionContext | `PROVEN_OPTIMIZATION` |
| Attention x V | `NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY` |
| Layer-0 QK h16816 | `NO_CURRENT_ACTION` |
| QK layers 1-27 | `ATTRIBUTION_ONLY` |
| Fused QKV | `ATTRIBUTION_ONLY` |
| `gate_proj` shared h16816 | `ATTRIBUTION_ONLY` |
| `down_proj` / `o_proj` | `NO_CURRENT_ACTION` |
| TensorRT `__myl_*` | `ATTRIBUTION_ONLY` |
| RMSNorm / RoPE | `NO_CURRENT_ACTION` |
| CUDA Graph current prototype | `NO_CURRENT_ACTION` |
| `up_proj` | `CLOSED_FOR_NOW` |

Only Attention x V receives a scoring update relative to Phase 6-F. No closed
candidate is reopened.

## Attention x V Reassessment

Phase 6-G recovered 112/112 AV instances across 28/28 decoder layers. There
are 112 unique correlation IDs, 112 `cuLaunchKernelEx` rows, 28 TensorRT
runtime layers, and one exact kernel:

```text
sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize64x128x32_stage5_warpsize2x2x1_tensor16x8x16_aligna2_alignc2_execute_kernel_trt
```

The representative steady decode boundary is:

| Boundary | AV duration | Share |
| --- | ---: | ---: |
| `PHASE3B_STEADY_DECODE_STEP_0..3` | `4,237,568 ns` | `2.866503%` of `147,830,560 ns` |
| Historical all trace | `4,237,568 ns` | `1.834842%` of `230,950,048 ns` |

The same raw duration appears in both rows only because every observed AV
instance already falls in decode steps 0-3. The denominators are separate and
are not interchangeable.

Phase 7 scores:

| Dimension | Phase 6-F | Phase 7 | Basis |
| --- | ---: | ---: | --- |
| R | 1 | 2 | Direct steady-boundary contribution and no warmup contamination. |
| A | 3 | 3 | Attention x V semantic identity remains HIGH. |
| S | 1 | 2 | Complete 112/112 kernel attribution, but exact workload and replacement shape remain UNKNOWN. |
| B | 1 | 1 | No inefficiency, defect, headroom, or benefit evidence. |
| F | 2 | 1 | The existing-SQLite attribution path is consumed; no directly owned AV NCU evidence exists. |
| U | 2 | 3 | Kernel arguments, runtime shape, tactic identity, backend identity, and headroom remain UNKNOWN. |

The total remains `6`, but the evidence is now stronger as attribution and
weaker as implementation feasibility. This is the correct bounded
interpretation of Phase 6-G/H.

## Ranking Before And After

Phase 6-F active ordering:

1. Layer-0 QK h16816, score 6, `NO_CURRENT_ACTION`.
2. Attention x V, score 6, `ATTRIBUTION_ONLY`.
3. QK layers 1-27, score 5, `ATTRIBUTION_ONLY`.
4. Fused QKV, score 4, `ATTRIBUTION_ONLY`.
5. `gate_proj`, score 4, `ATTRIBUTION_ONLY`.
6. `down_proj` / `o_proj`, score 4, `NO_CURRENT_ACTION`.
7. TensorRT `__myl_*`, score 1, `ATTRIBUTION_ONLY`.
8. RMSNorm / RoPE, score -1, `NO_CURRENT_ACTION`.
9. CUDA Graph current prototype, score -2, `NO_CURRENT_ACTION`.
Excluded: `up_proj`, score 8, `CLOSED_FOR_NOW`.

Phase 7 active ordering:

1. Attention x V, score 6, `NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY`.
2. Layer-0 QK h16816, score 6, `NO_CURRENT_ACTION`.
3. QK layers 1-27, score 5, `ATTRIBUTION_ONLY`.
4. Fused QKV, score 4, `ATTRIBUTION_ONLY`.
5. `gate_proj`, score 4, `ATTRIBUTION_ONLY`.
6. `down_proj` / `o_proj`, score 4, `NO_CURRENT_ACTION`.
7. TensorRT `__myl_*`, score 1, `ATTRIBUTION_ONLY`.
8. RMSNorm / RoPE, score -1, `NO_CURRENT_ACTION`.
9. CUDA Graph current prototype, score -2, `NO_CURRENT_ACTION`.
Excluded: Persistent ExecutionContext, score 15, `PROVEN_OPTIMIZATION`.
Excluded: `up_proj`, score 8, `CLOSED_FOR_NOW`.

The AV-to-Rank-1 change is not a claim that AV became an optimization target.
It reflects that AV now has the strongest directly attributable evidence
surface among active candidates, while Layer-0 QK remains blocked by failed
controlled reproducibility.

## Evidence Reasoning For No Target

The Phase 7 target test requires all five conditions:

1. Runtime significance: only AV has a clean active steady-boundary surface;
   its share is `2.866503%`, but contribution alone is not inefficiency.
2. Semantic identity: AV has HIGH identity from Phase 6-A through the runtime
   chain in Phase 6-G.
3. Kernel identity: AV has one exact xmma kernel and 112/112 correlation.
4. Clean optimization surface: only `PARTIALLY_SUPPORTED`; exact workload,
   tactic, backend, and replacement shape remain `UNKNOWN`.
5. Current implementation may be insufficient: `NOT PROVEN`. Zero committed
   NCU artifacts carry the exact AV kernel or an AV correlation ID.

No active candidate satisfies condition 5. Candidates with stronger
significance are closed or blocked:

- `up_proj` is closed because Phase 5 found no tactic defect and no CUDA
  replacement benefit.
- Layer-0 QK h16816 is real but was not reproduced under the controlled
  boundary; its workload identity and trigger remain `UNKNOWN`.
- Persistent ExecutionContext is already implemented and proven.

## Closed Candidates And Frozen Status

The user-supplied freeze remains exact. In particular:

```text
Persistent ExecutionContext: PROVEN_OPTIMIZATION
up_proj: CLOSED_FOR_NOW
Layer-0 QK h16816: NO_CURRENT_ACTION
Attention x V: NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY
QK layers 1-27: ATTRIBUTION_ONLY
Fused QKV: ATTRIBUTION_ONLY
gate_proj shared h16816: ATTRIBUTION_ONLY
down_proj / o_proj: NO_CURRENT_ACTION
TensorRT __myl_*: ATTRIBUTION_ONLY
RMSNorm / RoPE: NO_CURRENT_ACTION
```

The Phase 6 Attention branch remains closed for now. `NO_PROVEN` is a valid
negative reassessment and does not claim the xmma kernel is optimal.

## Validation

Validation was repository-only:

1. Starting branch, HEAD, recent commits, and remote were checked.
2. `README.md`, `ROADMAP.md`, `AGENTS.md`, `docs/project_management.md`,
   `docs/PROJECT_STATE.md`, and `docs/experiment_index.md` were read.
3. The Phase 6-D scoring model, Phase 6-F ranking, Phase 6-G attribution
   artifacts, and Phase 6-H feasibility boundary were reconciled.
4. Every Phase 7 performance number was traced to the evidence manifest.
5. Missing AV fields remain `UNKNOWN`; no similar-kernel NCU metric was
   transferred to AV.
6. No historical result was overwritten or deleted.

## Authorization State

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
CUTLASS tuning: NOT AUTHORIZED
Kernel replacement: NOT AUTHORIZED
Engine rebuild: NOT AUTHORIZED
Precision change: NOT AUTHORIZED
Tactic forcing: NOT AUTHORIZED
New profiling campaign: NOT AUTHORIZED
Next implementation phase: NOT STARTED
```
