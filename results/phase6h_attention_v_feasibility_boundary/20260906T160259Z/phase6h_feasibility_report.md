# Phase 6-H Attention x V Feasibility Boundary Report

## Executive Summary

Phase 6-H is an offline repository-only feasibility-boundary synthesis. The
final gate is:

```text
NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY
```

The Phase 6-G surface is real and directly attributable, but current evidence
does not prove that it is inefficient enough. Therefore it also does not prove
that a future optimization feasibility experiment is justified. This is not a
claim that the AV xmma kernel is optimal and not a claim of a TensorRT tactic
defect.

The user-supplied Phase 6-H prompt was truncated after `Final Gate:`. The exact
owner gate specification is `UNAVAILABLE_TRUNCATED_PROMPT`; the conservative
gates frozen in `phase6h_plan.md` were used.

## Source Boundary

No Jetson execution, benchmark, inference, Nsight Systems run, Nsight Compute
run, engine build, implementation, ONNX change, precision change, or runtime
change occurred. Phase 6-H only reconciled existing committed evidence.

The authoritative runtime chain is Phase 6-G:

```text
Phase 6-A Attention x V semantic identity
-> Phase 3-C AV NVTX range
-> contained cuLaunchKernelEx
-> Phase 6-G correlationId
-> xmma CUDA kernel
```

This chain recovered 112/112 decode AV instances across 28/28 layers. All 112
instances map to the exact kernel
`sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize64x128x32_stage5_warpsize2x2x1_tensor16x8x16_aligna2_alignc2_execute_kernel_trt`.

## Contribution And Denominator Discipline

| Boundary | AV duration | AV share | Denominator |
| --- | ---: | ---: | ---: |
| Representative steady decode, steps 0-3 | `4,237,568 ns` | `2.866503%` | `147,830,560 ns` |
| Historical all trace | `4,237,568 ns` | `1.834842%` | `230,950,048 ns` |

The raw duration is identical because all observed AV instances already fall in
decode steps 0-3. The percentages use different denominators and are not
interchangeable. Zero AV instances occur in warmup or steady prefill.

This contribution establishes importance for attribution, but by itself it does
not establish inefficiency, headroom, or replacement benefit.

## Feasibility Answer

| Question | Answer | Basis |
| --- | --- | --- |
| Is AV directly attributable? | `YES`, attribution-only, HIGH confidence | 112/112 direct correlation chains and 28/28 TensorRT layers |
| Is the isolated decode surface clean? | `PARTIALLY SUPPORTED` | Direct isolation is strong; replacement shape and backend identity remain `UNKNOWN` |
| Is AV inefficient enough? | `NOT PROVEN` | No directly owned AV NCU efficiency sample or headroom evidence |
| Is existing NCU evidence directly AV? | `NO` | No committed NCU artifact carries the exact AV kernel or an AV correlation ID |
| Is a future optimization feasibility experiment justified by current evidence? | `NO_PROVEN_JUSTIFICATION` | The evidence proves importance, not inefficiency |

## NCU Ownership Reconciliation

The exact AV kernel name does not occur in the committed Phase 3-E or Phase 5-B
NCU artifacts. The repository search over both `ncu` artifact roots returned no
match.

The nearest existing NCU evidence is not AV:

1. Phase 3-E rank 2 is a `tn_n` f16f16/f16f32 TensorRT GEMM and its operator
   mapping was `UNKNOWN`.
2. Phase 3-E rank 3 is a `f16f16_f16_tn_n` fused variant with different output
   precision and kernel name.
3. Phase 5-B explicitly profiles the frozen `up_proj` workload and captures a
   different `tn_n` TensorRT xmma variant plus a direct cuBLASLt kernel.

Phase 3-E rank-2/rank-3 values such as `71.92-76.88%` memory/L2,
`17.167166-17.697509%` HMMA pipe active, and `24.25-24.68%` achieved occupancy
therefore remain context for similar small decode GEMMs. They cannot be
transferred to the AV instances by kernel-name similarity. Direct DRAM was
reported as `N/A` in that evidence and must not be estimated.

## Clean-Surface Limits

The following fields remain `UNKNOWN` for the 112 AV instances:

- exact kernel arguments;
- exact runtime GEMM workload shapes;
- numeric TensorRT tactic identity;
- CUDA backend identity;
- directly measured achieved occupancy, HMMA activity, memory/L2 utilization,
  and DRAM throughput;
- proven speedup, benefit, or accuracy-impact headroom.

Because of these limits, the surface is clean enough to preserve as a bounded
attribution target, but it is not proven clean enough for replacement
feasibility. The Phase 6-G report's `NOT PROVEN` optimization-surface status
therefore remains correct.

## Gate Decision

Gate A is not selected because no directly owned AV efficiency or headroom
measurement exists. Gate B is not selected because there is no bounded positive
efficiency signal to carry forward. Gate C is selected:

```text
PASS / BOUNDED
NO_PROVEN_AV_OPTIMIZATION_OPPORTUNITY
```

This result is a valid negative feasibility-boundary outcome. It does not
authorize kernel replacement, CUDA/FlashAttention/plugin implementation,
engine rebuild, precision change, tactic forcing, or another phase.

## Exact Owner Decision

The owner may choose one of the following without automatically starting it:

1. close or pause the Attention branch;
2. perform another narrowly bounded attribution study that first proves direct
   AV NCU instance ownership before any optimization claim;
3. redirect to a different active target.

If direct AV NCU ownership is later authorized, it must use the same engine
invocation and workload boundary as the Phase 6-G surface or establish direct
correlation/instance ownership. Any optimization experiment remains a separate
decision.
