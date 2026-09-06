# Phase 6-C QK^T Dynamic Path Attribution Report

## Executive Summary

The historical Mixed persistent trace contains exactly eleven `/MatMul` NVTX
ranges. Seven launch h16816 and four launch xmma. Their immediate NVTX parents
differ consistently: h16816 is under
`{ForeignNode[onnx::MatMul_4 + ONNXTRT_Broadcast.../MatMul]}`, while xmma is
under `{ForeignNode[/Cast.../Add_111]}`. Therefore the old "two paths for the
same layer-0 QK^T" interpretation is not supported by the trace alone.

The final attribution-only gate is:

```text
QK_PATH_TRANSITION_UNRESOLVED
PASS / BOUNDED / GATE_D
```

Runtime tensor shapes and the h16816 workload identity remain unavailable. No
normalized h16816-versus-xmma performance is calculated, and no implementation
is authorized.

## Eleven-Launch Timeline

| Launch | Boundary | Path | Correlation | Duration ns | Immediate parent |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | WARMUP | h16816 | 9095 | 10,106,112 | onnx::MatMul_4 + Broadcast.../MatMul |
| 2 | WARMUP | h16816 | 10478 | 8,926,944 | onnx::MatMul_4 + Broadcast.../MatMul |
| 3 | PREFILL_S8 | h16816 | 11865 | 4,742,752 | onnx::MatMul_4 + Broadcast.../MatMul |
| 4 | DECODE0 | xmma | 12530 | 14,592 | /Cast.../Add_111 |
| 5 | DECODE0 | h16816 | 13709 | 10,102,688 | onnx::MatMul_4 + Broadcast.../MatMul |
| 6 | DECODE1 | xmma | 14364 | 31,008 | /Cast.../Add_111 |
| 7 | DECODE1 | h16816 | 15187 | 8,816,768 | onnx::MatMul_4 + Broadcast.../MatMul |
| 8 | DECODE2 | xmma | 15842 | 14,976 | /Cast.../Add_111 |
| 9 | DECODE2 | h16816 | 16665 | 4,690,080 | onnx::MatMul_4 + Broadcast.../MatMul |
| 10 | DECODE3 | xmma | 17320 | 15,264 | /Cast.../Add_111 |
| 11 | DECODE3 | h16816 | 18143 | 7,599,008 | onnx::MatMul_4 + Broadcast.../MatMul |

The h16816 kernel is
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1`, grid `1187x1x1`, block `64x1x1`.
The xmma kernel is
`sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_aligna2_alignc2_execute_kernel_trt`,
grid `1x1x16`, block `128x1x1`. Both have `ExecutionContext::enqueue` as an
ancestor, but this does not establish that they are the same engine invocation.

## Runtime Workload And Shapes

The profiled code executes S=8 prefill and four decode steps. In the decode loop
`old_len = 8 + step`, giving code-derived cache/key-length progression
`8, 9, 10, 11`. This is a workload-sequence hint only, not direct per-launch
runtime tensor-shape proof.

The SQLite schema has NVTX, runtime, and kernel tables but no kernel-argument or
runtime tensor-shape table. Consequently, per-launch Q, K, and output shapes are
`UNKNOWN` for all eleven launches. Shape confidence is `UNKNOWN`; the code
derived lengths must not be presented as runtime shape evidence.

## Path Transition Interpretation

For each decode step, the xmma launch precedes the h16816 launch, but parent
stacks differ. Same graph region, same engine invocation, and same mathematical
workload are each `NOT_PROVEN`. The transition mechanism is `UNKNOWN`.
Therefore h16816 and xmma timings are not directly comparable under current
evidence.

No `379x` or similar raw ratio is claimed as a tactic defect. Raw durations are
retained only as per-launch observations. The normalized performance fields are
`UNKNOWN` and marked `NOT_CALCULATED` because h16816 shape and work cannot be
established reliably.

## Cross-Layer Static Evidence

Phase 6-A static decode evidence exposes all 28 QK^T candidates as standalone
ONNX-named GEMM layers and records xmma tactics for layers 0 through 27. Static
Mixed prefill evidence represents the same QK^T nodes inside
`_gemm_mha_v2_*` fused attention layers. Separate h16816 runtime evidence exists
for gate/up projection contexts, with different launch configurations, and must
not be merged into the QK^T context.

Static layer tactics do not prove runtime shape equivalence. The
cross-layer runtime shape equivalence remains `UNKNOWN` for every layer. Only
layer 0 has this phase's direct eleven-launch runtime evidence; layers 1 through
27 were not isolated in Phase 6-C.

## Evidence Limits

The following remain unresolved:

- Actual runtime Q, K, and output tensor shapes for all eleven launches.
- Whether h16816 and xmma execute the same mathematical workload.
- Whether they belong to the same graph region or engine invocation.
- The trigger for selecting either observed kernel path.
- Whether any transition between the observed parent contexts occurs at all.
- A normalized h16816-versus-xmma performance comparison.

These limits are not gaps to be inferred away. They define Gate D.

## Optimization Surface

No clean optimization surface is proven. The xmma launches are small, but they
cannot be cleanly attributed to layer-0 QK^T over the h16816 launches. The
h16816 launches are large, but their workload and replacement surface remain
unknown. A frozen-engine audit would be required before any stronger attribution
claim, and it is not part of this phase.

## Gate

```text
Gate D: QK_PATH_TRANSITION_UNRESOLVED
Status: PASS / BOUNDED
Implementation: NOT AUTHORIZED
```

Gate D is selected because historical evidence is sufficient to correct the
earlier interpretation but insufficient to prove a same-workload dynamic kernel
path transition.

## Authorization State

```text
Custom CUDA: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Plugin: NOT AUTHORIZED
```

No engine rebuild, ONNX change, precision change, tactic forcing, or runtime
redesign is authorized.
