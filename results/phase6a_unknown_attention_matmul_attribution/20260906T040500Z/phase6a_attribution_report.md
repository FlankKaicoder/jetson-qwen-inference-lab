# Phase 6-A Unknown Attention MatMul Attribution Report

## 1. Executive Summary

Phase 6-A recovers the historical `unknown_attention_matmul / /MatMul_*`
candidate as 56 decode EngineInspector layers: 28 QK^T MatMuls and 28
Attention x V MatMuls, one pair per Qwen3 decoder layer 0 through 27. The ONNX
semantic identities are HIGH confidence.

The same ONNX nodes are shared semantically but have different runtime
representations: prefill EngineInspector exposes them inside the 28 fused
`_gemm_mha_v2_*` layers, while decode EngineInspector exposes standalone
ONNX-named GEMM layers. The exact CUDA kernel implementation semantics remain
UNKNOWN.

The final gate is `NO_PROVEN_ATTENTION_OPTIMIZATION_TARGET`, `PASS / BOUNDED`.
The raw `61.815776 ms` number is an all-trace aggregate, not a single MatMul
latency. It contains a tactic-inconsistent `54,984,352 ns`
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1` contribution whose attribution to this
semantic MatMul remains INCONCLUSIVE. The tactic-consistent xmma subset is only
`6.831424 ms`, or `2.957966%` of the `230.950048 ms` measured GPU kernel total.
No implementation is authorized.

## 2. Starting State

| Field | Value |
| --- | --- |
| Starting branch | `phase/05a-cuda-feasibility-baseline-study` |
| Starting HEAD | `271ff821c399891ac59ccdb5c18b5d6381008dcf` |
| Working branch | `phase/06a-attention-matmul-attribution` |
| Working tree at start | Protected Phase 2 artifact plus new Phase 6 experiment directory only |
| New Jetson execution | None |
| New profiling or benchmark | None |
| Engine/ONNX modification | None |

The protected directory
`experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T203103Z/`
remained untracked, untouched, and unstaged.

## 3. Evidence Sources

The recovery used five frozen artifacts, hashed in `analysis_summary.json`:

1. `experiments/Phase2-qwen3-quantization/artifacts/phase2_3e_20260904T034300Z/engine_precision_summary.json`
2. `results/phase4a_operator_attribution/20260904T131524Z/mixed_decode_engine_inspector.json`
3. `results/phase4a_operator_attribution/20260904T132820Z/onnx_node_inventory.csv`
4. `results/phase4a_operator_attribution/20260904T134556Z/runtime_nvtx_kernel_mapping.csv`
5. `results/phase3c_residual_runtime/20260904T093500Z_nsys/stats/mixed_persistent_cuda_gpu_kern_sum.csv`

The standalone decode EngineInspector copy was byte-hash matched to the copy
embedded in the Phase 2.3-E summary. Boundary rules are listed in
`evidence_manifest.csv`.

## 4. Candidate Inventory

The full inventory is in `candidate_inventory.csv`. It has exactly 56 rows.

| Candidate family | ONNX node | Decoder layer | TRT decode layer pattern | Input 0 | Input 1 | Output |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| QK^T | `/MatMul`, `/MatMul_2` through `/MatMul_54`, even numbers | `n/2` for `n = 0,2,4,...,54` | `/MatMul_n_myl0_*` | `[16,1,128]` | `[16,128,-1]` | `[16,1,-1]` |
| Attention x V | `/MatMul_1`, `/MatMul_3` through `/MatMul_55`, odd numbers | `(n-1)/2` for `n = 1,3,5,...,55` | `/MatMul_n_myl0_*` | `[16,1,-1]` | `[16,-1,128]` | `[16,1,128]` |

All operands and outputs are recorded as Half. Dynamic sequence/cache extents
are represented as `-1` in the EngineInspector dimensions and remain UNKNOWN in
the historical trace boundary. Historical row counts and times are aggregate
values, not per-call claims.

## 5. Graph Attribution

For every even QK candidate `n`, the first operand producer exposes the Q
branch anchors `/q_norm_l/Div` and `/q_norm_l/Mul`; the second exposes the K
branch anchors `/k_norm_l/Div`, `/k_norm_l/Mul_1`, `/Reshape_*`, and
`/Transpose_*`, where `l` is decoder layer `n/2`. Shapes are compatible with
`[H,S_q,D] x [H,D,S_k] -> [H,S_q,S_k]`. The immediate decode TRT consumer
contains the matching `/Softmax_l` anchor. The full three-hop edge rows are in
`graph_neighborhood.csv`.

For every odd Attention x V candidate `n`, the first operand producer is the
matching `/Softmax_l` output; the second producer exposes the matching
`/v_proj_l` and GQA Expand anchors. Shapes are compatible with
`[H,S_q,S_k] x [H,S_k,D] -> [H,S_q,D]`. The immediate decode TRT consumer
contains the matching `/o_proj_l` anchor.

Shape compatibility is treated only as corroborating evidence. Identity is not
based on the node name.

## 6. Semantic Attribution

| Semantic identity | Count | Confidence | Basis |
| --- | ---: | --- | --- |
| QK^T | 28/28 | HIGH | Q and K provenance, K reshape/transpose, shape compatibility, downstream Softmax, and decoder-layer anchors agree |
| Attention x V | 28/28 | HIGH | Softmax provenance, V/GQA Expand provenance, shape compatibility, downstream o_proj, and decoder-layer anchors agree |
| Other MatMul | 0 | N/A | No additional `/MatMul_*` candidate was recovered |
| UNKNOWN | 0 | N/A | No candidate failed the applicable anchor checks |

This is ONNX semantic attribution only. It does not prove that TensorRT executes
two standalone attention MatMul kernels.

## 7. Prefill / Decode Attribution

| Runtime context | Result | Evidence |
| --- | --- | --- |
| Prefill | Present as 28 fused `_gemm_mha_v2_*` layers; both QK and AV metadata for each layer are present, giving 56 semantic matches | Prefill EngineInspector metadata |
| Decode | Present as 56 standalone ONNX-named TRT GEMM layers | Decode EngineInspector |
| Context class | `SHARED_SEMANTIC_PREFILL_FUSED_DECODE_STANDALONE` | Joined static engine evidence |
| Historical trace prefill/decode latency split | UNKNOWN | The timing artifact aggregates the full captured window and does not provide a clean split for these rows |

`prefill_fused_matches=56` is expected because two semantic MatMuls match each
of 28 fused prefill layers.

## 8. TensorRT Fusion / Rewrite Analysis

The observed representation is:

```text
ONNX QK / AV
    |
    +--> Prefill TRT: fused into _gemm_mha_v2_*
    +--> Decode TRT: standalone ONNX-named GEMM layers
```

For decode, the full EngineInspector layer/tactic string is recovered. The
QK/AV GEMM tactic strings generally include
`sm80_xmma_gemm_f16f16_f16f32_f32_*`, but the exact backend, accumulator
semantics, physical layout, reformat topology, and one-to-one kernel mapping
remain UNKNOWN. Prefill attention work is already fused at the observed TRT
layer boundary. Whether there is an additional internal rewrite beyond that
boundary cannot be proved from the committed artifacts.

## 9. Runtime Contribution

The historical `61.815776 ms` is the sum of 57 NVTX-to-kernel mapping rows and
231 kernel instances across the 56 unique nodes and all 28 decoder layers. It is
an all-trace Phase 3-C Mixed persistent NSYS aggregate over representative
prefill plus four decode steps. It is not a single-node latency, not an
IProfiler layer time, not a CUDA event duration, and not an NCU duration.

| Quantity | Value | Boundary |
| --- | ---: | --- |
| Candidate family all-row total | `61.815776 ms` | 57 mapping rows / 231 instances / all trace |
| Tactic-consistent xmma subset | `6.831424 ms` | 224 instances; kernel name matches the layer tactic |
| Tactic-inconsistent h16816 contribution | `54,984,352 ns` | `/MatMul`, decode layer 0, 7 instances |
| Mixed persistent GPU kernel denominator | `230.950048 ms` | Phase 3-C captured workload aggregate |
| All-row family share | `26.765864%` | Same NSYS aggregate denominator |
| Tactic-consistent family share | `2.957966%` | Same NSYS aggregate denominator |

The h16816 row is `23.807898%` of the same denominator, but it is
`INCONCLUSIVE`: the observed kernel is not the expected tactic-consistent xmma
kernel for that layer. It cannot be claimed as proven QK/AV attention time. No
clean current attention contribution is established.

## 10. Evidence Limits

The following remain UNKNOWN or INCONCLUSIVE:

- Why `/MatMul` decode layer 0 maps to both h16816 and tactic-consistent xmma rows.
- Whether the suspicious h16816 time is a true execution of that node, a misattributed NVTX range, a tactic change, or another trace artifact.
- A clean prefill versus decode time split for the aggregate.
- Exact CUDA kernel implementation semantics for either semantic MatMul.
- Exact TensorRT backend identity, accumulator semantics, physical layout semantics, and one-to-one tactic-to-kernel mapping.
- Whether replacing a standalone decode MatMul would be a valid optimization surface.
- Whether the tactic-consistent `2.957966%` share is actionable after pipeline, memory, launch, and fusion constraints are considered.

No new profiling was performed because identity recovery was possible offline.
Contribution remains bounded rather than interpreted through the suspicious raw
number.

## 11. Phase 6-A Gate

```text
NO_PROVEN_ATTENTION_OPTIMIZATION_TARGET
PASS / BOUNDED
```

Reasons:

1. Semantic identity is recovered, but that alone does not authorize a target.
2. The raw family total is an aggregation artifact and is not a clean contribution.
3. The tactic-consistent subset is not a proven significant optimization target.
4. The suspicious h16816 contribution remains INCONCLUSIVE.
5. Prefill is fused and decode CUDA implementation semantics are UNKNOWN, so no clean optimization surface is proven.

## 12. Authorization State

```text
Custom CUDA Attention: NOT AUTHORIZED
FlashAttention: NOT AUTHORIZED
TensorRT Attention Plugin: NOT AUTHORIZED
```

No engine rebuild, ONNX modification, precision change, tactic forcing, or
implementation phase is authorized by Phase 6-A.
