# Phase 4-A.1 Layer Operator Mapping Report

## Environment

Jetson: `NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super`, host `nvidia-desktop`

TensorRT: `10.3.0`

ONNX: `1.22.0`

Method: read-only metadata and graph mapping. No engine rebuild, ONNX export, execution, benchmark, or profiling was performed.

## Input artifacts

EngineInspector JSON: `/tmp/phase4a_operator_attribution_20260904T131524Z/mixed_decode_engine_inspector.json`

EngineInspector SHA-256: `b9305150544221c601861aa7b7a86232bbc62d851ac45619fe6166758cc5fe71`

Mixed Decode ONNX: `/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.onnx`

ONNX SHA-256: `8dccf65451c548762896280b498358069eaba100824b0acb92b7cdc9e4c51d9a`

## Methodology

The study used only the existing Phase 4-A.0 EngineInspector JSON and the existing Phase 2.3-E Mixed Decode ONNX graph.

For each TensorRT layer, the script extracted every `[ONNX Layer: ...]` reference from `Metadata`. Each referenced node name was then validated against the ONNX graph. A candidate operator was recorded only when the node name itself identified a Qwen3 projection or normalization operator.

Confidence rules:

| Confidence | Meaning |
| --- | --- |
| `HIGH` | Metadata references existing ONNX node(s), and at least one referenced node identifies a recognized Qwen3 operator. |
| `MEDIUM` | Metadata references existing ONNX node(s), but the node name does not identify a recognized Qwen3 operator. |
| `LOW` | Metadata exists, but some referenced ONNX nodes are missing. |
| `UNKNOWN` | No TensorRT metadata exists. |

No mapping was inferred from tactic names or layer type alone.

## TensorRT layer inventory

TensorRT layers: `699`

Layers with populated metadata: `498`

Layers without populated metadata: `201`

Total metadata ONNX-node references: `2529`

Unmatched metadata ONNX-node references: `0`

## ONNX mapping result

The ONNX graph contains `6545` nodes, `840` initializers, `58` graph inputs, and `84` graph outputs.

Mapping confidence counts:

| Confidence | Count |
| --- | ---: |
| `HIGH` | 386 |
| `MEDIUM` | 112 |
| `LOW` | 0 |
| `UNKNOWN` | 201 |

The `112` `MEDIUM` rows have exact ONNX graph node matches, but their node names do not identify a recognized Qwen3 operator. They are therefore not promoted to operator attribution.

The `201` `UNKNOWN` rows have no populated metadata and are left unmapped.

## GEMM candidate attribution

GEMM candidate rows: `250`

| Confidence | Count |
| --- | ---: |
| `HIGH` | 194 |
| `MEDIUM` | 56 |

The `194` `HIGH` rows cover `196` projection-operator mentions because one TensorRT layer fuses `L2:v_proj`, `L2:k_proj`, and `L2:q_proj`.

Projection attribution by operator:

| Operator | Count |
| --- | ---: |
| `q_proj` | 28 |
| `k_proj` | 28 |
| `v_proj` | 28 |
| `o_proj` | 28 |
| `gate_proj` | 28 |
| `up_proj` | 28 |
| `down_proj` | 28 |

The `56` `MEDIUM` rows correspond to the two unnamed attention-path MatMul nodes per decoder layer. Their ONNX nodes are exact graph matches, but their Transformer-operator identity remains `UNKNOWN`.

## Evidence quality

`HIGH` attribution is supported by the chain:

```text
TensorRT EngineInspector Metadata
    ↓
Exact ONNX node name
    ↓
Existing node in Mixed Decode ONNX graph
    ↓
Recognized Qwen3 operator name
```

`MEDIUM` evidence stops after the ONNX-node match. `UNKNOWN` means no evidence was available. No tactic name or TRT layer type was used to infer an operator.

## Unknown cases

- Direct runtime kernel name: `UNKNOWN`
- Kernel-to-TensorRT-layer mapping: `UNKNOWN`
- Two attention-path MatMul nodes per layer: `UNKNOWN`
- Internal `__myl_*` semantics not identified by ONNX node names: `UNKNOWN`
- `201` TensorRT layers with no populated metadata: `UNKNOWN`
- ONNX intermediate tensor shapes not present as graph outputs: `UNKNOWN`

## Gate

Phase 4-A.1: `PASS / BOUNDED`

## Recommendation

Do not proceed to `Phase 4-A.2 Kernel Correlation` from this report alone. A future kernel-correlation step would need an explicitly authorized runtime-evidence method, because this phase intentionally did not recover kernel-to-layer mapping.
