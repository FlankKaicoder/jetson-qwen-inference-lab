# Phase 4-A.0 EngineInspector Probe Report

## Environment

TensorRT: `10.3.0`

CUDA: `12.6`, `V12.6.68`

Jetson: `NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super`, host `nvidia-desktop`

## Engine

Engine path: `/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine`

Engine size: `650,285,868` bytes

Engine SHA-256: `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c`

Engine rebuild: `false`

## Inspector capability

Layer information: `PASS`

JSON export: `PASS`

Shape recovery: `PASS / BOUNDED`

Precision recovery: `PASS / BOUNDED`

Tactic recovery: `PASS / BOUNDED`

Kernel metadata: `UNKNOWN`

## Findings

The existing Mixed Decode engine exposed `699` layers through
`TensorRT 10.3` EngineInspector JSON. Every layer record contained the observed
keys `Name`, `LayerType`, `Metadata`, `Inputs`, `Outputs`, `StreamId`, and
`TacticName`.

The observed layer-type distribution is:

| Layer type | Count |
| --- | ---: |
| `kgen` | 248 |
| `fusion` | 165 |
| `wait` | 114 |
| `signal` | 86 |
| `gemm` | 85 |
| `shape_call` | 1 |

These are TensorRT internal Inspector layer types, not ONNX operator names.

Input tensor precision counts are:

| Input precision | Count |
| --- | ---: |
| `Half` | 1065 |
| `Int64` | 953 |
| `Int8` | 266 |
| `Float` | 170 |

Output tensor precision counts are:

| Output precision | Count |
| --- | ---: |
| `Half` | 561 |
| `Int8` | 133 |

`498` layers had non-empty input/output tensor data and non-empty
`Metadata`; the remaining `201` layers did not. Among the populated layers,
`498` had non-empty tactic values spanning `68` unique tactic strings. The
tactic family counts were `248` `__myl_*` values and `250`
`sm80_xmma_gemm_*` values. No other tactic family was observed.

Shape information was present for `3148` input/output tensors. Of these,
`336` tensors contained dynamic `-1` dimensions and were preserved as dynamic
rather than inferred to a concrete size.

EngineInspector can currently recover:

- TensorRT layer identity and order.
- TensorRT internal layer type.
- Tensor metadata where the field is populated.
- Input/output shapes where the field is populated, including dynamic dimensions.
- Input/output precision where the field is populated.
- `TacticName` strings where populated.
- ONNX-origin metadata strings where populated.

EngineInspector cannot currently recover:

- A dedicated runtime kernel-name field.
- A direct kernel-to-layer mapping.
- A direct TensorRT-layer-to-Transformer-operator mapping.
- The semantic identity of internal `__myl_*` kernels.

The absence of a dedicated kernel-name field means direct kernel attribution
remains `UNKNOWN`. Tactic strings are useful evidence but are not treated as a
kernel-to-operator proof in this report.

## Gate

Phase 4-A.0: `PASS / BOUNDED`

## Recommendation

Proceed to `Phase 4-A.1 Layer -> Transformer Operator Mapping` as a read-only
metadata mapping study. Do not infer Transformer operator identity from tactic
names alone, and do not infer concrete dynamic shapes without an explicitly
recorded execution shape.
