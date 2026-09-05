# Phase 5-B Step 1: Engine Inspector / Tactic Attribution Recovery

## Gate

**Phase 5-B Step 1: `CASE_B_SUPPORTED_BOUNDED`.**

TensorRT `TacticName` strings were recovered for all 28 `up_proj` layers.
Numeric tactic IDs, runtime workspace sizes, the TensorRT backend identity, and
a complete tactic-to-kernel mapping were not recovered. The bounded result does
not prove a tactic defect and does not authorize CUDA kernel work.

## Question

For the decode-only, FP16, `M=1, K=1024, N=3072` `up_proj` operation:

1. What TensorRT implementation can be recovered?
2. Can tactic information be recovered?
3. What relationship exists between Inspector tactic labels and observed
   runtime kernel families?
4. Is there direct evidence of a tactic difference from direct cuBLASLt?

## Method

This was read-only attribution analysis over frozen evidence. No engine was
rebuilt, deserialized, or executed; no ONNX, builder, precision policy, tactic
selection, runtime lifecycle, CUDA kernel, or TensorRT Plugin was changed.

The repository evidence chain was:

1. Frozen TensorRT 10.3 EngineInspector JSON from Phase 4-A.0;
2. Frozen Phase 4-A.1 Inspector-Metadata-to-ONNX mapping;
3. Frozen Phase 5-A Step 3 CUPTI/NSYS kernel breakdown.

The reproducible script is
[phase5b_tactic_attribution.py](../../../experiments/Phase5-cuda-feasibility/scripts/phase5b_tactic_attribution.py).
It writes this directory's `tactic_attribution.csv` and
`tactic_attribution_analysis.json` only. Source hashes are recorded in the JSON
analysis.

## Engine And Coverage

The Mixed Decode engine remains identified by SHA-256
`445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c`.

The frozen Inspector JSON contains `699` layers. The Phase 4-A.1 mapping gives
`28/28` HIGH-confidence `up_proj` rows. Each row was joined to the corresponding
Inspector layer and to `7` observed Phase 5-A runtime instances, giving `196`
runtime observations.

## Recovered Fields

For all 28 logical layers, the following were recovered:

| Field | Result |
| --- | --- |
| TensorRT layer ID | `22, 51, 74, ..., 696` |
| TensorRT layer name | `up_proj` GEMM layer name |
| TensorRT Inspector layer type | `gemm` |
| ONNX node | exact named `/up_proj.../MatMul` node |
| Input activation | `[1,1,1024]`, `Half` |
| Expanded weight | `[1,1024,3072]`, `Half` |
| Output | `[1,1,3072]`, `Half` |
| Alpha/beta tensor precision | `Float` |
| Operator attribution confidence | `HIGH` |
| Inspector `TacticName` | recovered |
| Runtime kernel family counts | `h16816=3; sm80_xmma_gemm=4` per layer |

Twenty-five layers use:

```text
sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16
```

Decoder layers `8`, `14`, and `27` use:

```text
sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16
```

These are Inspector tactic strings. The presence of `f16f32_f32` in three
labels is not treated by itself as proof of effective FP32 accumulation
semantics. Likewise, `tn_n` is a label only, not a proven physical-layout
contract.

## NOT_AVAILABLE And UNKNOWN Fields

| Field | Status |
| --- | --- |
| Numeric tactic ID | `NOT_AVAILABLE` |
| Runtime workspace | `NOT_AVAILABLE` |
| Dedicated Inspector kernel-name field | `NOT_AVAILABLE` |
| Alpha/beta values or semantics | `UNKNOWN` |
| Accumulator dtype | `UNKNOWN` |
| Physical layout semantics | `UNKNOWN` |
| TensorRT backend identity | `UNKNOWN` |
| One-to-one tactic-to-runtime-kernel mapping | `UNKNOWN` |
| Direct Tensor Core measurement for all 28 rows | `NOT_AVAILABLE` |

The Inspector data model exposes `TacticName`, not a public numeric tactic ID
or backend identity. Absence of these fields is therefore recorded as
`NOT_AVAILABLE` at the Inspector boundary. Conclusions that require them remain
`UNKNOWN`.

## Runtime Kernel Attribution

The Phase 5-A Step 3 evidence has `196` `up_proj` NVTX range instances and
`196` correlated CUDA kernel events. There is one correlated kernel per
observed instance; this is not evidence of seven kernels per logical layer
invocation.

Across the seven historical trace invocations, each logical `up_proj` layer
observed:

| Kernel family | Count | Phase 5-A median duration |
| --- | ---: | ---: |
| `h16816` | `3` | `253.680 us` |
| `sm80_xmma_gemm` | `4` | `118.800 us` |

Family assignment changes by trace invocation rather than by decoder layer. The
exact runtime names include the xmma variant corresponding to the Inspector
tactic for layer family `8/14/27` and the non-fused xmma variant for the other
25 layers, but the switching mechanism remains `UNKNOWN`. Therefore
tactic-to-kernel-family mapping is bounded, not one-to-one identity evidence.

Tensor Core usage is bounded as follows: all 28 tactic labels contain
`tensor16x8x16`. Phase 3-E measured HMMA activity only for selected kernels and
does not provide a direct measurement for all 28 `up_proj` rows in this phase.

## TensorRT Versus cuBLASLt Comparison

The direct cuBLASLt frozen baseline used `cublasLtMatmul`, heuristic index `4`,
algorithm ID `21`, and workspace `0` bytes. Its FP16-input/FP32-accumulate
median was `80.077961 us`.

That identity is recorded only as comparison context. It does not establish
that TensorRT uses cuBLASLt, algorithm 21, or an equivalent path. The
comparison remains non-paired: the historical NSYS run has different timing
boundaries, invocation history, and uncontrolled clocks. No new benchmark was
run in this phase.

The bounded analytical difference remains:

| Path | Median / steady-state | Comparison |
| --- | ---: | ---: |
| Direct cuBLASLt | `80.078 us` | reference |
| TensorRT correlated xmma family | `118.800 us` | informational |
| TensorRT correlated h16816 family | `253.680 us` | informational |
| TensorRT steady-state correlated kernel | `147.424 us` | informational |

This supports a path difference at the observation level but does not prove
that the difference is a tactic-selection defect.

## Updated Gate Judgement

```text
CASE_B_SUPPORTED_BOUNDED
TACTIC_STRINGS_RECOVERED
NUMERIC_TACTIC_ID_NOT_AVAILABLE
RUNTIME_KERNEL_FAMILY_RECOVERED
TACTIC_TO_KERNEL_FAMILY_MAPPING_NOT_ONE_TO_ONE
TENSORRT_BACKEND_IDENTITY_UNKNOWN
NO_PROVEN_TACTIC_DEFECT
NO_IMPLEMENTATION_AUTHORIZED
```

## Exact Next Action

Stop after Phase 5-B Step 1. Await owner review before any further
investigation or implementation.
