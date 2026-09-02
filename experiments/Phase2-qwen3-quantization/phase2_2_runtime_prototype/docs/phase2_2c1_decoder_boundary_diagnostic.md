# Phase 2.2-C1D — Decoder Input Boundary Diagnostic

Date: 2026-09-02
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `f2249dd353b85d849292b59ca08c2a418888b36a`
Diagnostic artifact: `artifacts/phase2_2c1d_20260902T180000Z_c1d_diagnostic.json`

## Scope and Safety

This diagnostic reused the existing B4.2 prefill engine and C1 embedding engine from Jetson `/tmp`. No engine, model, B4.2 runtime, or historical artifact was overwritten. The only code addition is `src/phase2_2c1/c1d_decoder_boundary_diagnostic.py`.

## D0 — B4.2 Current Control

The existing 28-layer prefill engine was run with a deterministic contiguous FP16 random hidden tensor, `B=1,S=8,H=1024`, and `position_ids=[0..7]`. Portable FP16 versus TensorRT Layer 27 produced:

| max abs | RMSE | relative-L2 | cosine | result |
| ---: | ---: | ---: | ---: | --- |
| 5.0 | 0.50999 | 0.0201395 | 0.9997966 | PASS |

All outputs were finite, shape-correct and CUDA-resident. Therefore `B4.2_CURRENT_CONTROL = PASS`.

## D1 — Embedding Boundary Fingerprint

The canonical input was the serialized C1 portable FP16 embedding for IDs `[0,1,2,3,4,5,6,7]`. Both sides were `[1,8,1024]`, FP16, contiguous, strides `[8192,1024,1]`, 16,384 bytes.

Portable and TensorRT embedding tensors were byte-identical:

`sha256 = da04f5331a23473a72c65952945c87ba03e10ec84e57de2ae456e62e0bff2e2f`

Metrics were max abs `0`, RMSE `0`, relative-L2 `0`, cosine `1.00000048`. `DECODER_INPUT_A_BYTE_IDENTICAL_TO_B = YES`.

## D2 — Canonical Hidden Test

The exact same canonical FP16 tensor was supplied to the portable 28-layer stack and the existing TensorRT decoder. Layer 27 failed the B4.2 bounded criterion:

| max abs | RMSE | relative-L2 | cosine |
| ---: | ---: | ---: | ---: |
| 4268.0 | 56.9263 | 2.004247 | 0.5342684 |

The first-layer difference is small but nonzero (L0 relative-L2 `0.00378768`, cosine `0.9999921`) and grows progressively through the stack: L15 `0.0322621`, L20 `0.0913260`, L23 `0.185188`, L26 `0.342038`, L27 `2.004247`.

## D3 — Host-Staged vs Direct Device

Both paths used the same IDs, position IDs, engine and profile. Host-staged means explicit device-to-host copy followed by a contiguous FP16 device upload before decoder execution.

| path | max abs | RMSE | relative-L2 | cosine | result |
| --- | ---: | ---: | ---: | ---: | --- |
| host-staged | 4268.0 | 56.9263 | 2.004247 | 0.5342684 | FAIL |
| direct device | 4268.0 | 56.9263 | 2.004247 | 0.5342684 | FAIL |

The two outputs have the same canonical input hash and the same numerical result. Handoff transport is therefore not causal.

## D4 — Stream / Pointer Audit

Embedding and decoder both used CUDA stream pointer `0` (the current/default stream). Each `execute_async_v3` returned successfully and was followed by explicit stream synchronization. The embedding output pointer was retained until the decoder call; the staged path used a separate contiguous allocation. No missing producer-consumer ordering was observed.

## D5 — Decoder Input Contract

The prefill engine has 86 I/O tensors: exactly two inputs and 84 outputs.

| input | dtype | runtime shape | source |
| --- | --- | --- | --- |
| `hidden_states` | FP16 | `[1,8,1024]` | canonical embedding / control tensor |
| `position_ids` | INT64 | `[1,8]` | `arange(8)` |

There are no prefill KV, attention-mask, cache-length, or other auxiliary input bindings. Binding names, dtypes and shapes matched the engine contract.

## Root Cause

`CONFIRMED` for the following narrower conclusion: the C1 failure is not caused by embedding mathematics, input pointer lifetime, host/device staging, CUDA stream ordering, wrong binding selection, or missing prefill auxiliary inputs. The exact canonical embedding reaches both oracles identically.

`NOT CONFIRMED` for a single implementation defect. The evidence instead shows progressive numerical divergence inside the unchanged FP16 28-layer TensorRT decoder for this real embedding distribution, while the historical random-hidden B4.2 control remains bounded. Existing B4.2 acceptance evidence is therefore insufficient to claim C1 real-embedding agreement.

## Minimal Fix

No fix is applied. Rebuilding or rewriting the decoder is outside this diagnostic scope and is not justified by the current evidence. A future authorized investigation should compare per-layer TensorRT versus portable intermediates under identical canonical input and assess FP32 accumulation/normalization or engine graph precision, but C2 must not start.

## Gate

| Gate | Result |
| --- | --- |
| D0 B4.2 current control | PASS |
| D1 byte-identical decoder input | PASS / YES |
| D2 canonical hidden agreement | FAIL |
| D3 host-staged vs direct | BOTH FAIL, identical |
| D4 stream/pointer audit | PASS / no ordering defect found |
| D5 binding contract | PASS / only hidden + position inputs |
| C1 final gate | **BLOCKED** |

No OOM, exit 137, or CUDA execution failure occurred in this diagnostic run. B4.2 engine/code and historical artifacts were not modified.
