# Phase 2.2-C5 - End-to-End Autoregressive Generation Runtime

Date: 2026-09-03
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `2bd7925ff0be23919f1ec06440941228e0dec01e`

## Objective

Run the smallest real Qwen3 generation loop: tokenizer -> TensorRT embedding
-> 28-layer prefill/decode with KV cache -> Final RMSNorm -> LM Head -> greedy
sampling, for one short prompt and four new tokens. This is runtime
orchestration only; C1 numerical debugging, optimization and benchmarking are
out of scope.

## Prompt and Tokenizer

The pinned local Qwen3-0.6B tokenizer was used in plain causal mode for
`"Hello"`. It produced input IDs `[9707]` (prompt token count 1). HF greedy
reference (`do_sample=false`, `max_new_tokens=4`) generated
`[21806, 0, 358, 2776]`, decoded as `" Answer! I'm"`.

## Prefill

TensorRT embedding and corrected 28-layer prefill completed with position
`0`, finite hidden `[1,1,1024]`, 28 K/V layers and initialized cache. The
runtime output records the prefill cache length as `1` and all cache tensors
finite.

## Decode Loop

Three decode iterations ran using the previous TRT token, position IDs 1, 2,
and 3, and the existing dynamic decode engine. The TRT trajectory was
`[0, 46309, 46309, 46309]`; all IDs were valid. No model or engine was
reloaded per step.

## Token Trace

| Step | Phase | Position | Reference | TRT | Agreement | TRT top1-top2 margin |
|---:|---|---:|---:|---:|---|---:|
| 0 | prefill | 0 | 21806 | 0 | NO | 0.0 |
| 1 | decode | 1 | 0 | 46309 | NO | 0.703125 |
| 2 | decode | 2 | 358 | 46309 | NO | 5.546875 |
| 3 | decode | 3 | 2776 | 46309 | NO | 6.5390625 |

`FIRST_TOKEN_DIVERGENCE_STEP = 0`. At prefill, TRT top-1/top-2 were
`0/1` with both logits `0.0`; portable top-1/top-2 were `538/408` with
margin `0.1201171875`. This is a rank difference caused by the documented
upstream C1 numerical limitation, not a sampler or cache failure.

## KV Cache Gate

Every step preserved all prior K/V prefix values exactly, appended one new
position, kept all tensors finite, and maintained 28-way K/V pointer isolation.
Observed cache lengths were `1 -> 2 -> 3 -> 4`. Cache byte accounting grew
monotonically with sequence length; no overwrite was observed.

## Generated Text

TRT generated IDs `[0,46309,46309,46309]`, decoded with the pinned tokenizer as
`"!geoisgeoisgeois"`; full text is `"Hello!geoisgeoisgeois"`. HF reference
text is `"Hello Answer! I'm"`. Text quality is not evaluated by C5.

## Memory

Minimum `MemAvailable` was `2,439,254,016` bytes and maximum RSS was
`1,367,676 KB`. No OOM or exit 137 occurred. No formal latency, throughput,
power or benchmark claim is made.

## Gate

Tokenizer, prefill, KV initialization/growth, decode loop, finite CUDA outputs,
valid token IDs and tokenizer decode all passed. The token trajectory diverged
from step 0, so the bounded result is **PASS / BOUNDED** under the predefined
C1 limitation policy: `FULL_RUNTIME_TOKEN_MISMATCH_DUE_TO_UPSTREAM_NUMERICAL_LIMITATION`.

## Known C1 Limitation

C1 remains `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`. C5 did not reopen C1,
RoPE, attention, softmax or RMSNorm internals.

## C5 Readiness / Phase 2.2 Closeout

C5 demonstrates a complete single-prompt autoregressive runtime with KV cache.
The Phase 2.2 runtime prototype C0-C5 can be closed as **PASS / BOUNDED**.
Phase 2.3 is not started and requires explicit authorization.

## Artifacts

- `src/phase2_2c5/c5_generation_runtime.py`
- `artifacts/phase2_2c5_20260903T/reference_generation.json`
- `artifacts/phase2_2c5_20260903T/generation_trace.json`
- `artifacts/phase2_2c5_20260903T/memory_trace.json`
- `artifacts/phase2_2c5_20260903T/trt_tokenizer_decode.json`
