# Phase 2.2-C4 - Greedy Sampling Integration

Date: 2026-09-03
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `3a1bf6cf59be27dd0e07e415b2fe5eaeb009852c`

## Objective

Validate deterministic greedy sampling only:
`next_token_id = argmax(logits, dim=-1)`. No top-k/top-p, temperature,
penalties, beam search, tokenizer loop, multi-token generation, benchmark,
profiling or quantization was run.

## Starting State

C1 remains `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`; C2 and C3 are
`PASS / BOUNDED`. Existing embedding, corrected 28-layer decoder, Final
RMSNorm and LM Head engines were consumed read-only.

## Greedy Semantics

Input logits are `[B,1,151936]`; output token IDs are `[B,1]`. The reference
uses PyTorch CPU `argmax`; the C4 backend uses CPU/NumPy `argmax` after a host
copy. Both use first-index-wins for exact ties. This is a functional sampler,
not an optimized CUDA implementation.

## Sampler Backend

`SAMPLER_BACKEND = CPU_NUMPY_ARGMAX`. Host transfer is recorded as the actual
FP32 NumPy buffer size (`607,744` bytes for one `[1,1,151936]` logits tensor).

## Synthetic Validation

Clear-winner margin `3.0`, near-tie margin `0.0001000166`, and exact-tie
margin `0.0` all produced exact integer agreement and valid IDs. The exact-tie
top-k display can order equal values differently, but both argmax implementations
returned the first tied index (`34567`).

## Same-Logits Validation

C3 synthetic decode logits produced token `2629` on both reference and C4
sampler, with top1-top2 margin `0.0625` on both sides. C3 last-token logits
produced token `57133` on both sides; portable margin was `0.03515625` and TRT
margin `0.041015625`. All IDs were in `[0,151936)`.

## LM Head -> Sampler Validation

For both C3 same-input decode and last-token hidden inputs, portable LM Head
plus reference sampler and TensorRT LM Head plus C4 CPU sampler agreed exactly.
The corresponding C3 LM Head relative-L2 values remain `0.00164510` and
`0.00168146`.

## Full Runtime Single-Step

The read-only chain `input_ids [0..7] -> embedding -> 28-layer decoder ->
Final RMSNorm -> last-token LM Head -> CPU greedy sampler` produced finite
logits `[1,1,151936]` and a valid token. Last input token was `7`; portable
next token was `42`, TRT next token was `42`, so agreement is `YES`.

## Token Margin Analysis

Portable and TRT top-1 were both `42`, top-2 were both `320`, and both margins
were `0.421875`. The same-hidden logits relative-L2 was `0.00204306`; this is
an upstream numerical diagnostic and not a sampler failure.

## Optional Token Decode

Not run. Tokenizer decoding is not a C4 Gate.

## Memory

Minimum `MemAvailable` was `3,139,858,432` bytes and maximum RSS was
`2,000,580 KB`. No OOM or exit 137 occurred. No latency or performance claim
is made.

## Gate

Greedy semantics, synthetic edge cases, C3 same-logits agreement, LM
Head-to-sampler agreement, valid single-step token range and runtime execution
all passed. C4 result: **PASS / BOUNDED**.

## Known C1 Limitation

The full path remains `END_TO_END_DIAGNOSTIC_ONLY` because C1 decoder numerical
drift is unresolved. C4 did not reopen C1, RoPE or attention debugging.

## C5 Readiness

C4 establishes a deterministic single-step token boundary. C5 autoregressive
generation loop is not started and requires explicit authorization.

## Artifacts

- `src/phase2_2c4/c4_greedy_sampling.py`
- `artifacts/phase2_2c4_20260903T/c4_greedy_sampling_validation.json`
