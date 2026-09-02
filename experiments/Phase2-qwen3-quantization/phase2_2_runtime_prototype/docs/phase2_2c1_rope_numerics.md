# Phase 2.2-C1J - Qwen3 Layer 0 RoPE Numerical Root Cause

Date: 2026-09-03
Branch: `phase/02-qwen3-quantization`
Starting HEAD: `b8a9d828257d4a8a0bfd0cb5805f089e34b71095`

## Objective

Discriminate RoPE cache generation, position indexing, layout and arithmetic
semantics using the canonical `B=1,S=8,H=1024` FP16 input, real Layer 0
weights, and fresh TensorRT micro engines. B4.2 was loaded read-only.

## Layer 0 Structure and Input Contract

The source-faithful Qwen3 block is input RMSNorm, Q/K/V projections, per-head
Q/K RMSNorm, RoPE, GQA attention, output projection/residual, post-attention
RMSNorm, and SwiGLU MLP. Dimensions are Q=`16x128`, KV=`8x128`, rotary
dimension `128`, theta `1,000,000`, GQA repeat `2`. Positions are exactly
`0..7`; all compared tensors are finite FP16 unless explicitly marked FP32.

## Cache and Position Analysis

TensorRT native RoPE cache outputs did not match the portable FP16 cache:

| Cache | Portable vs native TRT relative-L2 | Portable vs TRT cosine |
| --- | ---: | ---: |
| Q cos | `0.0978644267` | `0.995215535` |
| Q sin | `0.3146529794` | `0.960980415` |
| inv_freq | `0.1477582157` | `0.994068683` |

The same TensorRT graph with `cache_fp32` semantics matched the FP32 cache
exactly (zero recorded difference for inv_freq/cos/sin and output). This is
consistent with TensorRT retaining FP32 cache values where the portable path
casts cos/sin to FP16 before multiply/add. The repository evidence therefore
confirms a precision-semantic difference in the TensorRT path; it does not
claim a defect in TensorRT's trigonometric implementation.

Position mapping was independently checked. Input positions `0..7` were used
for both Q and K, and a shifted `1..8` run matched its shifted oracle while
producing a materially different output from the zero-based run (Q relative-L2
`0.244311124`, K `0.028620996`). Position indexing is therefore not the source
of the observed native mismatch.

## Same-Input RoPE Micro Results

| Input | Native TRT output rel-L2 | FP32 arithmetic variant | FP32-cache variant | Worst head |
| --- | ---: | ---: | ---: | ---: |
| Q `[1,16,8,128]` | `0.0927107111` | `0.0927107111` | `0` | Q head `3` (`0.119861849` native) |
| K `[1,8,8,128]` | `0.0144213885` | `0.0144213885` | `0` | K head `1` (`0.070175037` native) |

The native and FP32-arithmetic Layer 0 variants both ended at relative-L2
`0.0347152092` against the portable Layer 0 output. The cache-FP32 variant
also ended at `0.0347152092`; this diagnostic does not establish an end-to-end
improvement because the existing B4.2 engine remains unchanged and its control
is `0.0037876803` against the portable reference.

## Layout Audit

Portable Qwen3 uses half-split `rotate_half = concat(-x[...,64:], x[...,:64])`
with rotary dimension `128`. An even/odd interleave negative control changed
Layer 0 output to relative-L2 `0.349987835`, so layout/order is confirmed as a
major semantic distinction but is not the native TensorRT mismatch.

## Amplification and First Divergence

The prior C1I probe measured Q RoPE `0.092714824`, K RoPE `0.014535242`, and
QK raw `0.043750536`; C1J reproduces the same-input Q/K RoPE values. The first
RoPE component is Q RoPE, and the cache comparison narrows the mechanism to
FP32-vs-FP16 cache/arithmetic ordering. Q/K projection and normalization remain
below `0.002` relative-L2 per C1I.

## Result and Root-Cause Ranking

**C1J result: `ROPE_PRECISION_SEMANTICS_CONFIRMED`.**

1. Confirmed: TensorRT RoPE path retains FP32 cache values; portable path casts
   cos/sin to FP16 before arithmetic.
2. Confirmed: positions are `0..7`; shifted control behaves as expected.
3. Confirmed: half-split layout and GQA repeat/interleave (`8 -> 16`) are the
   source-faithful semantics.
4. Not isolated: TensorRT fusion/tactic contribution beyond the observed
   cache dtype/order remains `UNKNOWN`.

No repair, engine replacement, C1K, C2, benchmark, profiler or 28-layer rebuild
was performed. No OOM or exit 137 occurred; the run exit code was `0`.

## Artifacts and Validation

Raw evidence: `artifacts/c1j_20260902T230000Z/` (`c1j_rope_numerics_20260902T230000Z.json`, `run.log`, `exit_code.txt`).
Diagnostic source: `src/phase2_2c1/c1j_rope_numerics.py`. The script passed
Jetson tools-venv `py_compile`; `git diff --check` passed. B4.2 control remained
read-only and reproduced relative-L2 `0.0037876803`.

## Next Step

C1 remains **BLOCKED**. The exact next authorized experiment is C1K only if
explicitly approved: a minimal RoPE precision/cast implementation comparison
using a fixed exported cache, without starting C2.
