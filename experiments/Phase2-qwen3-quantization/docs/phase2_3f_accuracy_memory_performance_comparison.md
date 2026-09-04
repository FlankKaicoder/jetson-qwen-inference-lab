# Phase 2.3-F — Accuracy / Memory / Performance Comparison

Status: `PASS / BOUNDED` (2026-09-04)

## Objective

Compare the frozen Phase 2.3-E mixed-precision runtime against the current
TensorRT FP16 runtime on numerical/behavioral change, memory/storage change,
and inference-performance change. Primary comparison is `TRT MIXED vs TRT
FP16`; HF BF16 remains semantic-reference context only and C1 is not re-opened.

## Frozen Runtime Candidates

- Reference: current TRT FP16 runtime (B4.2 engines + C2 norm + C3 LM head).
- Candidate: Phase 2.3-E `PRIMARY_POLICY_RUNTIME` (P2_FAMILY_GUARD_REFINED:
  63 FP16 + 133 PT-W8A8).

## Evaluation Corpus

The 12 disjoint Phase 2.3-B evaluation prompts were reused (same tokenizer,
same revision). The B4.2-derived engines support a maximum context of 16
tokens, so each prompt was truncated to its first 8 tokens for prefill, forced
decode and free-run generation; 8 prefill + 8 decode = 16 stays within the
engine optimization profile.

## Prefill Same-Input Numerical Comparison

Mixed vs TRT FP16 last-token logits over 12 prompts:

| Metric | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| relative_L2 | 0.3456 | 0.3311 | 0.7313 | 1.0 |
| cosine | 0.6203 | 0.9086 | 0.9663 | 0.9679 |
| RMSE | 1.556 | 1.574 | 3.408 | 4.589 |

- Top-1 agreement: 4 / 12 (33.3%).
- Top-5 overlap: 2.92 / 5 mean.

## Same-Prefix Decode Comparison

8 forced decode steps per prompt (96 total steps):

| Metric | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| relative_L2 | Infinity* | 0.3514 | NaN* | Infinity* |
| cosine | 0.5264 | 0.8879 | 0.9703 | 0.9733 |

- Top-1 agreement: 47.9%; top-5 overlap: 2.26 / 5 mean.
- `*` Some steps produce degenerate (zero-norm) reference logits from the
  documented C1 decoder drift, making relative-L2 undefined (Infinity). The
  cosine median is the more informative summary.

## Free-Run Generation Context

Greedy 8-token generation per prompt. First divergence is step 0 for 7 of 12
prompts; two prompts never diverge over the 9-token trajectory; agreement rate
ranges from 0% to 100%. This is behavioral trajectory context, not a sole
accuracy metric, and token mismatch is not a runtime failure.

## Performance Methodology

Same benchmark session, same harness, batch 1, warmup 5, measured repeats 10,
proper CUDA synchronization. Prefill TTFT (embedding -> decoder -> norm ->
head) and single-step decode TPOT at cache length S. CV stayed below 1.3% for
every cell.

## Prefill / TTFT

| Length | FP16 mean | Mixed mean | Speedup |
| --- | ---: | ---: | ---: |
| S=8 | 1508.1 ms | 2241.9 ms | 0.673 |
| S=16 | 1510.0 ms | 2242.9 ms | 0.673 |

Mixed prefill is ~48% slower than FP16.

## Decode / TPOT / Tokens Per Second

| Length | FP16 TPOT | Mixed TPOT | Mixed tokens/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| S=8 | 1959.6 ms | 2650.1 ms | 0.377 | 0.739 |
| S=16 | 1958.4 ms | 2649.3 ms | 0.377 | 0.739 |

Mixed decode is ~35% slower; throughput is 0.51 -> 0.38 tokens/s (26% lower).

S=32 and S=128 exceed the B4.2-derived engine optimization-profile maximum
sequence length 16 and are recorded as a bounded engine-profile limitation,
not a memory failure.

## Memory

- No OOM, no exit 137 in any required run.
- Peak process RSS was ~1.74 GB; minimum observed MemAvailable stayed above
  3.86 GB; CUDA memory remained bounded.

## Engine / Storage Footprint

| Engine | FP16 bytes | Mixed bytes | Reduction |
| --- | ---: | ---: | ---: |
| Prefill | 892,483,860 | 651,671,676 | 27.0% |
| Decode | 890,363,012 | 650,285,868 | 27.0% |

## Accuracy–Performance Tradeoff

The mixed runtime reduces serialized engine weight storage by 27% and proves
133 INT8 GEMM targets, but it is 26-49% slower end-to-end and introduces
substantial mixed-vs-FP16 numerical and token divergence. The quantization
policy therefore trades accuracy and latency for storage without an end-to-end
speedup under the current unoptimized runtime.

## Engineering Conclusion

Measured labels:

- Numerical: substantial mixed-vs-FP16 drift (prefill logits median relative-L2
  0.331; forced-decode cosine median 0.888; top-1 agreement 33-48%).
- Behavioral: free-run trajectories diverge at step 0 for a majority of prompts.
- Memory: no OOM/exit137; peak RSS ~1.74 GB.
- Storage: 27% smaller engines.
- Performance: `MIXED_RUNTIME_SLOWER` — no end-to-end speedup; kernel-level
  INT8 does not translate to runtime speedup because only 56.9% of parameters
  are quantized and Q/DQ plus remaining FP16 layers add overhead.

## Gate

`Phase 2.3-F = PASS / BOUNDED`. All comparison methodology completed with
finite outputs and no OOM/exit137; bounded by substantial numerical/token
drift, the 16-token engine context limit for S=32/128, and a slower mixed
runtime.

## Phase 2.3 Closeout

Phase 2.3-A/B/C/D/E/F are complete. Aggregate Phase 2.3 status is
`CLOSED / PASS / BOUNDED`; it is not numerically equivalent to FP16/HF, and C1
remains `CLOSED / NUMERICAL_LIMITATION_UNRESOLVED`.

## Evidence

- Script: `src/phase2_3f/phase2_3f_compare.py`
- Artifacts: `artifacts/phase2_3f_20260904T050000Z/`
- Mixed/FP16 engines remain Jetson-local under `/tmp`.
