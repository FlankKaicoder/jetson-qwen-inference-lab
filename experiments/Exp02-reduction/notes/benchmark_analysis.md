# Exp02.2 Benchmark / Stability Analysis

## Method

B1 used `N=16,777,229`, blocks 32/64/128/256/512/1024, V1-V7, three independent rounds, warmup 20 and 50 measured repetitions. B2 used `N=2^20, 2^22, 2^24, 2^24+13`, the B1 primary block for each version, three rounds, warmup 20 and 100 repetitions. B3 used `N=16,777,229`, the frozen primary block, five independent rounds, warmup 20 and 200 repetitions. CUDA Events covered every reduction pass and kernel launch; allocation, input generation, H2D, D2H, reference and CSV output were outside the timed region.

Every measured configuration copied the final device scalar back and checked it against the frozen double-reference tolerance. All measured rows passed correctness.

## Block survey and frozen candidates

B1 selected V1/B512 and V2-V7/B128 by lowest three-round mean. Secondary B256 was retained for V2-V7. V1 had higher B1 variance on some blocks, so the final B3 result is the stability authority; no round was deleted.

## Final stability

| Version | Block | Mean ms | Median ms | Sample std ms | CV % | Min ms | Max ms | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 512 | 2.725409 | 2.682311 | 0.126395 | 4.638 | 2.570060 | 2.857719 | 5/5 |
| V2 | 128 | 5.460660 | 5.460999 | 0.036368 | 0.666 | 5.419487 | 5.502149 | 5/5 |
| V3 | 128 | 3.704415 | 3.706161 | 0.003684 | 0.099 | 3.697853 | 3.706321 | 5/5 |
| V4 | 128 | 2.926582 | 2.930621 | 0.013624 | 0.466 | 2.903485 | 2.939676 | 5/5 |
| V5 | 128 | 1.626439 | 1.628676 | 0.012997 | 0.799 | 1.611595 | 1.641815 | 5/5 |
| V6 | 128 | 2.655211 | 2.657987 | 0.015669 | 0.590 | 2.631083 | 2.674031 | 5/5 |
| V7 | 128 | 2.321449 | 2.338882 | 0.026009 | 1.120 | 2.281556 | 2.339485 | 5/5 |

All final stability rows had correctness `PASS`. The fastest observed version under this benchmark is V5/B128. This is a benchmark result, not a microarchitectural conclusion.

## Paired comparisons

The paired unit is the independent round (n=5), not individual kernel repetitions. Delta is `latency_A - latency_B`; positive means B is faster. V6 vs V7: mean delta `0.333762 ms`, 95% CI `[0.316896, 0.350628] ms`, significant in favor of V7. V1 vs V5: `1.098970 ms`, CI `[0.938616, 1.259323]`, significant in favor of V5. V2 vs V3: `1.756246 ms`, CI `[1.713943, 1.798549]`; V3 vs V4: `0.777833 ms`, CI `[0.757972, 0.797693]`; V4 vs V5: `1.300143 ms`, CI `[1.272817, 1.327469]`; V5 vs V6: `-1.028772 ms`, CI `[-1.054724, -1.002820]`, significant in favor of V5.

## Hypotheses covered by Gate B

- H1: `SUPPORTED` for performance. V1 is slower than V5 and has more passes/launches; direct traffic mechanism is deferred to Gate C.
- H4: `SUPPORTED` for performance/pipeline metadata. V5 has fewer first-stage blocks and fewer passes than V1 and is fastest in B3.
- H7: `SUPPORTED`. V7 is faster than V6 with a paired 95% CI excluding zero.
- H8: `SUPPORTED`. B1 block performance is non-monotonic; larger blocks are not uniformly faster.
- H9: `INCONCLUSIVE`. B2 shows scaling differences, but launch and synchronization effects cannot be isolated from DVFS and version-specific work using benchmark timing alone.

Gate B is `PASS`: B1, B2 and B3 complete; all timed configurations passed correctness; raw and summary artifacts are retained. Gate C remains `NOT_STARTED` until its checkpoint commit is pushed.

Artifacts:

- `benchmark/raw/block_survey_20260830T172929Z.csv`
- `benchmark/raw/block_survey_summary_20260830T172929Z.csv`
- `benchmark/raw/block_candidates_20260830T172929Z.csv`
- `benchmark/raw/scaling_20260830T172929Z.csv`
- `benchmark/raw/scaling_summary_20260830T172929Z.csv`
- `benchmark/raw/stability_20260830T172929Z.csv`
- `benchmark/raw/stability_summary_20260830T172929Z.csv`
- `benchmark/raw/paired_comparisons_20260830T172929Z.csv`