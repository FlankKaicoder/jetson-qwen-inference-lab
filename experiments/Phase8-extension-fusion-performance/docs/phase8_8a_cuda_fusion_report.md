# Phase 8.8-A CUDA Fusion Microbenchmark Report

Date: 2026-09-20

Branch: `phase/08-extension-fusion-analysis`

Starting HEAD: `f83f3e8e6b46a839a24ba40e3a202eaea5be0177`

Status: `PASS / BOUNDED`

Gate: `NO_PROVEN_FUSION_BENEFIT`

## Result

The FP16 prototype reduced the profiled launch count from two kernels to one.
Its ten-trial CUDA Event mean was `6.3146%` lower and the single-launch NSYS
kernel-time sum was `5.5611%` lower. These observations do not prove an
operator-level fusion benefit: the CUDA Event median improved by only
`1.6886%`, NCU L1 global-load sectors increased by `1.4409%`, and global-store
sectors increased by `12x`. The predeclared all-condition gate therefore
failed. Phase 8.8-B was not started.

## Scope and evidence boundary

The owner authorized Phase 8.8-A only. The experiment added a standalone CUDA
microbenchmark under the new Phase 8.8 directory. It did not modify the Qwen3
model, existing Phase 8 code or artifacts, TensorRT graphs, or plugins. It did
not build a TensorRT engine or run Phase 8.8-B/C.

Phase 8.7 established that the repository does not prove a missing FP16
TensorRT RMSNorm fusion target. Detailed Mixed/INT8 evidence shows that
TensorRT can fuse normalization with downstream quantization, but that is not
FP16 proof. Phase 8.8-A therefore tests a standalone CUDA hypothesis and does
not assume that native TensorRT is unfused.

## Environment and protocol

| Item | Value |
| --- | --- |
| Device | Jetson Orin Nano Super, `Orin`, SM87, 8 SMs |
| CUDA | 12.6.68; driver/runtime API `12060` |
| Nsight Systems | 2024.5.4.34 |
| Nsight Compute | 2024.3.1.0 |
| Shape | FP16 `M=1`, `K=1024`, `N=3072` |
| RMSNorm epsilon | `1e-6` |
| Seed | `20260920` |
| Timing | 50 warmup pairs; 10 trials; 500 operations per CUDA Event interval |
| Trial order | Alternating baseline-fusion / fusion-baseline |
| Clock control | Unmodified; NCU used `--clock-control none` |

The shape matches the Qwen3-0.6B decode-side MLP projection dimensions, but
inputs, RMS weights, and linear weights are deterministic synthetic values.
No checkpoint weights were loaded.

## Implementations

The baseline launches one 256-thread RMSNorm kernel, writes a 1024-element FP16
intermediate, and calls `cublasGemmEx` for the `1024 -> 3072` projection.

The fusion path launches one simple CUDA kernel with 32 blocks and 256 threads
per block. Each block computes RMSNorm into a block-local shared-memory vector;
its eight warps then process strided output rows. This removes the global
normalized intermediate and one launch, but repeats normalization across 32
blocks and uses one-lane-per-output scalar stores. It is intentionally a
prototype, not a production GEMM implementation.

## Correctness

The gate required finite output and relative-L2 `<= 0.01` for the baseline and
fusion paths against an FP32-accumulation reference, and for fusion versus
baseline. All checks passed.

| Comparison | Max abs | Relative-L2 | Cosine | Result |
| --- | ---: | ---: | ---: | --- |
| Baseline vs reference | `0.0004888773` | `0.0001988578` | `0.9999999802` | PASS |
| Fusion vs reference | `0.0004888773` | `0.0001988549` | `0.9999999802` | PASS |
| Fusion vs baseline | `0.0002441406` | `0.0000110587` | `0.9999999999` | PASS |

## Performance

| Metric | Baseline | Fusion | Reduction | Evidence |
| --- | ---: | ---: | ---: | --- |
| Kernel count | `2` | `1` | `50.0000%` | NSYS one captured operation |
| CUDA Event mean | `0.1058619 ms` | `0.0991772 ms` | `6.3146%` | 10 x 500 calls |
| CUDA Event median | `0.1092214 ms` | `0.1073771 ms` | `1.6886%` | 10 trial means |
| L1 global-load sectors | `199872` | `202752` | `-1.4409%` | NCU |
| L1 global-store sectors | `256` | `3072` | `-1100.0000%` | NCU |
| NSYS GPU kernel time | `0.256064 ms` | `0.241824 ms` | `5.5611%` | Separate single-operation captures |
| NCU kernel duration sum | `0.174688 ms` | `0.253504 ms` | `-45.1182%` | Diagnostic profiled launches |

Negative reduction means the fusion result increased. The L1 sector counts
correspond to `6,395,904` versus `6,488,064` load bytes and `8,192` versus
`98,304` store bytes when multiplied by the 32-byte sector size. These are L1
global access-sector values, not DRAM bytes, achieved bandwidth, or effective
bandwidth. DRAM read/write counters remain `UNKNOWN`.

NCU achieved active-warps values were `16.55%` for RMSNorm, `16.48%` for the
cuBLAS kernel, and `65.59%` for fusion. These are achieved active-warp metrics,
not theoretical occupancy. Higher fusion occupancy did not establish lower
operator latency. The NCU duration ordering conflicts with the NSYS and
batched CUDA Event ordering; separate profiler runs, replay/instrumentation,
and uncontrolled clocks prevent treating either profiler duration as a stable
latency replacement. GPU-time attribution is therefore bounded, while the
formal latency gate is based on the repeated CUDA Event protocol.

## Gate

`FUSION_BENEFIT_PROVEN` required every check below to pass.

| Check | Result |
| --- | --- |
| Correctness | PASS |
| Kernel count reduced | PASS |
| CUDA Event mean at least 5% lower | PASS |
| CUDA Event median at least 5% lower | FAIL |
| NCU global-load sectors reduced | FAIL |
| NCU global-store sectors reduced | FAIL |
| NSYS GPU kernel time at least 5% lower | PASS |

Final classification:

```text
PASS / BOUNDED / NO_PROVEN_FUSION_BENEFIT
```

The result does not prove that RMSNorm + Linear fusion cannot be beneficial.
It shows that this simple implementation reduced launch count but did not
reduce measured global traffic or satisfy the stability requirement. A more
complex implementation could change the result, but Phase 8.8-B is explicitly
conditional on Phase 8.8-A proving benefit and is therefore not authorized by
this evidence.

## Failed attempts and evidence

Two NSYS orchestration failures are preserved. Attempt 1 used unsupported
`--stop-on-range-end`; attempt 2 used `stop-shutdown`, which produced a report
but returned before the script could export stats. A first NCU derived summary
also selected the `avg` rollup instead of `.sum`; the raw NCU CSV was unchanged,
the incorrect derived files were preserved, and the corrected analyzer rebuilt
the summaries from raw evidence.

- Formal compact evidence:
  `artifacts/phase8_8a_20260920T123247Z/`
- Source and runner:
  `phase8_8_a_cuda_fusion_microbenchmark/` and `scripts/`
- Jetson formal workspace:
  `/tmp/phase8_8a_20260920_7ylLfy/`
- Jetson failed workspaces:
  `/tmp/phase8_8a_20260920_oHlj70/` and
  `/tmp/phase8_8a_20260920_xmPtTn/`
- Raw `.nsys-rep` files remain in the Jetson formal workspace and are not
  committed.

## Exact next action

Stop. Do not implement `RMSNormLinearPlugin`, modify TensorRT, or run Qwen3
single-layer validation. Phase 8.8-B requires a fresh owner decision that
explicitly changes or overrides the Phase 8.8-A dependency gate.
