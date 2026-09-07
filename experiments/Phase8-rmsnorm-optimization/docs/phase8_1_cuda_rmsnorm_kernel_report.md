# Phase 8.1 CUDA RMSNorm Kernel Report

## Experiment Goal

Phase 8.1 implemented and measured three standalone CUDA RMSNorm kernels for
hidden size `1024`, token counts `8` and `1`, FP16, and optional BF16. The
goal was correctness plus an independent C++/CUDA kernel benchmark, not a
full-model integration or a claim about Qwen3 runtime benefit.

The authorized boundary was:

```text
CUDA RMSNorm implementation
-> correctness against FP32 reduction
-> CUDA Event benchmark
-> analysis and report
```

No NCU, NSYS, TensorRT Plugin, ONNX change, engine rebuild, full-model run, or
Phase 8.2 work was performed.

## Design

All versions use one CUDA block per token and the formula

```text
y = x / sqrt(mean(x^2) + eps) * weight
```

with `eps = 1e-6`. Inputs and weights are deterministic `normal(0, 1)`
values generated with seed `20260907`; they are not Qwen3 checkpoint tensors.
The reference uses FP32 reduction on the host.

| Version | Threads per token block | Reduction | Memory | Notes |
| --- | ---: | --- | --- | --- |
| V0 | 1024 | Shared-memory tree reduction | Scalar loads/stores | Tree reduction has multiple block-wide synchronization barriers. |
| V1 | 1024 | `__shfl_down_sync` warp reduction plus one final warp reduction | Scalar loads/stores | Fewer block-wide synchronization points and much less shared-memory traffic than V0. |
| V2 | 512 | `__shfl_down_sync` warp reduction | `half2` / `bfloat162` loads and stores | Each thread handles two contiguous elements. |

The correctness oracle computes the FP32 mean square, inverse RMS, and
weighted output on the host. A candidate value is compared after conversion
to the selected storage dtype. The predefined gate was finite output and
relative-L2 `<= 0.005`; this threshold was not adjusted after the run.

## Build And Execution Environment

| Field | Value |
| --- | --- |
| Device | Jetson Orin Nano Super |
| Compute capability | `8.7` |
| CUDA compiler | `/usr/local/cuda-12.6/bin/nvcc`, CUDA `12.6.68` |
| CMake | `3.22.1` |
| Host compiler | GNU C++ `11.4.0` |
| Build type | `Release` |
| CUDA architecture | `87` |
| OS | Ubuntu 22.04, aarch64 |

The Jetson repository checkout remained unchanged at
`phase/03e-tensorrt-kernel-attribution@bf7abc67eb58662a68316045e166aa9f611330d7`.
Phase 8.1 source was copied to a fresh
`/tmp/phase8_1_rmsnorm_20260907T093447Z/` directory. No Jetson checkout was
switched, reset, cleaned, or synchronized.

Build and artifact hashes are recorded in
`../artifacts/phase8_1_20260907T093447Z/run_manifest.txt`.

## Correctness Results

All twelve cases were finite and passed the predefined gate.

| Dtype | Shape | V0 relative-L2 | V1 relative-L2 | V2 relative-L2 | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| FP16 | `[1,8,1024]` | `0.000208692` | `0.000208692` | `0.000208692` | PASS |
| FP16 | `[1,1,1024]` | `0.000206635` | `0.000206635` | `0.000206635` | PASS |
| BF16 | `[1,8,1024]` | `0.0016336` | `0.0016336` | `0.0016336` | PASS |
| BF16 | `[1,1,1024]` | `0.00165305` | `0.00165305` | `0.00165305` | PASS |

Corresponding max absolute errors were:

| Dtype | Shape | Max abs error |
| --- | --- | ---: |
| FP16 | `[1,8,1024]` | `0.00194883` |
| FP16 | `[1,1,1024]` | `0.00179863` |
| BF16 | `[1,8,1024]` | `0.0305176` |
| BF16 | `[1,1,1024]` | `0.012115` |

Raw evidence:

```text
experiments/Phase8-rmsnorm-optimization/artifacts/phase8_1_20260907T093447Z/correctness_summary.csv
experiments/Phase8-rmsnorm-optimization/artifacts/phase8_1_20260907T093447Z/correctness_raw.csv
```

A pilot run exposed a BF16 dispatch bug that invoked the FP16 launcher for the
BF16 case. Correctness failed as expected. This was fixed before the formal
run. The pilot evidence is retained as
`pilot_correctness_summary.csv` and `pilot_benchmark_trials.csv`; the formal
tables above are the reported results.

## Benchmark Protocol

| Field | Value |
| --- | --- |
| Warmup | 50 kernel launches |
| Repetitions per trial | 200 kernel launches |
| Trials | 5 |
| Timing | CUDA Events |
| Primary `mean_ms` | Mean of 1000 per-call event-pair values across 5 trials |
| Secondary `amortized_mean_ms` | One event pair around each 200-launch trial, averaged over 5 trials |
| Streams | One dedicated stream per case |
| Precision | FP16 and BF16 |
| Shapes | Prefill `[1,8,1024]`; decode `[1,1,1024]` |

The two timing views are deliberately separated. The per-call CUDA Event view
includes CUDA Event overhead and one host submission per launch. The amortized
view includes queued back-to-back launch behavior. Neither number is labeled as
an NCU kernel duration.

## Latency Results

### FP16

| Shape | Version | Per-call event mean (ms) | Amortized event mean (ms) | CV |
| --- | --- | ---: | ---: | ---: |
| Prefill `[1,8,1024]` | V0 | `0.02369286405` | `0.01890995193` | `0.06339257602` |
| Prefill `[1,8,1024]` | V1 | `0.02021680002` | `0.01597116804` | `0.02931778117` |
| Prefill `[1,8,1024]` | V2 | `0.01891695993` | `0.0145360961` | `0.04623218089` |
| Decode `[1,1,1024]` | V0 | `0.02214268798` | `0.01773638368` | `0.05617119577` |
| Decode `[1,1,1024]` | V1 | `0.01916262398` | `0.01460956788` | `0.03976700474` |
| Decode `[1,1,1024]` | V2 | `0.01759222411` | `0.01308825612` | `0.1054116501` |

### BF16

| Shape | Version | Per-call event mean (ms) | Amortized event mean (ms) | CV |
| --- | --- | ---: | ---: | ---: |
| Prefill `[1,8,1024]` | V0 | `0.0237172161` | `0.01890556812` | `0.0495397898` |
| Prefill `[1,8,1024]` | V1 | `0.02027360004` | `0.01604211187` | `0.02899302392` |
| Prefill `[1,8,1024]` | V2 | `0.01899411203` | `0.01450598383` | `0.03729358974` |
| Decode `[1,1,1024]` | V0 | `0.02227996794` | `0.01782016015` | `0.03281546623` |
| Decode `[1,1,1024]` | V1 | `0.01909088006` | `0.014579808` | `0.03419244923` |
| Decode `[1,1,1024]` | V2 | `0.01738649605` | `0.01303142428` | `0.03419334168` |

Raw evidence:

```text
experiments/Phase8-rmsnorm-optimization/artifacts/phase8_1_20260907T093447Z/benchmark_summary.csv
experiments/Phase8-rmsnorm-optimization/artifacts/phase8_1_20260907T093447Z/benchmark_trials.csv
```

## Performance Analysis

On FP16 prefill, V1 improved the per-call event mean by about `14.7%` versus
V0, and V2 improved by about `20.2%`. On FP16 decode, the corresponding V0
improvements were about `13.5%` for V1 and `20.6%` for V2. The BF16 ordering
was the same.

The V0-to-V1 change is better explained by reduction and synchronization than
by memory access. V1 keeps the same 1024 threads and same scalar memory access
pattern, but replaces most of V0's shared-memory tree barriers with warp
shuffle reduction. The measured V1 improvement is therefore consistent with
reduced block-wide synchronization and shared-memory pressure.

The V1-to-V2 change is consistent with vectorized memory access because V2
uses contiguous `half2` / `bfloat162` loads and stores. However, V2 also halves
the thread count from 1024 to 512 and changes per-thread work. Therefore the
V2-versus-V1 benefit cannot be attributed to memory access alone from this
benchmark. NCU would be required to separate memory transactions, achieved
occupancy, issue behavior, and stall causes.

Phase 8.0's stable PyTorch call baseline was host-launch/operator-overhead
dominated, and `PYTORCH_RMSNORM_KERNEL_ONLY_LATENCY` remained `UNKNOWN`. It is
not compared numerically to these CUDA kernels here and is not evidence that a
full Qwen3 model will speed up.

## Limitations

- This is a synthetic hidden-size-1024 RMSNorm benchmark, not a Qwen3
  checkpoint, TensorRT engine, or runtime integration.
- The correctness weight is deterministic random data, not the actual Qwen3
  `model.norm.weight`.
- No NCU, NSYS, achieved occupancy, theoretical occupancy, DRAM counter, or
  effective-bandwidth evidence was collected in Phase 8.1.
- CUDA Event timing is not identical to a hardware profiler kernel duration.
- Clock and power state were not changed or controlled.
- V2's block-shape and memory-layout changes are coupled, so the exact causal
  decomposition of its improvement is `INCONCLUSIVE` without NCU.
- The BF16 path passed the preset gate, but its observed error is larger than
  FP16, as expected for lower mantissa precision.
- This result does not prove that replacing Qwen3 RMSNorm will improve the full
  model.

## Gate

```text
PASS / BOUNDED
```

This means Phase 8.1 met the authorized kernel implementation, correctness,
and CUDA Event benchmark scope. It does not establish a full-model optimization
benefit or authorize Phase 8.2. Phase 8.2, including any NCU profiling, remains
a separate owner decision.
