# Phase 9.3-B2 Qwen3-VL Decoder-Side Runtime Bottleneck Attribution

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized decoder-side runtime bottleneck profiling and attribution
after TensorRT Vision integration. The required observations were prefill
latency, decode latency per token, KV-cache behavior, CUDA kernel distribution,
and GEMM/attention/memory contribution.

No TensorRT-LLM migration, optimization, quantization, decoder modification,
CUDA modification, TensorRT rebuild, or persistent environment change occurred.
No new tolerance, correctness gate, optimization gate, or deployment gate was
applied.

## Gate

**PASS / BOUNDED - DECODER_BOTTLENECK_ATTRIBUTION_RECORDED**

The clean latency run completed one warmup and three measured 16-token
generations. The dedicated Nsight Systems run completed one warmup and one
profiled 16-token generation. All required metric classes were recorded.

The result is a fixed-workload attribution record. It is not an optimization
result, correctness result, or deployment recommendation.

## Protocol And Identity

The protocol was frozen before measurement as `protocol.json`. Its SHA-256 is
`69d72349a65f62c5eaa760a8e89f396b9e12dd738859fdaaf77fd27a10091c12`.

| Field | Value |
| --- | --- |
| Starting branch | `phase/09-qwen3vl-migration` |
| Starting HEAD | `e2d109c25202c58b93255b23afffaa0b3892a8b9` |
| Model identity | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| `config.json` SHA-256 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `model.safetensors` SHA-256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| TensorRT engine SHA-256 | `aa5c200eb5dcbc5abaef55bc014b5210236fb9ca34394217a024d3796e44823c` |
| Device | Jetson Orin, `cuda:0`, capability `8.7` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| CUDA | `12.6` |
| TensorRT | `10.3.0` |
| Transformers | `4.57.3` |
| Nsight Systems | `2024.5.4.34-245434855735v0` |
| Attention implementation | `eager` |
| Power mode | `25W`, unchanged |

The deterministic workload was the same 448x448 red-square image and prompt
`"Describe the image."` used by Phase 9.3-B1. Sampling was greedy for exactly
16 generated tokens. The processor produced finite FP16
`pixel_values=[784,1536]` and `image_grid_thw=[[1,28,28]]`.

Both successful runs generated the same token sequence:

```text
1986, 1986, 374, 264, 4285, 11, 52484, 2168, 16445,
264, 6437, 2518, 9334, 30188, 389, 264
```

and decoded:

```text
ThisThis is a simple, geometric image featuring a solid red square centered on a
```

## Execution Design

The clean latency run used no NVTX synchronization and no per-step cache
introspection inside the decode loop. It recorded one warmup and three measured
generations. Prefill used a CUDA event. Decode used host wall clock and also
recorded per-call CUDA-event durations without synchronizing each step.

The profile run used one warmup and one measured generation under Nsight Systems
with CUDA and NVTX tracing. Each profiled decode step synchronized before its
NVTX range closed, and the KV cache was inspected after that synchronization.
Consequently, profile-run timing is diagnostic only and is not used as the clean
latency result.

Two earlier attempts are preserved:

1. The first latency run failed before any trial because NVIDIA PyTorch did not
   expose `torch.cuda.nvtx.is_active`. The adapter gate was changed to the
   frozen run-mode flag.
2. The first successful profile run ended the TensorRT visual NVTX range before
   asynchronous engine kernels completed; it captured only 4.88 ms of kernels
   against a 136.83 ms CUDA-event visual boundary. The final profile run
   synchronized inside the profile-only visual NVTX range.

The frozen protocol was not changed after the first failure.

## Clean Latency Metrics

Means are over three measured generations. The TensorRT FP16 visual backend and
the unchanged eager PyTorch FP16 decoder were used.

| Metric | Mean | Median | Stddev |
| --- | ---: | ---: | ---: |
| Preprocess | `12.367943660744155` ms | `11.992116997134872` ms | `3.8535127530245847` ms |
| TensorRT visual adapter | `98.074462890625` ms | `97.51526641845703` ms | `2.4047404275524915` ms |
| LLM prefill | `275.9990743001302` ms | `276.380126953125` ms | `3.350261319824786` ms |
| Derived language-decoder prefill | `177.92461140950522` ms | `178.4329833984375` ms | `1.2729912860468253` ms |
| Decode, 15 calls, host | `1876.101424335502` ms | `1877.879337000195` ms | `5.685701946894933` ms |
| Decode, 15 calls, CUDA events | `1865.9511973063152` ms | `1868.1852340698242` ms | `5.348423246562001` ms |
| Decode per token, host | `125.07342828903347` ms | `125.19195580001299` ms | `0.3790467964596633` ms |
| Generation total | `2152.1004986356324` ms | `2150.353915857617` ms | `4.362663787874652` ms |
| Throughput | `7.434617388506059` tokens/s | `7.4406356470017565` tokens/s | `0.015055816223512675` tokens/s |

Decode was `0.8717536311733093` of the mean generation wall time and prefill was
`0.12824636882669066`. The derived language-decoder prefill value is calculated
as prefill minus the visual adapter boundary; it is not an independently isolated
measurement.

## Diagnostic Profile Timing

These values include per-step synchronization, NVTX tracing, cache inspection,
and profiler overhead. They are attribution evidence, not clean latency claims.

| Metric | Value |
| --- | ---: |
| Visual adapter CUDA event | `140.43431091308594` ms |
| LLM prefill CUDA event | `418.0461120605469` ms |
| Derived language-decoder prefill | `277.61180114746094` ms |
| Decode event total, 15 steps | `3263.0535888671875` ms |
| Decode event mean per step | `217.53690592447916` ms |
| Decode event median per step | `216.7950439453125` ms |
| Decode event stddev per step | `4.515857480888965` ms |
| Decode host wall, 15 steps | `3290.4495889961254` ms |

The 15 profiled decode-step NVTX ranges contained 27,825 kernels and
`1619.511904` ms total kernel duration, mean `107.96746026666666` ms per step.
The broader decode NVTX range contained 27,884 kernels and `1620.602368` ms.
The small difference is diagnostic/cache-activity overlap outside the per-step
model-call ranges.

## KV Cache Behavior

The runtime cache was `DynamicCache` with 28 layers. Every key and value was
FP16 with shape `[1,8,sequence_length,128]`.

| Boundary | Sequence length | Logical bytes | Finite ratio |
| --- | ---: | ---: | ---: |
| After prefill | `210` | `24,084,480` | `1.0` |
| After decode step 1 | `211` | `24,199,168` | not sampled per step |
| After decode step 15 | `225` | `25,804,800` | `1.0` |

Across the 15 subsequent decode calls, the logical key/value payload increased
by `1,720,320` bytes, or `114,688` bytes per token. This is a shape-and-dtype
estimate of the logical tensor payload. It is not a CUDA allocator snapshot,
TensorRT allocation measurement, or profiler DRAM attribution.

## CUDA Kernel Attribution

Kernel families are classified by kernel-name heuristics in
`analyze_nsys_decoder_profile.py`. GEMM means a kernel name matching GEMM,
matmul, CUTLASS, nvjet, or cuBLAS patterns. Memory-like means elementwise,
copy, reduction, normalization, indexing, softmax, and related names. These are
not DRAM counters, achieved-bandwidth measurements, or microarchitectural
roofline attribution.

The dedicated attention family intentionally counts only kernel names explicitly
identified by the heuristic as attention kernels. No such fused attention kernel
was observed.

| Boundary | Kernels | Kernel duration | GEMM | Memory-like | Other | Dedicated attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full profiled generation | `30,803` | `1991.283232` ms | `1443.28208` ms / `72.4800%` | `420.254496` ms / `21.1047%` | `127.746656` ms / `6.4153%` | `0` / `0%` |
| Prefill including visual | `2,109` | `345.79168` ms | `173.381312` ms / `50.1404%` | `78.015968` ms / `22.5616%` | `94.3944` ms / `27.2981%` | `0` / `0%` |
| TensorRT visual range | `206` | `138.450528` ms | `126.879808` ms / `91.6427%` | `0` / `0%` | `11.57072` ms / `8.3573%` | `0` / `0%` |
| Prefill excluding visual | `1,903` | `207.341152` ms | `46.501504` ms / `22.4275%` | `78.015968` ms / `37.6269%` | `82.82368` ms / `39.9456%` | `0` / `0%` |
| Decode range | `27,884` | `1620.602368` ms | `1269.900768` ms / `78.3598%` | `317.364928` ms / `19.5831%` | `33.336672` ms / `2.0571%` | `0` / `0%` |

The visual NVTX range captured `138.450528` ms of kernel time, consistent with
the diagnostic CUDA-event visual boundary of `140.43431091308594` ms.

The two largest decode kernel rows were:

| Kernel | Count | Duration | Share of decode kernel time |
| --- | ---: | ---: | ---: |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | `1,275` | `989.883808` ms | `61.0821%` |
| `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn` | `1,680` | `279.909472` ms | `17.2732%` |

Together, these two GEMM-class kernels were `1269.79328` ms, or `78.3511%` of
decode kernel time. They cannot be separated into attention projection, MLP, LM
head, or other GEMM operations by kernel name alone.

Attention-associated kernels that were identifiable by name were limited to:

| Boundary | Kernel | Count | Duration |
| --- | --- | ---: | ---: |
| Full generation | `_gemm_mha_v2_...` | `24` | `17.064224` ms |
| Full generation | `softmax_warp_forward` | `448` | `6.568832` ms |
| Prefill including visual | `_gemm_mha_v2_...` | `24` | `17.064224` ms |
| Prefill including visual | `softmax_warp_forward` | `28` | `2.748832` ms |
| Decode | `softmax_warp_forward` | `420` | `3.82` ms |

The absence of a dedicated attention kernel does not mean attention work was
absent. In this eager PyTorch path, attention-related computation is embedded in
generic GEMM, softmax, elementwise, and copy kernels. The remaining attention
contribution inside those generic kernels is `UNKNOWN`.

## Memory Activity

Within the full profiled generation range, Nsight recorded:

| Activity | Count | Bytes | Duration |
| --- | ---: | ---: | ---: |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | `238` | `8,899,987` | `1.291648` ms |
| `CUPTI_ACTIVITY_KIND_MEMSET` | `1,530` | `6,546,692` | `1.813984` ms |

This is API-level activity attribution, not a DRAM counter or effective-bandwidth
measurement.

## Memory Snapshots

The profile-run PyTorch allocator snapshots were:

| Snapshot | Allocated | Reserved |
| --- | ---: | ---: |
| After model load | `4,255,079,424` B | `4,324,327,424` B |
| After TensorRT injection | `3,441,164,800` B | `4,257,218,560` B |
| After trials | `3,452,896,256` B | `4,301,258,752` B |

These are allocator snapshots. They do not include TensorRT-managed allocations
or total GPU memory and are not profiler DRAM measurements.

## Attribution

For this fixed workload, decode remains the dominant end-to-end stage after
TensorRT Vision integration. In the dedicated profile, decode kernel time was
dominated by GEMM-class kernels at `78.3598%`. Memory-like kernel names
contributed `19.5831%`, while other kernel names contributed only `2.0571%`.

The eager decoder did not emit a kernel whose name identified a dedicated fused
attention implementation. Its attention-related work is therefore distributed
across generic GEMM, softmax, elementwise, and copy kernels. The exact attention
share within those kernels is `UNKNOWN`.

These observations support only the bounded conclusion that GEMM-class kernels
are the largest named-kernel component of decode kernel time. They do not prove
which optimization would help, that attention is inefficient, or that memory is
or is not the limiting microarchitectural resource.

## Raw Profile Provenance

The final Nsight report and SQLite export remain Jetson-local and are not
committed to Git.

| File | Size | SHA-256 |
| --- | ---: | --- |
| `/tmp/phase9_3b2_20260916T102808Z/profile/phase9_3B2_profile.nsys-rep` | `4,669,588` B | `624f9d48680b8a01df8e2c68ec70d7b22fe9b844f977b556adbb1539717191b3` |
| `/tmp/phase9_3b2_20260916T102808Z/profile/phase9_3B2_profile.sqlite` | `17,006,592` B | `a1a818412a313b79dbce85d845477bb28a0f846a4b3667013a2f2624c2aed4f7` |

## Limitations And Non-Claims

- Three clean measured generations and one profiled generation are bounded
  attribution evidence, not a robust benchmark sweep.
- The result applies only to the deterministic image, prompt, 16-token greedy
  workload, unchanged C1 TensorRT visual engine, and eager PyTorch FP16 decoder.
- Profile timing is not comparable to clean timing because each step
  synchronized and performed cache inspection.
- Kernel families are kernel-name heuristics, not profiler microarchitecture
  counters.
- DRAM counters, achieved occupancy, effective bandwidth, host launch-gap
  attribution, and exact attention share inside generic kernels are `UNKNOWN`.
- Power was not requested or measured and is `UNKNOWN`.
- No optimization, correctness, deployment, or TensorRT-LLM conclusion is made.

## Evidence

- Frozen protocol: `artifacts/phase9_3B2_20260916T102808Z/protocol.json`
- Clean latency result:
  `artifacts/phase9_3B2_20260916T102808Z/phase9_3B2_latency_result.json`
- Profile result:
  `artifacts/phase9_3B2_20260916T102808Z/phase9_3B2_profile_result.json`
- Nsight SQLite attribution:
  `artifacts/phase9_3B2_20260916T102808Z/phase9_3B2_profile_analysis.json`
- Harness:
  `src/phase9_3B2/run_decoder_bottleneck_profile.py`
- Parser:
  `src/phase9_3B2/analyze_nsys_decoder_profile.py`
- Manifest:
  `artifacts/phase9_3B2_20260916T102808Z/artifact_manifest.json`

The next action is to stop and await Gate review. No further rerun, optimization,
input sweep, decoder change, TensorRT-LLM migration, quantization, engine
rebuild, CUDA change, or environment change is authorized by this report.
