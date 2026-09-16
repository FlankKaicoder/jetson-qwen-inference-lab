# Phase 9.3-B1 Qwen3-VL End-to-End Stage Latency Breakdown

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized an end-to-end stage latency breakdown only. The frozen
protocol measured preprocess, visual encoder, projector, LLM prefill, and
decode latency per token for both the PyTorch FP16 visual path and the
unchanged Phase 9.2-C1 TensorRT FP16 visual adapter.

No optimization, quantization, decoder modification, TensorRT rebuild, CUDA
modification, persistent environment modification, or benchmark sweep occurred.
No correctness gate, optimization gate, or deployment gate was applied.

## Gate

**PASS / BOUNDED - STAGE_LATENCY_ATTRIBUTION_RECORDED**

All six measured generations completed successfully. Each backend produced the
same 16-token sequence across its warmup and three measured trials. The required
stage metrics were recorded.

The result is bounded attribution evidence for one deterministic workload. It
is not a correctness or deployment result. The two backends diverged at token
index 8, so their later token trajectories differed. The cause was not
diagnosed because that was outside this authorization.

## Protocol And Identity

The protocol was frozen before measurement as `protocol.json`. Its SHA-256 is
`b85ee2e8fa29c25c424c573ddaefd85fe97eae0dbf0ffa145612463a4e2821db`.

| Field | Value |
| --- | --- |
| Model identity | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| `config.json` SHA-256 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `model.safetensors` SHA-256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| TensorRT engine SHA-256 before/after | `aa5c200eb5dcbc5abaef55bc014b5210236fb9ca34394217a024d3796e44823c` |
| Device | Jetson Orin, `cuda:0`, capability `8.7` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| CUDA | `12.6` |
| TensorRT | `10.3.0` |
| Transformers | `4.57.3` |
| Attention implementation | `eager` |
| Image | Deterministic 448x448 red square |
| Prompt | `"Describe the image."` |
| Sampling | Greedy, fixed 16 generated tokens |
| Warmups per backend | `1` |
| Measured trials per backend | `3` |
| Backend order | PyTorch FP16, then TensorRT FP16 |
| Power mode | `25W`, unchanged |

The processor produced `input_ids=[1,210]`, finite FP16
`pixel_values=[784,1536]`, and finite `image_grid_thw=[[1,28,28]]`.

## Timing Boundaries

- `preprocess_ms` is host wall clock through chat templating, processor call,
  CUDA transfer, FP16 cast, and synchronization.
- PyTorch `vision_encoder_ms` is `visual_total_ms` minus the measured merger and
  three deepstack-merger projector events.
- PyTorch `projector_ms` is the sum of the final merger and three deepstack
  merger module events.
- TensorRT exposes only the unchanged combined visual adapter boundary, so its
  internal encoder and projector split is `UNKNOWN`.
- `prefill_ms` is a CUDA-event duration around the complete image-plus-text
  model forward.
- `decode_ms` is host wall clock around the 15 subsequent cached decode calls.
- Attribution stages are not all independent slices. Prefill contains the
  visual backend, embedding, language decoder, final norm, and LM head.
  `derived_language_decoder_prefill_ms` is computed by subtracting the visual
  boundary from prefill and is not an independently isolated measurement.

## PyTorch FP16 Stage Metrics

| Stage | Mean | Median | Stddev |
| --- | ---: | ---: | ---: |
| Preprocess | `12.423090331139974` ms | `13.521749002393335` ms | `3.1316847784775166` ms |
| Vision encoder | `230.17195530732474` ms | `231.47750234603882` ms | `2.6268925985810276` ms |
| Projector | `7.5359253485997515` ms | `7.591840028762817` ms | `0.16150999005626793` ms |
| Visual total | `237.70788065592447` ms | `238.83139038085938` ms | `2.5955027432383475` ms |
| LLM prefill | `424.7467854817708` ms | `426.558349609375` ms | `3.2815795740491596` ms |
| Derived language-decoder prefill | `187.03890482584634` ms | `187.00599670410156` ms | `0.8370026633254544` ms |
| Decode, 15 calls | `2034.242804996514` ms | `2032.8578019980341` ms | `8.180145322645732` ms |
| Decode per token | `135.6161869997676` ms | `135.5238534665356` ms | `0.5453430215097163` ms |
| Generation total | `2458.9895904782848` ms | `2459.5810685995966` ms | `10.90357388953243` ms |
| Throughput | `6.506822911297302` tokens/s | `6.50517285413563` tokens/s | `0.02886300967920807` tokens/s |

## TensorRT FP16 Stage Metrics

| Stage | Mean | Median | Stddev |
| --- | ---: | ---: | ---: |
| Preprocess | `10.465708997799084` ms | `10.63284499105066` ms | `0.8977059439823972` ms |
| Vision encoder | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` |
| Projector | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` |
| Visual adapter combined | `100.83846537272136` ms | `99.62620544433594` ms | `7.772144720546054` ms |
| LLM prefill | `283.19141642252606` ms | `280.9220275878906` ms | `14.841942437828124` ms |
| Derived language-decoder prefill | `182.3529510498047` ms | `181.2958221435547` ms | `7.069856480798775` ms |
| Decode, 15 calls | `2033.8434489967767` ms | `2033.0353979952633` ms | `2.776861646560205` ms |
| Decode per token | `135.5895632664518` ms | `135.53569319968423` ms | `0.18512410977068036` ms |
| Generation total | `2317.0348654193026` ms | `2312.4822635793826` ms | `17.121046476207546` ms |
| Throughput | `6.905627998580208` tokens/s | `6.9189719860745456` tokens/s | `0.05088854683085398` tokens/s |

The TensorRT combined visual boundary is `0.42421170511667733` of the PyTorch
visual total on this workload. This observation is descriptive only and is not
an optimization claim.

## Attribution

| Backend | Prefill share of generation | Decode share of generation | Visual share of prefill | Vision encoder share of prefill | Projector share of prefill | Derived language-decoder prefill share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP16 | `0.17273224218861197` | `0.8272677578113881` | `0.5596460968769977` | `0.5419039370627637` | `0.017742159814234017` | `0.4403539031230023` |
| TensorRT FP16 | `0.12222147394026299` | `0.877778526059737` | `0.3560788199253496` | `UNKNOWN` | `UNKNOWN` | `0.6439211800746503` |

For this fixed workload, decode is the dominant generation stage on both
backends. Within prefill, the PyTorch visual path is larger than the derived
language-decoder boundary. After replacing that visual boundary with TensorRT,
the derived language-decoder boundary becomes the larger prefill component. The
TensorRT encoder/projector split remains `UNKNOWN`.

Mean deltas, TensorRT minus PyTorch, were `-141.55536905924475` ms for prefill,
`-0.3993559997372813` ms for 15 decode calls, and `-141.95472505898215` ms for
generation total. Mean throughput changed by
`+0.39880508728290565` tokens/s. These are not robust deployment claims with
only three trials per backend.

## Output Agreement

All three PyTorch trials and its warmup produced:

```text
ThisThis is a simple, geometric image of a red square. The square is
```

All three TensorRT trials and its warmup produced:

```text
ThisThis is a simple, geometric image featuring a solid red square centered on a
```

The first divergence was token index `8`: PyTorch emitted `315` while TensorRT
emitted `16445`. The cause is `UNKNOWN`. No rerun, logit diagnosis, or
correctness gate was authorized. This divergence limits cross-backend
behavioral comparability but does not invalidate the fixed-length stage latency
attribution.

## Memory

All values are PyTorch CUDA allocator snapshots. They do not include
TensorRT-managed device allocations and are not profiler DRAM counters.

| Snapshot | Allocated | Reserved |
| --- | ---: | ---: |
| After model load | `4,255,079,424` B | `4,324,327,424` B |
| After PyTorch backend | `4,263,599,616` B | `4,466,933,760` B |
| After TensorRT injection | `3,500,045,312` B | `4,290,772,992` B |
| After TensorRT backend | `3,503,256,576` B | `4,332,716,032` B |

The process peak allocator snapshot was `4,364,569,600` bytes allocated and
`4,466,933,760` bytes reserved. Power was not requested or measured and is
therefore `UNKNOWN`.

## Execution Attempts

The first execution failed before any TensorRT backend trial because the
TensorRT adapter omitted `spatial_merge_size`. The failed result, console, and
exit code are preserved. The only correction was to retain that original
visual-model attribute in the adapter. The frozen protocol was unchanged, and
the corrected run restarted from model loading.

## Limitations And Non-Claims

- Three measured generations per backend are attribution evidence, not a formal
  statistical benchmark.
- The result applies only to the deterministic image, prompt, fixed vision
  boundary, eager PyTorch decoder, unchanged C1 TensorRT engine, and 16-token
  greedy workload.
- TensorRT internal encoder and projector timings are `UNKNOWN`.
- Cross-backend token divergence at index 8 is recorded but remains
  `UNKNOWN`; no numerical root cause was diagnosed.
- No full-vocabulary logits or power measurements were requested or collected.
- This result does not establish optimization benefit, deployment readiness, or
  realistic-image robustness.

## Evidence

- Frozen protocol: `artifacts/phase9_3B1_20260916T094011Z/protocol.json`
- Raw result: `artifacts/phase9_3B1_20260916T094011Z/phase9_3B1_result.json`
- Console log: `artifacts/phase9_3B1_20260916T094011Z/console.log`
- Exit code: `artifacts/phase9_3B1_20260916T094011Z/exit_code.txt`
- Command log: `artifacts/phase9_3B1_20260916T094011Z/command_log.md`
- Preserved failed attempt:
  `artifacts/phase9_3B1_20260916T094011Z/failed_attempt1_adapter_contract_failure/`
- Harness:
  `src/phase9_3B1/run_end_to_end_stage_breakdown.py`

The next action is to stop and await Gate review. No further rerun, diagnosis,
optimization, engine rebuild, benchmark sweep, quantization, decoder change,
CUDA change, or environment change is authorized by this report.
