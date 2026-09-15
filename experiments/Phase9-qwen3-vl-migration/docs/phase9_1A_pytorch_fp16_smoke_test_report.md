# Phase 9.1-A Qwen3-VL PyTorch FP16 Inference Smoke Test

Date: 2026-09-15 (Asia/Shanghai)

## Scope And Authorization

This was an explicitly authorized PyTorch FP16 inference smoke test. It loaded
the pinned local Qwen3-VL checkpoint, verified processor loading and the image
input pipeline, and performed one successful greedy inference generation. It
did not use TensorRT, ONNX, quantization, optimization, or a benchmark sweep.
No package, driver, model, or environment configuration was changed.

## Gate

**PASS / BOUNDED — DEFAULT_SDPA_INCOMPATIBLE**

The smoke test passed with eager attention. The default SDPA path reached the
model forward pass but failed because Transformers passed an unsupported
`enable_gqa` argument to the installed NVIDIA PyTorch SDPA implementation. This
result is evidence of one bounded image-to-text generation only. It is not
correctness validation, performance evidence, TensorRT evidence, or a claim that
eager attention is optimal.

## Execution Attempts

| Attempt | Attention | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Default SDPA | Model load succeeded; generation failed with `scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`; exit `1` | `attempt1_sdpa_failure/` |
| 2 | `eager` | Processor, image pipeline, FP16 model load, and one generation succeeded; exit `0` | `attempt2_eager_success/` |

The failed attempt was preserved because it establishes the exact runtime
incompatibility. The successful attempt is the Phase 9.1-A gate evidence.

## Runtime Dependency Status

| Component | Status |
| --- | --- |
| Host | `nvidia-desktop`, Linux `5.15.148-tegra-aarch64` |
| Python | `3.10.12` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| CUDA | `12.6`, available |
| Device | Orin, SM capability `(8, 7)` |
| Transformers | `4.57.3` |
| Safetensors | `0.8.0` |
| Pillow | `11.3.0` |
| Accelerate | `1.14.0` |
| NumPy | `1.23.5` |
| Runtime venv | `/home/nvidia/.venvs/jetson-qwen-phase1-hf` |

The compatibility defect is in the default Transformers SDPA bridge and the
installed NVIDIA PyTorch build, not in the checkpoint checksum or processor
contract.

## Checkpoint Identity

| Field | Value |
| --- | --- |
| Repository | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Local path | `/home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203` |
| `config.json` SHA-256 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `model.safetensors` SHA-256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| Weight-file size | `4,255,140,312` bytes |

Both hashes match the Phase 9.0-B checkpoint manifest.

## Processor And Image Pipeline

Processor loading succeeded as `Qwen3VLProcessor` with `Qwen2TokenizerFast` and
`Qwen2VLImageProcessorFast`. A deterministic RGB `448 x 448` test image with a
red square was passed through the chat template and image processor.

The successful processor output was finite:

| Tensor | Dtype | Shape |
| --- | --- | --- |
| `input_ids` | int64 | `[1, 220]` |
| `attention_mask` | int64 | `[1, 220]` |
| `pixel_values` | float16 | `[784, 1536]` |
| `image_grid_thw` | int64 | `[1, 3]` |

These values verify the image preprocessing and tensor-production path only;
they do not establish image quality, robustness, or broader prompt behavior.

## FP16 Model Load

The model loaded as `Qwen3VLForConditionalGeneration` on `cuda:0` in eager
attention mode. It reported:

| Field | Value |
| --- | --- |
| Model dtype | `torch.float16` |
| Parameters | `2,127,532,032` |
| FP16 parameters | `2,127,532,032` |
| Device | `cuda:0` |

All model parameters were FP16. The count matches the pinned checkpoint
metadata.

## One Inference Generation

The prompt asked for a one-word color answer about the red square. Generation
used greedy decoding with a four-token cap.

| Field | Value |
| --- | --- |
| Sampling rule | Greedy |
| Requested new tokens | `4` |
| Generated tokens | `2` |
| Token IDs | `[2518, 151645]` |
| Decoded text | `" red"` |
| Score tensors | `2` |
| Scores finite | `true` |

The second token is the end-of-turn token. The semantic answer matched the
constructed test image. This is a single smoke-test observation, not a
correctness suite or performance measurement.

## GPU And Host Memory

| Snapshot | CUDA allocated | CUDA reserved | CUDA peak allocated | CUDA peak reserved | Host available |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before run | `0` | `0` | N/A | N/A | `4,494,307,328` B |
| After model load | `4,260,212,224` B | `4,347,396,096` B | N/A | N/A | `828,424,192` B |
| After generation | `4,295,296,000` B | `4,471,128,064` B | `4,366,713,344` B | `4,471,128,064` B | `320,823,296` B |

The process maximum RSS after load and generation was `1,875,030,016` bytes.
Host memory after generation was very tight, but no OOM occurred. These are
point/peak allocator snapshots, not a full power or benchmark memory profile.

## Limitations

- This was one deterministic synthetic image and one prompt.
- Wall-clock values were recorded only as execution diagnostics; no timing,
  throughput, latency, or optimization conclusion is valid.
- Eager attention is not a performance recommendation.
- No TensorRT, ONNX, quantization, benchmark sweep, or optimization work was
  performed.
- The Jetson repository checkout remained frozen and was not synchronized.

## Evidence

- Successful result:
  `attempt2_eager_success/smoke_result.json`
- Failed SDPA result:
  `attempt1_sdpa_failure/smoke_result.json`
- Runtime dependency probe:
  `runtime_dependency_probe.json`
- Smoke-test source:
  `experiments/Phase9-qwen3-vl-migration/src/phase9_1A/run_fp16_smoke.py`
