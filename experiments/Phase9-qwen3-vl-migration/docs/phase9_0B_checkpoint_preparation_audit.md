# Phase 9.0-B Qwen3-VL Checkpoint Preparation Audit

Date: 2026-09-15 (Asia/Shanghai)

## Scope And Authorization

This was an authorized checkpoint preparation and read-only model audit. It
downloaded the selected `Qwen/Qwen3-VL-2B-Instruct` checkpoint to Jetson,
recorded its identity, calculated checksums, and inspected the config
architecture and safetensors metadata. It did not perform ONNX conversion,
TensorRT engine build, benchmark, profiling, quantization, or any environment
modification.

## Gate

**PASS / BOUNDED**

The checkpoint is now present at a pinned revision with a complete file
manifest, SHA-256 checksums, and on-device metadata evidence. This gate is only
for checkpoint preparation. It is not evidence that the model runs correctly,
that TensorRT migration works, or that any performance conclusion exists.

## Model Identity

| Field | Value |
| --- | --- |
| Repository | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| License | Apache-2.0 |
| Local path | `/home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203` |
| Model type | `qwen3_vl` |
| Architecture | `Qwen3VLForConditionalGeneration` |
| Checkpoint dtype | BF16 |
| Total parameters | `2,127,532,032` |
| File count | `12` |
| Directory size | `4,266,666,579` bytes |

The official Hugging Face endpoint was unreachable from Jetson. The download
therefore used `https://hf-mirror.com` with the same repository and an explicit
revision pin. No package or environment configuration was changed; the mirror
endpoint was passed only to the downloader process.

## Checkpoint And Checksum Evidence

The single weight file is:

```text
model.safetensors
4,255,140,312 bytes
SHA-256 7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0
```

The pinned `config.json` is:

```text
config.json
1,505 bytes
SHA-256 bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b
```

The complete 12-file SHA-256 manifest is stored in
`experiments/Phase9-qwen3-vl-migration/artifacts/phase9_0B_20260915T132321Z/sha256sums.txt`.

## Architecture Evidence

### Vision Encoder

| Field | Value |
| --- | --- |
| Depth | 24 |
| Hidden size | 1024 |
| Intermediate size | 4096 |
| Attention heads | 16 |
| Patch size | 16 |
| Spatial merge size | 2 |
| Temporal patch size | 2 |
| Output hidden size | 2048 |
| DeepStack visual indexes | 5, 11, 17 |

### Text Decoder

| Field | Value |
| --- | --- |
| Layers | 28 |
| Hidden size | 2048 |
| Intermediate size | 6144 |
| Attention heads | 16 |
| KV heads | 8 |
| Head dimension | 128 |
| GQA repeat | 2 |
| RMSNorm epsilon | 1e-6 |
| RoPE theta | 5,000,000 |
| MRoPE | interleaved, sections 24/20/20 |
| Max positions | 262,144 |
| Vocab size | 151,936 |
| Tied embeddings | true |

Special image/video tokens are image `151655`, video `151656`, vision start
`151652`, and vision end `151653`.

## Safetensors Metadata

The read-only metadata scan found `625` tensors and `2,127,532,032` parameters,
all BF16. Aggregate module counts include:

| Module prefix | Parameters |
| --- | ---: |
| `model.language_model.layers` | `1,409,408,000` |
| `model.visual.blocks` | `302,309,376` |
| `model.language_model.embed_tokens` | `311,164,928` |
| `model.visual.deepstack_merger_list` | `75,540,480` |
| `model.visual.merger` | `25,174,016` |
| `model.visual.pos_embed` | `2,359,296` |
| `model.visual.patch_embed` | `1,573,888` |
| `model.language_model.norm` | `2,048` |

Derived averages are `50,336,000` parameters per text decoder layer and
`12,596,224` parameters per visual block. These are aggregate metadata facts,
not runtime execution evidence.

## Limitations

- This audit did not load the full model into GPU memory.
- It did not run a forward pass, tokenizer/processor round trip, benchmark, or
  profile.
- It did not verify whether the checkpoint can be converted, built, or executed
  by TensorRT or TensorRT-LLM.
- It did not establish memory, latency, power, or correctness behavior.
- The mirror was used only because the official endpoint timed out; the revision
  and checksums are the identity anchors.

## Evidence

- Manifest:
  `experiments/Phase9-qwen3-vl-migration/artifacts/phase9_0B_20260915T132321Z/model_manifest.json`
- Config copy:
  `experiments/Phase9-qwen3-vl-migration/artifacts/phase9_0B_20260915T132321Z/config.json`
- Processor configs:
  `preprocessor_config.json` and `video_preprocessor_config.json`
- Safetensors metadata:
  `experiments/Phase9-qwen3-vl-migration/artifacts/phase9_0B_20260915T132321Z/safetensors_metadata.json`
- Checksums:
  `experiments/Phase9-qwen3-vl-migration/artifacts/phase9_0B_20260915T132321Z/sha256sums.txt`
