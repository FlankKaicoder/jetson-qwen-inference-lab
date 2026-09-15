# Phase 9.0 Qwen3-VL Model And Environment Audit

Date: 2026-09-15 (Asia/Shanghai)

## Scope And Authorization

This was a read-only model and environment audit. It did not download a model,
convert weights, build or deserialize an engine, run a benchmark, profile, or
change an environment. It did not modify the Jetson repository, model area,
package set, power mode, clocks, or device state.

## Gate

**BLOCKED / NO_LOCAL_MODEL**

The selected candidate `Qwen/Qwen3-VL-2B-Instruct` was not found on Jetson.
There was no local model path, no Hugging Face cache directory, and no file
matching the Qwen3-VL model family outside the installed Transformers library.
Therefore all checkpoint-dependent model facts remain `UNKNOWN`. This audit does
not authorize a download or infer model behavior from the upstream repository.

## Local Model Audit

The checked roots included `/home/nvidia/models` and
`/home/nvidia/.cache/huggingface/hub`. The Hugging Face cache directory was not
present. A broader read-only scan of `/home/nvidia` found only the frozen
Qwen3-0.6B checkpoint:

```text
/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca
```

No Qwen3-VL checkpoint, `config.json`, safetensors, tokenizer manifest, or
engine matching Qwen3-VL was found. The only engine files belonged to prior PPE
and Phase 2/Phase 8 diagnostics; none were deserialized or executed.

Consequently, these required Phase 9.0 fields are `UNKNOWN`:

- exact model revision and SHA-256 manifest
- safetensors metadata and tensor-name inventory
- total and per-module parameter counts
- vision encoder and projector implementation details
- decoder dimensions, GQA/RMSNorm/RoPE details, and tied-embedding state
- image/video input contract and fixed prefill/decode shapes

## Environment Audit

### Device And OS

| Field | Value |
| --- | --- |
| Device | NVIDIA Orin (nvgpu), SM `8.7` |
| Kernel | Linux `5.15.148-tegra`, aarch64 |
| L4T | `R36 (release), REVISION: 4.3` |
| Driver | `540.4.0` |
| CUDA reported by driver | `12.6` |
| Power mode | `25W` |
| Disk | `176 GiB` available on `/` |
| Memory | `7989903360` bytes total, `5166972928` bytes available at audit time |

The `jetson_clocks --show` path required root and was not attempted. Current
SM/memory clocks are therefore `UNKNOWN` for this audit. `nvidia-smi` also
reported its clock and power-limit fields as `N/A` on this platform. A single
2-second `tegrastats` read was used only to observe idle-state counters; no
state was changed.

### Python And Libraries

System Python is `3.10.12`. It has NVIDIA PyTorch
`2.5.0a0+872d972e41.nv24.08` with CUDA `12.6` and TensorRT `10.3.0`, but not
Transformers, safetensors, or TensorRT-LLM.

The Phase 1 HF environment
`/home/nvidia/.venvs/jetson-qwen-phase1-hf` has:

| Package | Version |
| --- | --- |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| Transformers | `4.57.3` |
| Safetensors | `0.8.0` |
| Tokenizers | `0.22.2` |
| TensorRT | `10.3.0` |
| NumPy | `1.23.5` |
| Pillow | `11.3.0` |

`pip check` reported no broken requirements in the Phase 1 HF environment or
the Phase 2 TRT-tools environment.

### Qwen3-VL Library Support

Transformers `4.57.3` exposes `transformers.models.qwen3_vl`. The audit
confirmed the following classes are importable without loading a model:

```text
Qwen3VLForConditionalGeneration
Qwen3VLProcessor
Qwen3VLVideoProcessor
```

`qwen3_vl` is also present in `CONFIG_MAPPING`. This is library-support
evidence only. It is not evidence that the model checkpoint is present, loadable
on device, or that the full multimodal execution path works on Jetson.

TensorRT-LLM was not found in the audited Python environments. Its migration
feasibility therefore remains `UNKNOWN`.

## Interpretation

The environment contains the minimum PyTorch/Transformers/TensorRT pieces to
begin a later checkpoint audit, but the selected model is absent. Phase 9.0
cannot establish the model identity or architecture contract without either:

1. an explicitly authorized Qwen3-VL checkpoint download, or
2. a user-provided local checkpoint path.

No automatic download, conversion, engine build, or benchmark is authorized by
this audit.

## Next Action

Stop after this audit. The owner must decide whether to authorize a separate
Qwen3-VL download/checksum audit or provide a local checkpoint path. Only after
the checkpoint is present and audited may Phase 9.0 be reconsidered for a
`PASS` or `INCONCLUSIVE` decision. Phase 9.1 remains unauthorized.

## Evidence

Machine-readable manifest:
`experiments/Phase9-qwen3-vl-migration/artifacts/phase9_0_20260915T125952Z/manifest.json`
