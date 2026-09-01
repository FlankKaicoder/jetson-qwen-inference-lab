# Phase 1.1 - Qwen3-0.6B Hugging Face BF16 Reference

## Purpose

Establish a deterministic, batch-1, short-context functional and memory reference for Qwen3-0.6B on the Jetson Orin Nano Super. This is not a latency, throughput, power, or model-quality benchmark.

## Environment

- Formal venv: `/home/nvidia/.venvs/jetson-qwen-phase1-hf`, `system-site-packages=true`.
- Python 3.10.12; NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08` from `/usr/local/lib/python3.10/dist-packages/torch`; CUDA 12.6; Orin SM87; BF16 supported.
- Transformers 4.57.3 and Accelerate 1.14.0. Pillow 11.3.0 was added only inside the formal venv because the system Pillow lacked `PIL.Image.Resampling` required during Transformers model import.
- The failed partial environment `/home/nvidia/.venvs/jetson-qwen-phase1` remains preserved and untouched.

Gate A remains `PASS`; `pip check` passes and the NVIDIA PyTorch/CUDA stack is unchanged.

## Exact Model Identity

- Repository: `Qwen/Qwen3-0.6B` (post-trained conversational model).
- Resolved revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Jetson snapshot: `/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca`, outside Git.
- `model.safetensors`: 1,503,300,328 bytes; SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Full ten-file SHA256 manifest: `artifacts/model_manifest_20260901T065018Z.txt`.

Jetson resolved DNS and had a default route but could not establish HTTPS to Hugging Face (`Errno 101` / timeout). Windows resolved the exact model SHA through the public API, downloaded that exact revision with `snapshot_download`, and transferred the complete ten-file snapshot to the external Jetson model path. Jetson then recomputed every SHA256. No floating `main` download was used.

Config identity matches the Phase 1.0 audit: `Qwen3ForCausalLM`, `qwen3`, 28 layers, hidden size 1024, intermediate size 3072, 16 query heads, 8 KV heads, head dimension 128, RMSNorm epsilon `1e-6`, RoPE theta 1,000,000, BF16, and `max_position_embeddings=40960`.

## Loader Design

`src/hf_bf16_reference.py` uses local files only, BF16, batch 1, `device_map={"": "cuda:0"}`, inference mode, cache-enabled generation, non-thinking chat templates, greedy decoding, and machine-readable JSON output. It does not use remote code, quantization, compilation, ONNX, TensorRT, or external attention packages.

The first preserved run failed because Transformers SDPA passed `enable_gqa`, which the installed PyTorch 2.5 SDPA API does not accept. The reference therefore explicitly uses Transformers' built-in `eager` attention implementation. This is a functional compatibility choice, not a performance selection. Failure evidence is retained in `artifacts/phase1_1_reference_failed_sdpa_20260901T065911Z.log`.

## BF16 and Device Placement

- Loaded class: `Qwen3ForCausalLM`.
- Loaded unique parameter devices: `cuda:0` only; `hf_device_map={"": "cuda:0"}`.
- Loaded unique parameter dtype: `torch.bfloat16` only.
- Loaded unique parameter count: 596,049,920. The safetensors file contains 751,632,384 serialized elements; config has `tie_word_embeddings=true`, so the loaded module shares embedding/output weight storage. This is not a model identity mismatch.
- No CPU or disk offload occurred.

On Jetson, `cuda:0` describes PyTorch device placement within unified physical memory; it does not imply separate discrete VRAM.

## Forward Correctness

The non-thinking `Hello.` chat input had 14 tokens. Forward produced BF16 logits with shape `[1, 14, 151936]`; every value was finite, with no CUDA error or OOM.

## Generation Correctness

All cases used batch 1, `enable_thinking=False`, `do_sample=False`, `max_new_tokens=32`, and inference mode. Greedy decoding is used only for deterministic runtime smoke testing.

| Case | Input tokens | Generated tokens | Stop | Output | Sanity |
| --- | ---: | ---: | --- | --- | --- |
| Arithmetic | 26 | 8 | EOS | `2 + 3 = 5` | PASS |
| Chinese | 22 | 2 | EOS | `北京` | PASS |
| CUDA | 23 | 11 | EOS | `CUDA stands for **Compute Unified Device Architecture**.` | PASS |

These outputs are not an accuracy or language-quality benchmark.

## Memory Methodology

The runner records three distinct views: PyTorch CUDA allocator counters, `/proc/meminfo` unified system RAM/swap, and a scoped 200 ms `tegrastats` sampler stopped only by its recorded PID. M0-M5 are process start, post-load, post-forward, and post-generation 1/2/3. No cache clearing, swap change, power change, or clock change was used.

| Stage | MemAvailable bytes | Swap used bytes | Torch allocated | Torch reserved | Torch max allocated | Torch max reserved | CUDA mem free |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 4,100,358,144 | 294,387,712 | 0 | 0 | 0 | 0 | 4,075,433,984 |
| M1 | 3,239,448,576 | 294,387,712 | 1,192,638,976 | 1,509,949,440 | 1,503,803,392 | 1,509,949,440 | 3,234,508,800 |
| M2 | 2,749,599,744 | 362,545,152 | 1,201,159,680 | 1,509,949,440 | 1,503,803,392 | 1,509,949,440 | 2,745,352,192 |
| M3 | 2,747,166,720 | 362,545,152 | 1,201,161,216 | 1,514,143,744 | 1,503,803,392 | 1,514,143,744 | 2,744,971,264 |
| M4 | 2,747,166,720 | 362,545,152 | 1,201,161,216 | 1,514,143,744 | 1,503,803,392 | 1,514,143,744 | 2,744,971,264 |
| M5 | 2,746,023,936 | 362,545,152 | 1,201,161,216 | 1,514,143,744 | 1,503,803,392 | 1,514,143,744 | 2,743,828,480 |

- Model-load MemAvailable decrease: 860,909,568 bytes.
- Model-load PyTorch allocated increase: 1,192,638,976 bytes.
- Post-load to post-generation PyTorch allocated increase: 8,522,240 bytes.
- Minimum checkpoint MemAvailable: 2,746,023,936 bytes (2.557 GiB).
- Successful-run swap increase: 68,157,440 bytes (65 MiB), stable after forward.
- Tegrastats: 79 samples; peak RAM 4,869/7,620 MB; peak swap 346/3,810 MB.

The optional approximately 128-token case was skipped because remaining memory headroom was measurable but not generous; skipping is permitted and avoids turning Phase 1.1 into a context sweep.

## Theoretical Versus Observed

Phase 1.0's 1.400 GiB BF16 weight number is a serialized-parameter lower bound using 751,632,384 elements. The model file is 1,503,300,328 bytes (1.400 GiB). Loaded tied parameters report about 1.111 GiB allocated at M1, while allocator peak/reservation is about 1.401/1.410 GiB. The system MemAvailable delta at load is about 0.802 GiB because Jetson uses unified memory and file cache/accounting differs from PyTorch allocator accounting. CUDA context, mapped/file-backed pages, allocator behavior, tokenizer/runtime objects, activations, KV cache, and other processes explain why these views are not expected to match exactly.

`torch.cuda.memory_allocated` is an allocator view and is not labeled as discrete VRAM usage.

## Gates

- Gate A - isolated environment: `PASS`.
- Gate B - exact model acquisition: `PASS`.
- Gate C - BF16 functional correctness: `PASS`.
- Gate D - bounded memory viability: `PASS`. The run completed without OOM/offload; swap growth was modest and stable, with 2.557 GiB minimum checkpoint MemAvailable.
- Phase 1.1: `PASS / CLOSED`.

## Limitations and Next Dependency

- Eager attention is required by the current PyTorch 2.5/Transformers 4.57.3 compatibility intersection; performance implications are not measured here.
- The run is short-context and batch 1. It does not establish maximum context length or concurrency capacity.
- Tegrastats RAM is system-wide and cannot attribute every byte to this process.
- No TTFT, prefill/decode latency, TPOT, throughput, power, or quality conclusion is made.
- Phase 1 remains in progress. Phase 1.2 formal benchmark requires separate authorization and methodology.
