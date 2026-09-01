# Phase 1.0 - Qwen3 Baseline Runtime Feasibility Audit

Audit date: `2026-09-01`

Gate P1.0: `PASS WITH CONSTRAINTS`

## Decision

The safe Phase 1.1 baseline is Path C: Qwen3-0.6B through Hugging Face Transformers on the already working NVIDIA Jetson PyTorch 2.5/CUDA 12.6 stack, initially in BF16 with batch 1 and a bounded short context. This is a correctness/reference baseline, not the final optimized runtime.

There is no clean officially supported TensorRT-LLM intersection for all three current requirements: Qwen3, Jetson Orin SM87, and the installed JetPack 6.2/CUDA 12.6/TensorRT 10.3 stack. Latest TensorRT-LLM supports Qwen3 but not SM87 in its tested hardware matrix and requires a substantially newer software stack. The Jetson-specific 0.12 branch targets SM87 and CUDA 12.6 but does not recognize Qwen3.

## Repository and device state

Phase 0 and Exp01-Exp04 are `PASS / CLOSED`. Windows, GitHub and Jetson started clean and synchronized at `exp/04-gemm@9ede5e03773d23194f06059c339f32d539f7b7be`. The audit branch was created from that commit as `phase/01-qwen3-baseline`.

The target is a Jetson Orin Nano Super (`aarch64`, CC 8.7) with 7.4 GiB unified RAM and 3.7 GiB zram swap. It runs Ubuntu 22.04.5, L4T 36.4.3 and the JetPack 6.2 compute stack: CUDA 12.6 and TensorRT 10.3. The `nvidia-jetpack` metapackage itself is absent, so the JetPack identification is based on NVIDIA's official L4T 36.4.3 mapping and matching installed component versions.

The installed Python is system Python 3.10.12 with no active project venv. NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08` reports CUDA 12.6, Orin CC 8.7 and BF16 support. Transformers, Accelerate, Safetensors, Hugging Face Hub and TensorRT-LLM are absent. Docker and NVIDIA Container CLI are absent. No model weights, HF cache, engine, checkpoint or TensorRT-LLM source tree was found in the bounded user-directory search.

## Qwen3-0.6B metadata

| Field | Official metadata |
| --- | --- |
| Architecture | `Qwen3ForCausalLM` |
| Model type | `qwen3` |
| Parameters | model card: 0.6B; HF safetensors metadata: 751,632,384 total |
| Non-embedding parameters | 0.44B (model card) |
| Layers | 28 |
| Hidden size | 1024 |
| Intermediate size | 3072 |
| Q heads / KV heads | 16 / 8 (GQA) |
| Head dimension | 128 |
| RMSNorm epsilon | `1e-6` |
| RoPE theta | `1,000,000` |
| Context | model card: 32,768; config `max_position_embeddings`: 40,960 |
| Weight dtype | BF16 |
| Transformers | `>=4.51.0`; older versions raise `KeyError: 'qwen3'` |
| License | Apache-2.0 |

The context discrepancy is retained rather than normalized away: the public card advertises 32,768 while the checked config declares 40,960 positions.

## TensorRT-LLM compatibility intersection

| Intersection | Status | Evidence and constraint |
| --- | --- | --- |
| Latest TensorRT-LLM x Qwen3 | `SUPPORTED` | Official main source registers `Qwen3ForCausalLM`; the Qwen guide explicitly includes Qwen3-0.6B. |
| Latest TensorRT-LLM x Jetson Orin | `UNOFFICIAL` | Linux aarch64 is an OS target, but the official GPU matrix lists Ampere SM80/86, not Orin SM87. Unlisted architectures receive community-level support. |
| `v0.12.0-jetson` x Jetson | `SUPPORTED` | NVIDIA's Jetson README specifies JetPack 6.1, CUDA 12.6 and `--cuda_architectures 87`. The current device is JetPack 6.2/L4T 36.4.3, so exact 6.2 validation remains unperformed. |
| `v0.12.0-jetson` x Qwen3 | `UNSUPPORTED` | Its support matrix ends at Qwen2. `QWenConfig` accepts only `qwen`, `qwen2`, `qwen2_moe`; model mapping lacks `Qwen3ForCausalLM`. |

The official GitHub release and PyPI distribution report TensorRT-LLM 1.2.1 (published 2026-04-20) with Python >=3.10, TensorRT 10.14.1, PyTorch 2.9.1-2.10, Transformers 4.57.3 and CUDA 13-era dependencies. NVIDIA's versioned 1.2.1 Linux installation page calls for CUDA Toolkit 13.1 and is tested on Ubuntu 24.04/Python 3.12. Its versioned hardware page lists B200/GB200/B300/GB300/DGX Spark, H100/H200/GH200, L20/L40/L40S and A100; Jetson Orin/SM87 is absent. Installing it on this Jetson would require disruptive stack changes and is not a safe baseline action.

The Jetson branch tip is `9d38cb7d1e77473ba27d5ee3b099e2e8eeacffe4` dated 2025-12-16; its tip change is a memory-profiler fix. The branch reports TensorRT-LLM 0.12.0, TensorRT 10.3, PyTorch 2.4.0a0-2.4.0 and Transformers 4.38.2-4.42.4. This shows post-release maintenance but no Qwen3 backport. Its Transformers ceiling predates Qwen3's minimum 4.51.0, independently confirming the missing native intersection. A Qwen3 backport would need model registration/config translation and careful validation of Qwen3 RMSNorm, RoPE, GQA and checkpoint loading; it is source modification, not configuration.

## Theoretical memory budget

All values below are `THEORETICAL ESTIMATE`, not runtime measurements. Weight lower bounds use the HF safetensors total of 751,632,384 parameters and exclude metadata/alignment/runtime duplication.

| Weight precision | Bytes | GiB lower bound |
| --- | ---: | ---: |
| FP32 | 3,006,529,536 | 2.800 |
| FP16 | 1,503,264,768 | 1.400 |
| BF16 | 1,503,264,768 | 1.400 |
| INT8 | 751,632,384 | 0.700 |
| INT4 | 375,816,192 | 0.350 |

For batch 1 FP16/BF16 KV cache:

```text
bytes/token = layers * 2(K,V) * KV heads * head_dim * bytes
            = 28 * 2 * 8 * 128 * 2
            = 114,688 bytes = 112 KiB/token
```

| Sequence length | KV cache MiB | KV cache GiB |
| ---: | ---: | ---: |
| 128 | 14 | 0.014 |
| 512 | 56 | 0.055 |
| 1,024 | 112 | 0.109 |
| 2,048 | 224 | 0.219 |
| 4,096 | 448 | 0.438 |
| 8,192 | 896 | 0.875 |

Actual memory also includes activations, allocator fragmentation, CUDA context, kernels, tokenizer/loading overhead, runtime workspace, engine allocations and temporary build memory. Batch size multiplies the KV term. Therefore these numbers demonstrate only a plausible short-context starting point, not a proven maximum context or successful deployment.

## Runtime candidate matrix

| Path | Qwen3 | Jetson | Modification | Risk | Learning / future quantization | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| A: latest TensorRT-LLM | Native | SM87 not in official tested matrix; current stack mismatched | CUDA/TensorRT/PyTorch/container or source changes | Very high | High if platform gap is solved | Not recommended now |
| B: `v0.12.0-jetson` | Not native | Explicit SM87 route, exact README baseline is JetPack 6.1 | Qwen3 model/converter backport and newer Transformers reconciliation | High | High engineering value, poor first baseline reproducibility | Not recommended for Phase 1.1 |
| C: HF PyTorch | Native with Transformers >=4.51 | Existing CUDA PyTorch works and reports BF16 | Install missing user-space packages later in an isolated environment; no model code backport | Medium | Best correctness reference; later quantization comparison baseline | Recommended |
| D: plain TensorRT/ONNX | No turnkey Qwen3 decoder runtime established | TensorRT 10.3 present | Export/custom ops plus autoregressive loop, KV cache, dynamic shapes and sampling runtime | High | Useful later for custom-runtime learning | Not recommended as first baseline |

Alternative runtimes such as llama.cpp or MLC LLM may be useful fallback comparisons, but this audit does not redirect the project away from NVIDIA GPU inference. They were not installed or executed.

## Gate and Phase 1.1 recommendation

Gate P1.0 is `PASS WITH CONSTRAINTS`: a safe reference path exists and the memory lower bounds are compatible with a bounded baseline, but no native, officially supported TensorRT-LLM/Qwen3/Jetson-SM87 intersection exists in the current environment.

Recommended Phase 1.1:

1. Preserve the JetPack/PyTorch stack.
2. Create an isolated Python environment designed to reuse the NVIDIA PyTorch build.
3. Install only a Qwen3-capable Transformers stack after explicit approval.
4. Download Qwen3-0.6B only after explicit approval and record exact revision/checksums.
5. Establish a batch-1 BF16 correctness and memory baseline at short context before any quantization or TensorRT-LLM backport work.

Phase 1.1 was not executed.

## Official sources

- NVIDIA JetPack 6.2: https://developer.nvidia.com/embedded/jetpack-sdk-62
- NVIDIA JetPack 6.2 release notes: https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/release-notes/index.html
- TensorRT-LLM v1.2.1 release: https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1
- TensorRT-LLM v1.2.1 supported hardware: https://nvidia.github.io/TensorRT-LLM/1.2.1/supported-hardware.html
- TensorRT-LLM v1.2.1 supported models: https://nvidia.github.io/TensorRT-LLM/1.2.1/models/supported-models.html
- TensorRT-LLM v1.2.1 Linux installation: https://nvidia.github.io/TensorRT-LLM/1.2.1/installation/linux.html
- TensorRT-LLM build workflow: https://nvidia.github.io/TensorRT-LLM/architecture/workflow.html
- Current Qwen3 implementation: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/tensorrt_llm/_torch/models/modeling_qwen3.py
- Current Qwen3 guide: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/examples/models/core/qwen/README.md
- Jetson branch README: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.12.0-jetson/README4Jetson.md
- Jetson branch support matrix: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.12.0-jetson/docs/source/reference/support-matrix.md
- Jetson branch Qwen config: https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.12.0-jetson/tensorrt_llm/models/qwen/config.py
- Jetson branch tip: https://github.com/NVIDIA/TensorRT-LLM/commit/9d38cb7d1e77473ba27d5ee3b099e2e8eeacffe4
- Qwen3-0.6B config: https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
- Qwen3-0.6B model card: https://huggingface.co/Qwen/Qwen3-0.6B
- TensorRT-LLM official PyPI distribution: https://pypi.org/project/tensorrt-llm/

## Stop point

- No package was installed or removed.
- No model weights were downloaded.
- No TensorRT-LLM source was cloned or built.
- No engine was built.
- No inference was run.
- No power, clock, swap, CUDA, TensorRT, PyTorch, SSH, Git automation or system configuration was changed.
- Phase 1.1, Phase 2 quantization and Exp05 Softmax were not started.
