# Phase 2.0 Quantization Backend Audit

## Purpose and stop point

The goal is to determine whether an INT8 or INT4 PyTorch-native backend can execute on Orin SM87 with the existing NVIDIA PyTorch 2.5/CUDA 12.6 stack. This is a feasibility audit, not a Qwen3 quantization or performance experiment.

No full Qwen3 model was quantized, no formal quantized benchmark was run, and no BF16 model file was modified.

## Frozen BF16 reference

Qwen/Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`, weight SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`; NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`, CUDA 12.6, Transformers 4.57.3, eager attention, BF16, batch 1, 25W. Phase 1 medians remain historical evidence and are not rerun here.

## Environment and installation

The isolated venv `/home/nvidia/.venvs/jetson-qwen-phase2-quant` uses system site packages. Its Torch resolves to `/usr/local/lib/python3.10/dist-packages/torch`, version `2.5.0a0+872d972e41.nv24.08`, CUDA `12.6`, device `Orin`, capability `(8, 7)`. The Phase 1 venv was not modified.

HF packages were installed with `--no-deps` at the already validated versions, including `regex==2026.9.3` and `hf-xet==1.6.0`; `pip check` then reported `No broken requirements found.` This does not alter Torch. TorchAO `0.12.0` dry-run (pip 24.0) planned only `torchao-0.12.0`; installation completed with `--no-deps`.

## TorchAO result

The wheel is pure Python (`py3-none-any`) and exposes source definitions for `quantize_`, `Int8WeightOnlyConfig`, `Int8DynamicActivationInt8WeightConfig`, and `Int4WeightOnlyConfig`. However, normal `import torchao` fails:

`ModuleNotFoundError: No module named 'torch._C._distributed_c10d'; 'torch._C' is not a package`

The failure occurs while importing TorchAO Float8 support through `torch.distributed._functional_collectives`. Because the package cannot pass its import gate on the unmodified NVIDIA Torch build, no quantized module was constructed and no CUDA forward was claimed.

The native operator probe independently found `torch.ops.aten._weight_int4pack_mm` present and INT8-related aten names including `_weight_int8pack_mm`; operator presence alone is not evidence that a TorchAO quantization path is callable.

## Micro-probe matrix

| Scheme | Quantization | CUDA forward | Finite | CPU fallback | Status |
| --- | --- | --- | --- | --- | --- |
| W8A16 | not reached | not reached | UNKNOWN | UNKNOWN | BLOCKED_BY_TORCHAO_IMPORT |
| A8W8 | not reached | not reached | UNKNOWN | UNKNOWN | BLOCKED_BY_TORCHAO_IMPORT |
| W4A16 | not reached | not reached | UNKNOWN | UNKNOWN | BLOCKED_BY_TORCHAO_IMPORT |

Planned dimensions were 1024->1024, 1024->2048, and 1024->3072 with BF16 inputs of shapes `[1,1024]` and `[32,1024]` on `cuda:0`. They were intentionally not executed after the import gate failed.

## Other backend survey

bitsandbytes is not installed or built. Jetson/L4T aarch64 support and CUDA 12.6/SM87 compatibility require a dedicated source-build validation; this phase therefore records `SOURCE_BUILD_REQUIRED / HIGHER ENGINEERING RISK`.

TensorRT 10.3 provides INT8 and weight-compression capabilities, but that does not provide a complete Qwen3 autoregressive runtime. It remains a future runtime backend candidate; no engine was built.

Orin SM87 supports BF16 and FP16 arithmetic. INT8/INT4 are storage/packing and kernel-path questions, not proof of native Tensor Core acceleration. FP8 and NVFP4 claims from newer hardware/framework documentation are not applicable to this Orin feasibility decision without an SM87-supported kernel.

## Gates

- Gate A - isolated environment: `PASS` (NVIDIA Torch/CUDA preserved; venv isolated).
- Gate B - TorchAO compatibility: `BLOCKED` (0.12.0 import ABI failure).
- Gate C - CUDA quantization micro-probe: `BLOCKED` (import gate prevents constructing paths).
- Gate D - backend decision: `INCONCLUSIVE`; defer formal selection until a backend that imports and executes on SM87 is independently validated.
- Phase 2.0 overall: `BLOCKED` at the TorchAO candidate, with survey evidence retained.

## Recommended next experiment order

Phase 2.1 should begin only after an explicitly authorized backend feasibility rerun or alternate backend is proven on the unchanged stack. Candidate order is TorchAO-compatible repair/alternate build (if NVIDIA supplies a supported wheel), then a controlled TensorRT runtime investigation. bitsandbytes remains deferred pending a Jetson-specific source-build plan. No Phase 2.1 work is started by this audit.

## Explicit constraints

No PyTorch/CUDA/TensorRT system component was replaced. No bitsandbytes source build was performed. No TensorRT engine was built. Phase 2.1 has NOT started. Exp05 Softmax has NOT started. The Phase 1 BF16 reference remains frozen.
