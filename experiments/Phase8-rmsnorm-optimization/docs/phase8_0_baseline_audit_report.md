# Phase 8.0 RMSNorm Baseline Audit Report

## Executive Summary

Phase 8.0 is an explicitly bounded environment audit and PyTorch RMSNorm
baseline-preparation step authorized by the owner after Phase 7. It did not
implement CUDA kernels, modify ONNX, rebuild or deserialize a TensorRT engine
for execution, change precision, force tactics, or run Nsight Systems or
Nsight Compute.

Final gate:

```text
PASS / BOUNDED
PHASE8_1_BASELINE_PREPARED
```

This gate means the Jetson environment, frozen Qwen3 model, existing engine
inventory, and PyTorch RMSNorm baseline evidence are sufficient to prepare for
Phase 8.1. It does **not** claim that Qwen3 RMSNorm is inefficient or that a
custom CUDA RMSNorm kernel will improve the real model.

## Scope And Authorization

- Authorized: environment audit, model/engine inventory, and a PyTorch
  RMSNorm correctness/latency/memory baseline.
- Not performed: CUDA kernel implementation, TensorRT plugin, engine build or
  rebuild, engine execution, model export, ONNX modification, precision
  change, tactic forcing, runtime redesign, profiling, or benchmarking of the
  full model.
- The owner's Phase 8 direction supersedes Phase 7's stop condition for this
  narrow step only. Phase 7's negative optimization-target conclusion remains
  valid and is not reinterpreted by Phase 8.0.

## Starting Git State

| Field | Value |
| --- | --- |
| Windows working directory | `E:\nvidia-qwen` |
| Branch at Phase 8.0 start | `phase/06h-attention-v-feasibility-boundary` |
| Starting HEAD | `ffe88c6e34d1b164522db5a85f2f3e16ecab153c` |
| Starting phase branch | Created as `phase/08-rmsnorm-optimization` from the starting HEAD |
| Windows preserved untracked evidence | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T203103Z/`, `results/phase6a_unknown_attention_matmul_attribution/{20260906T040200Z,20260906T040300Z,20260906T040400Z}/` |
| Jetson branch | `phase/03e-tensorrt-kernel-attribution` |
| Jetson HEAD | `bf7abc67eb58662a68316045e166aa9f611330d7` |
| Jetson working tree | Eight untracked historical paths; preserved unchanged |

The Jetson checkout was not switched, reset, cleaned, or synchronized. The
Phase 8.0 script was copied to `/tmp` and its compact artifacts were copied
back to the Windows repository. This keeps the Jetson checkout unchanged.

## Jetson Environment

Read-only audit on `nvidia-desktop`:

| Field | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | `5.15.148-tegra` |
| L4T | `R36.4.3` |
| Device | Orin |
| Compute capability | `8.7` |
| CUDA | `12.6.68` via `/usr/local/cuda-12.6/bin/nvcc` |
| TensorRT | `10.3.0` / TensorRT v100300 |
| Nsight Compute | `2024.3.1.0` |
| Nsight Systems | `2024.5.4.34` |
| PyTorch venv | `/home/nvidia/.venvs/jetson-qwen-phase1-hf` |
| Python | `3.10.12` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| `torch.cuda` | Available, device `Orin`, capability `(8, 7)` |

`jetson_clocks --show` required root and was not run with elevated permission.
No power mode, clocks, or system package was changed.

## Qwen3 Model State

| Field | Value |
| --- | --- |
| Model directory | `/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca` |
| Revision identity | `c1899de289a04d12100db370d81485cdf75e47ca` |
| `model.safetensors` SHA256 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| `config.json` SHA256 | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| Checkpoint dtype | BF16 |
| Layers | 28 |
| Hidden size | 1024 |
| RMSNorm epsilon | `1e-6` |
| RMSNorm weight tensors | 113 |
| RMSNorm breakdown | 56 input/post-attention `[1024]`, 56 Q/K norm `[128]`, 1 final `[1024]` |

The checkpoint hash exactly matches the frozen Phase 1 identity. Phase 8.0
loaded only `model.norm.weight`, not the full model.

## Existing TensorRT Engine Inventory

All four engines were inspected by file existence, size, and SHA-256 only.
They were not deserialized or executed.

| Engine | Size | SHA256 |
| --- | ---: | --- |
| `/tmp/phase2_2b4_2_20260902T082326Z/prefill_28layer.engine` | 852 MiB | `c31f161396c2dc211c418640bce241d3b612e88be0c2da285ec088639ffe960b` |
| `/tmp/phase2_2b4_2_20260902T082326Z/decode_28layer.engine` | 850 MiB | `16377216614750dfb126e384f9cc26613c558a528e6dd83ee112c25e1f91e60b` |
| `/tmp/phase2_3e_20260904T020000Z/work/mixed_prefill_28layer.engine` | 622 MiB | `3258b0a82fd4884a291e2bf0e3dded179434a9d1a5d034c4fc4e9f772c8964ec` |
| `/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine` | 621 MiB | `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c` |

The Mixed Decode hash matches the frozen Phase 3+ evidence identity.

## PyTorch RMSNorm Baseline

The formal run is
`artifacts/phase8_0_20260907T091524Z/`. The earlier
`artifacts/phase8_0_20260907T091305Z/` is retained as a pilot run and only
documents the per-call measurement limitation. The formal implementation is
`torch.nn.functional.rms_norm`; the correctness oracle is explicit FP32
reduction and multiplication. Inputs are deterministic seeded `randn`, the
weight is Qwen3 `model.norm.weight`, and the reported shapes are Qwen3 prefill
and decode hidden states.

Protocol:

| Field | Value |
| --- | --- |
| Seed | `20260907` |
| Shapes | `[1, 8, 1024]` and `[1, 1, 1024]` |
| Dtypes | BF16 and FP16 |
| Warmup | 50 calls |
| Repetitions per trial | 200 calls |
| Trials | 5 |
| Per-call timing | CUDA event pair around each call, host synchronize after each call |
| Amortized timing | 200 calls between one CUDA event pair |
| Host timing | Python perf-counter around 200 asynchronous submissions, then drain |
| Correctness | Against explicit FP32 reduction |

### Correctness

All four cases produced finite tensors.

| Case | Dtype | Relative-L2 | Cosine | Max abs |
| --- | --- | ---: | ---: | ---: |
| Prefill `[1,8,1024]` | BF16 | `0.003625764511525631` | `0.9999943971633911` | `0.14460182189941406` |
| Prefill `[1,8,1024]` | FP16 | `0.0003673394676297903` | `0.9999995231628418` | `0.011663436889648438` |
| Decode `[1,1,1024]` | BF16 | `0.0032095012720674276` | `0.999997615814209` | `0.05933380126953125` |
| Decode `[1,1,1024]` | FP16 | `0.00035894280881620944` | `1.0000001192092896` | `0.010223388671875` |

No accuracy threshold was required for Phase 8.0. These values establish that
the PyTorch baseline ran correctly under the recorded reference; they do not
authorize a precision change.

### Latency

The formal run's aggregate means are:

| Case | Dtype | Per-call event mean | Amortized event mean | Host submit mean |
| --- | --- | ---: | ---: | ---: |
| Prefill | BF16 | `0.265140096 ms` | `0.198005791 ms` | `0.196713063 ms` |
| Prefill | FP16 | `0.273001024 ms` | `0.195746849 ms` | `0.196652384 ms` |
| Decode | BF16 | `0.271276224 ms` | `0.195675137 ms` | `0.196138818 ms` |
| Decode | FP16 | `0.271044576 ms` | `0.192396866 ms` | `0.193529210 ms` |

Per-call CUDA Event means have CV `0.041437897-0.186772460`. Amortized CUDA
Event means have CV `0.002233256-0.008902449`. The amortized event mean and
host submit mean agree within about `0.002 ms`, so the stable PyTorch baseline
is dominated by host-side operator/launch overhead rather than an isolated
GPU kernel time. Therefore:

```text
PYTORCH_RMSNORM_KERNEL_ONLY_LATENCY: UNKNOWN
STABLE_PYTORCH_CALL_BASELINE: RECORDED
```

This is not a performance optimization conclusion.

### Memory

The formal environment had `5,958,610,944` CUDA free bytes out of
`7,989,940,224` total bytes at process observation time. The baseline tensor
footprints are:

| Case | Input + output | Weight |
| --- | ---: | ---: |
| Prefill | `2 * 16,384` bytes | `2,048` bytes |
| Decode | `2 * 2,048` bytes | `2,048` bytes |

The raw JSON records stable `torch_allocated_bytes` of `67,584` for prefill and
`10,240` for decode, with `torch_reserved_bytes` of `2,097,152`, after the
trial loops. Peak allocator counters were not separately captured in this
run and are not inferred.

## Reproducibility And Evidence

- Formal runner script: `scripts/run_phase8_0_baseline.sh`
- Baseline implementation: `src/rmsnorm_baseline.py`
- Formal artifacts:
  `artifacts/phase8_0_20260907T091524Z/`
- Pilot artifacts:
  `artifacts/phase8_0_20260907T091305Z/`
- Formal script SHA256:
  `fa78b5f740f2af6be1608a141b00a77c0728c4a7cd280b9903d2edd98a7474b6`
- Formal summary SHA256:
  `d06a9d8eaacc4c383a73b86b0b23735ddeae86df86f214567c92ac3207f7b90a`

The first pilot used per-call timing only and was retained without overwriting
the formal run. Its higher observed variance is documented, not hidden.

## Gate Decision

| Gate | Result | Evidence |
| --- | --- | --- |
| A: Git and working-tree audit | `PASS` | Starting Git state and preserved untracked evidence above |
| B: Jetson environment | `PASS` | CUDA 12.6, TensorRT 10.3, NCU 2024.3.1, NSYS 2024.5.4, SM87 |
| C: Qwen3 model and RMSNorm inventory | `PASS` | Frozen checkpoint hash, 28 layers, 113 RMSNorm tensors |
| D: PyTorch RMSNorm baseline | `PASS / BOUNDED` | Finite correctness, recorded latency, stable timing variance, memory observed |
| Overall | `PASS / BOUNDED` | Phase 8.1 baseline preparation is complete |

The principal boundary is that PyTorch host launch overhead dominates the
stable call baseline. Phase 8.1 must compare CUDA V0/V1/V2 using the same
per-call and amortized timing methods and must not relabel either number as
kernel-only latency.

## Exact Next Action

Stop here and await explicit owner authorization for Phase 8.1. If authorized,
the next action is to implement `cuda-kernel/rmsnorm/` V0, V1 and V2 with
correctness gates and the matching benchmark protocol above. No NCU profiling
should start in Phase 8.1; Phase 8.2 remains a separate boundary.
