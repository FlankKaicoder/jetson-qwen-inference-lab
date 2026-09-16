# Phase 9.2-C2-R1 Non-Finite Output Diagnosis

Date: 2026-09-16 (Asia/Shanghai)

## Scope And Authorization

The owner authorized diagnosis of the Phase 9.2-C2 non-finite outputs using the
same fixed shape boundary: `pixel_values=[784,1536]` and
`grid_thw=[1,28,28]`. Allowed work was FP32 reference inference, FP16/FP32
finite-statistics comparison, temporary diagnostic hooks, and inspection of
intermediate tensor finite status.

No TensorRT rebuild or execution, benchmark, latency measurement, optimization,
quantization, model modification, CUDA change, or persistent environment change
occurred. The 266 forward-output hooks were registered at runtime and removed
after each pass.

## Gate

**PASS / BOUNDED — C2_INPUT_GENERATION_NONFINITE**

The observed non-finite outputs are supported as an input-generation mismatch,
not Vision Encoder instability. The exact C2 workload used direct CUDA FP16
`torch.linspace`, which produced `1,073,184` NaN values. When the same nominal
input was generated in FP32 and cast to FP16, both FP32 and FP16 model passes
completed with fully finite outputs. The reason inside the CUDA FP16
`linspace` implementation was not diagnosed.

## Model And Environment

| Field | Value |
| --- | --- |
| Model identity | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| `config.json` SHA-256 | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `model.safetensors` SHA-256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| Device | Jetson Orin, `cuda:0` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| Attention implementation | `eager` |
| Visual parameters | `406,957,056` |

## Diagnostic Attempts

Attempt 1 reconstructed the C2 input with direct CUDA FP16 `torch.linspace`
and then cast it to FP32. That input was already non-finite, so the FP32 pass
inherited NaNs and was not a valid finite reference. The first non-finite
module-output hook was `patch_embed.proj`, but its input was also non-finite.
This attempt is preserved and was rejected from the root-cause comparison.

Attempt 2 generated the FP32 reference directly with CUDA FP32
`torch.linspace`; generated the FP16 comparison input by casting that FP32
source to FP16; and separately audited the original direct CUDA FP16
`torch.linspace`. Both model passes were valid and fully finite.

## Input Generation Audit

| Input | Finite ratio | NaN | Min | Max | Mean | Std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 9.2-C2 direct CUDA FP16 `linspace` | `0.10881696428571429` | `1,073,184` | `NaN` | `NaN` | `0.0` finite-only | `1.0000038146972656` finite-only |
| Direct CUDA FP32 `linspace` | `1.0` | `0` | `-1.0` | `1.0` | `-8.109475313489156e-10` | `0.5773510336875916` |
| FP32 source cast to FP16 | `1.0` | `0` | `-1.0` | `1.0` | `0.0` | `0.5773507356643677` |

The input element count was `1,204,224`. The direct CUDA FP16 generator
contained no infinities; all non-finite elements were NaN.

## Module Hook Result

Attempt 2 registered 266 hooks and removed all of them. It recorded zero
non-finite module outputs in FP32 and zero in FP16. No submodule was found to
be the first non-finite producer.

In rejected Attempt 1, the first non-finite module output was
`patch_embed.proj` (`Conv3d`), call index `1`, output shape `[784,1024,1,1,1]`,
finite ratio `0.10714285714285714`, and `716,800` NaNs. Its
`input_finite=false` status means it propagated the non-finite workload rather
than demonstrating submodule instability.

## Output Finite Statistics

All four outputs were finite in both passes. Mean and standard deviation below
are over all elements; because all elements are finite, finite-only and overall
statistics coincide. Standard deviation is PyTorch's sample standard deviation.

| Output | Dtype | Finite ratio | Min | Max | Mean | Std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `final_hidden` | FP32 | `1.0` | `-4.500116348266602` | `11.94353199005127` | `0.008453300222754478` | `0.4709877371788025` |
| `deepstack_0` | FP32 | `1.0` | `-5.502180099487305` | `11.068126678466797` | `0.002678860677406192` | `0.36612316966056824` |
| `deepstack_1` | FP32 | `1.0` | `-8.969695091247559` | `8.524663925170898` | `-0.00210172520019114` | `0.5449541807174683` |
| `deepstack_2` | FP32 | `1.0` | `-8.741447448730469` | `21.621845245361328` | `0.009011543355882168` | `0.6258755326271057` |
| `final_hidden` | FP16 | `1.0` | `-4.5` | `12.1796875` | `0.008456148207187653` | `0.4712662100791931` |
| `deepstack_0` | FP16 | `1.0` | `-5.50390625` | `11.0703125` | `0.002674565650522709` | `0.36612066626548767` |
| `deepstack_1` | FP16 | `1.0` | `-8.96875` | `8.5234375` | `-0.0021177097223699093` | `0.5445386171340942` |
| `deepstack_2` | FP16 | `1.0` | `-8.765625` | `21.65625` | `0.008977613411843777` | `0.6263872981071472` |

## Root-Cause Classification

| Candidate cause | Classification | Evidence |
| --- | --- | --- |
| FP16 numerical instability | `NOT_REPRODUCED` | The finite FP16 comparison pass produced all four outputs with finite ratio `1.0` and statistics close to FP32. |
| Input/boundary mismatch | `SUPPORTED / C2_CUDA_FP16_LINSPACE_NONFINITE` | The exact direct CUDA FP16 input generator used by C2 was only `10.881696428571429%` finite. Shape and grid contracts were unchanged. |
| Specific Vision Encoder submodule instability | `NOT_OBSERVED` | With finite input, zero of 266 module-output hook records were non-finite in either dtype. Attempt 1's `patch_embed.proj` signal inherited non-finite input. |

## GPU Memory

| Pass | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: |
| FP32 | `1,754,426,880` | `1,822,425,088` |
| FP16 | `940,955,648` | `1,000,341,504` |

## Limitations And Non-Claims

- The conclusion applies to this deterministic nominal input, exact shape
  boundary, and the installed PyTorch build.
- The FP16 comparison used FP32-to-FP16 cast input. That is a proper finite
  reference comparison, but it is not bit-identical to the invalid direct
  CUDA FP16 `linspace` workload.
- The internal reason that direct CUDA FP16 `torch.linspace` produced NaNs was
  not diagnosed.
- This phase does not validate TensorRT correctness, because TensorRT was not
  executed and the C2 engine input was invalid.
- No realistic image, dynamic workload, optimization, quantization, or
  benchmark claim is made.

## Evidence

- Final result:
  `artifacts/phase9_2C2R1_20260916T071026Z/nonfinite_diagnosis_attempt2.json`
- Final console:
  `artifacts/phase9_2C2R1_20260916T071026Z/console_attempt2.log` (local-only
  due to the repository's `*.log` ignore rule)
- Rejected attempt:
  `artifacts/phase9_2C2R1_20260916T071026Z/failed_attempt1_inherited_nan_input/`
- Command log:
  `artifacts/phase9_2C2R1_20260916T071026Z/command_log.md`
- Artifact manifest:
  `artifacts/phase9_2C2R1_20260916T071026Z/artifact_manifest.json`
- Script:
  `src/phase9_2C2R1/run_vision_nonfinite_diagnosis.py`

The next action is to stop and await Gate review. No TensorRT rerun, engine
rebuild, input replacement, submodule repair, benchmark, optimization, or
quantization is authorized by this report.
