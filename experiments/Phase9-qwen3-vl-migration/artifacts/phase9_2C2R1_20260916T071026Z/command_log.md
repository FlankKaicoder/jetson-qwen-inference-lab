# Phase 9.2-C2-R1 Command Log

## Repository Start State

- Windows branch: `phase/09-qwen3vl-migration`
- Windows starting HEAD: `17365de97b89e1285d161eac38e3080ab3c4d2c0`
- Jetson branch: `phase/08-rmsnorm-optimization`
- Jetson HEAD: `6fb18773014a43f51c82b91338f2972131a4ce86`
- No Jetson repository fetch, pull, branch switch, reset, clean, or sync was
  performed.

## Run Attempts

### Attempt 1: `INVALID_DIAGNOSTIC_INPUT`

- The first diagnostic reconstructed the Phase 9.2-C2 workload with direct
  CUDA FP16 `torch.linspace`, then cast it to FP32 for the FP32 run.
- Input inspection showed `1,073,184` NaN elements out of `1,204,224`, so the
  FP32 pass inherited non-finite input and was not a valid finite reference.
- The first non-finite module output was `patch_embed.proj`, but its hook
  reported `input_finite=false`, so it did not prove submodule instability.
- The complete JSON, console, and script are preserved under
  `failed_attempt1_inherited_nan_input/`.

### Attempt 2: `VALID_FINITE_REFERENCE_AND_FP16_COMPARISON`

- Used direct CUDA FP32 `torch.linspace` as the FP32 reference.
- Used the same FP32 source cast to FP16 for the FP16 comparison.
- Separately audited direct CUDA FP16 `torch.linspace` and reproduced the
  non-finite C2 input.
- Registered and then removed 266 forward-output hooks on the visual model.
- No TensorRT import, TensorRT execution, benchmark, latency measurement,
  optimization, quantization, model modification, CUDA change, or persistent
  environment change occurred.
