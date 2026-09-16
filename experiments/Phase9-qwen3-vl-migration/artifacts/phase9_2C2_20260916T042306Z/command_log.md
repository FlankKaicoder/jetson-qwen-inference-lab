# Phase 9.2-C2 Command Log

## Repository Start State

- Windows branch: `phase/09-qwen3vl-migration`
- Windows starting HEAD: `b3fbe54299bd003c606d7680acc8f5bfc3a3b3a4`
- Jetson branch: `phase/08-rmsnorm-optimization`
- Jetson HEAD: `6fb18773014a43f51c82b91338f2972131a4ce86` (one commit behind its Phase 8 origin tip)
- No Jetson repository fetch, pull, branch switch, reset, clean, or sync was performed.

## Preflight

1. Verified the tracked Windows tree was clean before the C2 script and report
   changes; pre-existing untracked files were left untouched.
2. Read `PROJECT_STATE`, the experiment index, Phase 9.2-C1 evidence, and the
   C2 script.
3. Verified pinned checkpoint config and weight SHA-256 values and the C1
   engine SHA-256 on Jetson.
4. Verified `/tmp` had 171 GiB available and CUDA was available with zero bytes
   allocated/reserved before the C2 process.
5. Created isolated `/tmp/phase9_2c2_20260916T042306Z/src` and `results`
   directories.
6. Copied the script to Jetson and passed remote `py_compile`.

## Run Attempts

### Attempt 1: `BLOCKED_PRE_MODEL`

- The first execution stopped at
  `torch.cuda.reset_peak_memory_stats(torch.device('cuda:0'))` because this
  NVIDIA Torch build rejected the device object argument.
- No model forward or TensorRT execution occurred. The console log and script
  are preserved under `failed_attempt1_invalid_device/`.
- Narrowly patched the script to set the current CUDA device and call memory
  APIs without an explicit device argument. No environment package or CUDA
  state was changed.

### Attempt 2: `EXECUTED_NON_FINITE`

- Used the existing Phase 1 HF Python with a per-process
  `PYTHONPATH=/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/lib/python3.10/site-packages`
  bridge. Nothing was installed or persisted.
- Loaded the visual weights strictly and executed the PyTorch FP16 model once.
- Deserialized the C1 engine once, created one execution context, set the fixed
  shape and four output buffers, and executed the engine once.
- Both PyTorch and TensorRT outputs were non-finite. The script wrote its JSON
  and exited nonzero, correctly refusing to report a successful correctness
  comparison.

## Restriction Record

- No benchmark was run.
- No latency or throughput was measured.
- No optimization, tactic override, or backend modification occurred.
- No quantization occurred.
- No CUDA kernel or persistent CUDA/environment setting was modified.
- No model conversion or TensorRT build occurred.
