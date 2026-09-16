# Phase 9.2-C2-R2 Command Log

## Repository Start State

- Windows branch: `phase/09-qwen3vl-migration`
- Windows starting HEAD: `d1c5341c3ad0280b23e9f406292874972945d61d`
- Jetson branch: `phase/08-rmsnorm-optimization`
- Jetson HEAD: `6fb18773014a43f51c82b91338f2972131a4ce86`
- No Jetson repository fetch, pull, branch switch, reset, clean, or sync was
  performed.

## Preflight

1. Verified the Phase 9.2-C2-R1 conclusion and corrected finite input path.
2. Reverified the pinned model config hash, model weight hash, and unchanged
   Phase 9.2-C1 engine hash on Jetson.
3. Verified `/tmp` had 171 GiB available and CUDA was available with zero bytes
   allocated/reserved before the process.
4. Created isolated `/tmp/phase9_2c2r2_20260916T080511Z/src` and `results`
   directories.
5. Copied the C2-R2 script to Jetson and passed remote `py_compile`.

## Execution

- Used the existing Phase 1 HF Python with a per-process
  `PYTHONPATH=/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/lib/python3.10/site-packages`
  bridge. Nothing was installed, removed, upgraded, or persisted.
- Generated the nominal input with direct CUDA FP32 `torch.linspace`, then cast
  it to FP16.
- Loaded the visual weights strictly and executed the PyTorch FP16 model once.
- Deserialized the unchanged C1 TensorRT engine once, created one execution
  context, set the fixed shape and four output buffers, and executed the engine
  once.
- Compared all four outputs in FP32 for max absolute error, mean absolute
  error, and cosine similarity.

## Restriction Record

- No benchmark was run.
- No latency or throughput was measured.
- No TensorRT engine rebuild, optimization, tactic override, or backend
  modification occurred.
- No quantization occurred.
- No CUDA kernel or persistent CUDA/environment setting was modified.
