# Phase 9.3-A Command Log

## Repository Start State

- Windows branch: `phase/09-qwen3vl-migration`
- Windows starting HEAD:
  `0dd70f788c118e759ffbd0050111c20ceba4c856`
- Jetson branch: `phase/08-rmsnorm-optimization`
- Jetson HEAD: `6fb18773014a43f51c82b91338f2972131a4ce86`
- No Jetson repository fetch, pull, branch switch, reset, clean, or sync was
  performed.

## Protocol Freeze

The harness protocol was frozen before measurement with:

- one PyTorch versus TensorRT visual-output integration check
- one warmup generation per backend
- three measured generations per backend
- unchanged decoder path and fixed 16-token greedy decode
- CUDA-event timing for prefill first-token latency and total generation latency
- unchanged `25W` power mode and unchanged clocks

The protocol SHA-256 is
`4f1f83c68e0d6d1edcf389e586e13dd7243989396157a3dcfc5caf7f78dc94b8`.

## Preflight

Reverified the pinned model config hash, model weights hash, and unchanged
Phase 9.2-C1 engine hash. Confirmed ample `/tmp` space, CUDA available with
zero allocator usage, and power mode `25W`.

## Attempt 1

Remote `py_compile` passed. The first execution loaded the model and reached
the TensorRT integration boundary, then failed while serializing the final
result because raw trial tensors were retained in the result object. The failed
attempt is documented in
`failed_attempt1_serialization_failure.md`.

## Corrected Execution

1. Copied the corrected harness and verified remote `py_compile`.
2. Used the existing Phase 1 HF Python with the per-process
   `PYTHONPATH=/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/lib/python3.10/site-packages`
   TensorRT-tools bridge. Nothing was installed, removed, upgraded, or
   persisted.
3. Preprocessed the deterministic red-square image and fixed prompt.
4. Ran one PyTorch visual reference pass and one TensorRT visual pass, then
   compared all four output tensors.
5. Replaced `model.model.visual` with the runtime TensorRT adapter and released
   the original visual reference. The language model and generation rule were
   unchanged.
6. Ran one warmup and three measured generations for PyTorch FP16, followed by
   one warmup and three measured generations for TensorRT FP16.
7. Verified the engine hash again after the run.

## Restriction Record

- No optimization occurred.
- No TensorRT engine was rebuilt or modified.
- No quantization occurred.
- No decoder weights, code, generation rule, or CUDA setting was modified.
- No persistent environment modification occurred.
- No benchmark sweep or input sweep occurred.
