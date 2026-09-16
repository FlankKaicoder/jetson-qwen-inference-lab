# Phase 9.2-D1 Command Log

## Repository Start State

- Windows branch: `phase/09-qwen3vl-migration`
- Windows starting HEAD:
  `d0367668dc31f64d16b51829bd26d4922b70c19a`
- Jetson branch: `phase/08-rmsnorm-optimization`
- Jetson HEAD: `6fb18773014a43f51c82b91338f2972131a4ce86`
- No Jetson repository fetch, pull, branch switch, reset, clean, or sync was
  performed.

## Protocol Freeze

The benchmark protocol was created before measurement with:

- 3 warmups per backend
- 30 measured trials per backend
- PyTorch FP16 followed by unchanged TensorRT FP16
- CUDA-event timing around model forward or `execute_async_v3`
- 100 ms `tegrastats` sampling for backend setup+warmup+measurement
- unchanged `25W` power mode and unchanged clocks

The protocol SHA-256 is
`e6a450e9dd8f538b0627aa19bb488300474e843869b55cb1ec3c3f62fb6cc545`.

## Preflight

Reverified the pinned model config hash, model weights hash, and unchanged
Phase 9.2-C1 engine hash on Jetson. Confirmed `/tmp` had ample free space, CUDA
was available with zero initial allocator usage, `/usr/bin/tegrastats` existed,
and power mode was `25W`.

## Execution

1. Created isolated `/tmp/phase9_2d1_20260916T083418Z/src` and `results`
   directories.
2. Copied only the frozen protocol and D1 benchmark script to Jetson.
3. Passed remote `py_compile` before model or engine loading.
4. Used the existing Phase 1 HF Python with the per-process
   `PYTHONPATH=/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/lib/python3.10/site-packages`
   TensorRT-tools bridge. Nothing was installed, removed, upgraded, or
   persisted.
5. Generated the corrected finite workload with direct CUDA FP32 `linspace`
   followed by FP32-to-FP16 cast.
6. Ran 3 PyTorch FP16 warmups and 30 measured CUDA-event trials.
7. Deserialized the unchanged C1 TensorRT engine, created one execution
   context, ran 3 warmups and 30 measured CUDA-event trials.
8. Recorded last-output finite status, CUDA allocator snapshots, and
   `tegrastats` board/rail samples for each backend.

## Restriction Record

- No optimization occurred.
- No TensorRT engine was rebuilt or modified.
- No quantization occurred.
- No model weights or configuration were modified.
- No CUDA kernel, clock, power mode, or CUDA environment setting was modified.
- No persistent environment modification occurred.
