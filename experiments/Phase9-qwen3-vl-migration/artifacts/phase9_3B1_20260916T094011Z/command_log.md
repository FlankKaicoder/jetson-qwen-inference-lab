# Phase 9.3-B1 Command Log

## Repository Start State

- Windows branch: `phase/09-qwen3vl-migration`
- Windows starting HEAD:
  `2b790487795ca0d612677dad820a30f847d9377b`
- Jetson branch: `phase/08-rmsnorm-optimization`
- Jetson HEAD: `6fb18773014a43f51c82b91338f2972131a4ce86`
- No Jetson repository fetch, pull, branch switch, reset, clean, or sync was
  performed.
- Jetson power mode was checked and remained `25W`.

## Protocol Freeze

The protocol was frozen before measurement at
`2026-09-16T09:40:11Z`. It specifies the deterministic 448x448 red-square
image, prompt `"Describe the image."`, greedy decoding, 16 generated tokens, one
warmup and three measured generations per backend, backend order PyTorch FP16
then TensorRT FP16, and no inter-trial sleep.

The original frozen protocol SHA-256 is
`b85ee2e8fa29c25c424c573ddaefd85fe97eae0dbf0ffa145612463a4e2821db`. It was not
changed after the failed attempt.

## Local Preflight

- JSON parsing and Python AST parsing passed.
- Original harness SHA-256:
  `8e7cfd6f491b16708a310b1b7c17faf3927e8d48f8ac8be0fdf4a8106d664022`
- Corrected harness SHA-256:
  `aed4b2eb15771d30e8b0505b65452ebbb876596927fe5678b49898a20aa4cdb8`
- Remote protocol and harness hashes were verified after copying to
  `/tmp/phase9_3b1_20260916T094011Z/`.
- Pinned model config and weight hashes and unchanged C1 engine hash were
  verified inside the harness.

## Attempt 1

The first run loaded the model, entered the PyTorch backend, and completed all
PyTorch trials. It failed before any TensorRT backend trial because the runtime
TensorRT adapter omitted `spatial_merge_size`, which the preprocessing guard
read from `model.model.visual`. Exit code was `1`.

The failure was classified as a harness adapter-contract defect. Its raw
result, console log, and exit code are preserved under
`failed_attempt1_adapter_contract_failure/`. The failed run was not used for
measured conclusions.

## Corrected Execution

The adapter was changed only to retain the original visual model's
`spatial_merge_size`. No protocol, model, engine, decoder, timing, workload, or
environment setting changed. The corrected harness was copied to Jetson and its
SHA-256 was verified.

The remote command used the existing Phase 1 HF Python with the per-process
TensorRT-tools `PYTHONPATH` bridge:

```bash
PYTHONPATH=/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/lib/python3.10/site-packages /home/nvidia/.venvs/jetson-qwen-phase1-hf/bin/python src/run_end_to_end_stage_breakdown.py \
  --model-path /home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203 \
  --engine-path /tmp/phase9_2c1_20260916T034956Z/engine/qwen3_vl_vision_fp16.trt \
  --protocol config/protocol.json \
  --result-path results/phase9_3B1_result.json
```

The corrected run exited `0`. Result, console, and exit code were copied back
into the artifact directory.

## Restriction Record

- No optimization occurred.
- No TensorRT engine was rebuilt or modified.
- No quantization occurred.
- No decoder weights, decoder code, generation rule, or CUDA setting was
  modified.
- No Python package was installed, removed, upgraded, or persisted.
- No benchmark sweep or input sweep occurred.
- No power measurement was requested or collected; power is `UNKNOWN`.
