# Phase 9.3-B2 Command Log

## Repository Start State

- Windows branch: `phase/09-qwen3vl-migration`
- Windows starting HEAD:
  `e2d109c25202c58b93255b23afffaa0b3892a8b9`
- Jetson repository remained intentionally frozen on
  `phase/08-rmsnorm-optimization` at
  `6fb18773014a43f51c82b91338f2972131a4ce86`.
- No Jetson repository fetch, pull, branch switch, reset, clean, or sync was
  performed.
- Jetson power mode was checked and remained `25W`.

## Protocol Freeze

The protocol was frozen before measurement at `2026-09-16T10:28:08Z`. Its
SHA-256 is
`69d72349a65f62c5eaa760a8e89f396b9e12dd738859fdaaf77fd27a10091c12`.

The protocol reuses the Phase 9.3-B1 deterministic image, prompt, 16-token
greedy rule, and unchanged TensorRT visual backend. It defines a clean latency
run and a separate diagnostic Nsight profile run.

## Preflight And Profiler Feasibility

- Local Python compile checks passed for the B2 harness and parser.
- Local JSON parsing and SHA-256 checks passed for the frozen protocol.
- The local harness SHA-256 before the final profile fix was
  `d7a9432a79cf5a56853147526ad0e60b662296a8f85504fe18459b480243ebbd`; after the
  visual NVTX synchronization fix it was
  `a23ed62c6ccca4215231d7e7766bd9f2eb3ea501509c0921b0cfd19563c08aad`.
- The local parser SHA-256 after the final visual-exclusion fix was
  `a5ba215aa0f1ecd67d801301d7f6057e673b597fcc4a301b9eceaa4dbdc49f54`.
- Remote protocol and script hashes were verified after copying to
  `/tmp/phase9_3b2_20260916T102808Z/`.
- A minimal CUDA/NVTX smoke test confirmed Nsight Systems 2024.5.4 could produce
  a `.nsys-rep` and SQLite export with CUDA kernels and NVTX ranges under
  temporary sudo.
- `sudo -n` worked. The temporary sudo use was only for CUPTI capture and did
  not change environment configuration.

## Attempt 1

The clean latency run failed during warmup because NVIDIA PyTorch did not expose
`torch.cuda.nvtx.is_active`. Model loading and TensorRT engine deserialization
succeeded, but no trial completed. Exit code was `1`. The failed result, console,
and exit code are preserved under
`failed_attempt1_nvtx_api_mismatch/`.

The only correction was to gate the adapter NVTX range on the frozen run-mode
flag rather than the unavailable API. The protocol was unchanged.

## Clean Latency Run

The corrected run exited `0` and used:

```bash
PYTHONPATH=/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/lib/python3.10/site-packages \
  /home/nvidia/.venvs/jetson-qwen-phase1-hf/bin/python \
  src/run_decoder_bottleneck_profile.py \
  --model-path /home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203 \
  --engine-path /tmp/phase9_2c1_20260916T034956Z/engine/qwen3_vl_vision_fp16.trt \
  --protocol config/protocol.json \
  --result-path results/phase9_3B2_latency_result.json \
  --run-mode latency
```

The run completed one warmup and three measured generations.

## Attempt 2

The first Nsight profile run exited `0` and completed one measured generation,
but its TensorRT visual NVTX range ended before asynchronous engine kernels
completed. It captured only 4.88 ms of visual kernels while the CUDA-event visual
boundary was 136.83 ms. Its result and partial attribution are preserved under
`attempt2_visual_nvtx_range_async/`.

The only correction was to synchronize inside the profile-only visual NVTX range
after `execute_async_v3`. The clean latency run, model, engine, decoder, and
protocol were not changed.

## Final Profile Run

The final profile run exited `0` and used:

```bash
sudo env CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/lib/python3.10/site-packages \
  /usr/local/bin/nsys profile \
  -o profile/phase9_3B2_profile \
  --force-overwrite true \
  --trace=cuda,nvtx \
  /home/nvidia/.venvs/jetson-qwen-phase1-hf/bin/python \
  src/run_decoder_bottleneck_profile.py \
  --model-path /home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203 \
  --engine-path /tmp/phase9_2c1_20260916T034956Z/engine/qwen3_vl_vision_fp16.trt \
  --protocol config/protocol.json \
  --result-path results/phase9_3B2_profile_result.json \
  --run-mode profile \
  --intended-nsys-report-path /tmp/phase9_3b2_20260916T102808Z/profile/phase9_3B2_profile.nsys-rep
```

Nsight SQLite export and the parser both exited `0`. The parser validated one
profiled generation, one prefill range, one visual range, one decode range, and
15 decode-step ranges.

## Restriction Record

- No TensorRT-LLM migration occurred.
- No optimization occurred.
- No quantization occurred.
- No decoder weights, decoder code, generation rule, or CUDA setting was
  modified.
- No TensorRT engine was rebuilt or modified.
- No Python package was installed, removed, upgraded, or persisted.
- No benchmark sweep or input sweep occurred.
- Power was not requested or measured; power is `UNKNOWN`.
