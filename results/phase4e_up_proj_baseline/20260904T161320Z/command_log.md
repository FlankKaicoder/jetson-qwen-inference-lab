# Phase 4-E Command Log

## Local preflight

```powershell
git status --short --branch
git branch --show-current
git log -1 --oneline
git remote -v
```

Observed:

- Branch: `phase/04a-tensorrt-operator-attribution-recovery`
- HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Tracked diff: none
- Deleted files: none
- Existing untracked artifact directories were preserved.

## Remote preflight

```powershell
ssh jetson "cd /home/nvidia/projects/jetson-qwen-inference-lab && git status --short --branch && git log -1 --oneline && test -s /tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine && sha256sum /tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine && stat -c %s /tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine"
```

Observed:

- Host: `nvidia-desktop`
- Remote branch: `phase/03e-tensorrt-kernel-attribution`
- Remote HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Engine SHA-256: `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c`
- Engine bytes: `650285868`

## API introspection

An ephemeral Python script was streamed through SSH stdin to
`/home/nvidia/.venvs/jetson-qwen-phase2-quant/bin/python -`. It was not written
to the remote repository or to remote persistent storage. It deserialized the
existing engine and inspected `num_io_tensors`, tensor modes, tensor shapes,
optimization-profile shapes, `IProfiler`, and profiler-related execution-context
methods.

Observed:

- TensorRT: `10.3.0`
- Engine I/O tensors: `142`
- `IProfiler.report_layer_time(layer_name, ms)`: present
- `IExecutionContext.report_to_profiler()`: present
- `IExecutionContext.enqueue_emits_profile`: present
- `IExecutionContext.execute_async_v3`: present

## Smoke probe

The first smoke probe used a non-default stream, deterministic tensor fill value
`0.125`, batch 1, cache length 8, position 8, `enqueue_emits_profile=false`,
and `report_to_profiler()` after each execution. It reported `699` layers per
iteration and matched all `28` expected `up_proj` layers.

## Measurement probe

The measurement probe used:

- 5 warmup iterations
- 30 measured iterations
- cache length 8
- position 8
- deterministic synthetic Half activation and past-cache tensors filled with
  `0.125`
- `torch.cuda.Stream`
- `enqueue_emits_profile=false`
- `IExecutionContext.report_to_profiler()` after each execution

The per-layer raw SSH stdout was analyzed. The per-layer summaries are retained
in `up_proj_timing.csv`; per-iteration aggregate values and interpretation limits
are retained in `timing_analysis.json`. Raw per-layer stdout was not separately
stored, in order to keep this phase to the six required artifacts.

No remote repository file was modified. No engine, ONNX, quantization policy, or
runtime architecture was changed.
