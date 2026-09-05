# Phase 4-F Command Log

Timestamp: `20260904T164404Z`

## Git preflight

```powershell
git status --short --branch
git branch --show-current
git log -1 --oneline --decorate
git remote -v
```

Observed:

- Branch: `phase/04a-tensorrt-operator-attribution-recovery`
- HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Tracked diff: none
- Deleted files: none
- Existing untracked artifact directories were preserved.

## Remote read-only artifact check

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 jetson "hostname; test -d /tmp/phase3c_nsys_20260904T093500Z && echo RAW_DIR_PRESENT || echo RAW_DIR_MISSING; find /tmp/phase3c_nsys_20260904T093500Z -maxdepth 3 -type f -printf '%s %p\n' | sort -nr | head -100"
```

Observed:

- Host: `nvidia-desktop`
- Raw NSYS directory: `RAW_DIR_PRESENT`
- `mixed_persistent.nsys-rep`: `782415` bytes
- `mixed_persistent.sqlite`: `3477504` bytes
- `fp16_persistent.nsys-rep`: `642649` bytes
- `fp16_persistent.sqlite`: `2981888` bytes

The raw artifacts were hashed but not copied or modified.

## Remote toolchain check

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 jetson "nsys --version; /home/nvidia/.venvs/jetson-qwen-phase2-quant/bin/python -c 'import tensorrt as trt; print(trt.__version__)'"
```

Observed:

- Nsight Systems: `2024.5.4.34-245434855735v0`
- TensorRT: `10.3.0`

## TensorRT API presence audit

An ephemeral Python script was streamed through SSH stdin. It did not
deserialize an engine, execute inference, profile, or benchmark.

Observed present:

- `ProfilingVerbosity`
- `IProfiler`
- `IExecutionContext.execute_async_v3`
- `IExecutionContext.enqueue_emits_profile`
- `IExecutionContext.nvtx_verbosity`
- `IExecutionContext.profiler`
- `IExecutionContext.report_to_profiler`
- `IExecutionContext.set_debug_listener`
- `IExecutionContext.get_debug_listener`
- `IExecutionContext.get_debug_state`
- `IExecutionContext.set_all_tensors_debug_state`
- `IExecutionContext.set_tensor_debug_state`
- `ICudaEngine.create_engine_inspector`

Presence was not treated as proof that any untested interface can provide
kernel-to-layer mapping.

## Read-only SQLite correlation query

The existing Mixed persistent SQLite trace was opened in read-only URI mode.
No file was exported, copied, rebuilt, or modified. The query joined:

```text
NVTX_EVENTS.textId -> StringIds.value
NVTX_EVENTS.globalTid + time containment -> CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_RUNTIME.correlationId -> CUPTI_ACTIVITY_KIND_KERNEL.correlationId
```

The `up_proj` filter used:

```sql
StringIds.value LIKE '/up_proj%/MatMul'
```

Observed:

- Unique `up_proj` NVTX ranges: `28`
- Unique `up_proj` TensorRT layers: `28`
- Correlated kernel launch rows: `196`
- Kernel launches per range: `7`
- Ranges with more than one kernel name: `28`
- Ranges with at least one h16816-family launch: `28`
- Ranges with at least one `sm80_xmma_gemm_*` launch: `28`
- h16816-family launch rows: `84`
- `sm80_xmma_gemm_*` launch rows: `112`

No raw trace or old artifact was modified.
