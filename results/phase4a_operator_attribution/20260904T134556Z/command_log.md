# Phase 4-A.2 Command Log

Timestamp: `2026-09-04T13:45:56Z`

## Git preflight

```powershell
git status --short --branch
git branch --show-current
git log -1 --oneline --decorate
git remote -v
```

Recorded state:

- Branch: `phase/04a-tensorrt-operator-attribution-recovery`
- HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Pre-existing untracked directories were preserved.
- Phase 3-E, Phase 4-A.0, and Phase 4-A.1 artifacts were not modified.

## TensorRT API introspection

The following was run on Jetson through SSH without deserializing an engine or
executing inference:

```python
import tensorrt as trt
print(trt.__version__)
print(hasattr(trt, "IProfiler"))
print(hasattr(trt, "ProfilingVerbosity"))
print(hasattr(trt.IExecutionContext, "execute_async_v3"))
```

Result:

- TensorRT `10.3.0`
- `IProfiler`: present
- `ProfilingVerbosity`: present
- `IExecutionContext.execute_async_v3`: present
- Additional methods observed: `profiler`, `report_to_profiler`,
  `set_debug_listener`, `get_debug_listener`, `get_debug_state`,
  `set_all_tensors_debug_state`, and `set_tensor_debug_state`.

No runtime profiling, benchmark, or inference was run.

## Read-only correlation analysis

The local analysis joined:

1. `results/phase3c_residual_runtime/20260904T093500Z_nsys/stats/mixed_persistent_nvtx_kern_sum.csv`
2. `results/phase4a_operator_attribution/20260904T131524Z/mixed_decode_engine_inspector.json`
3. `results/phase4a_operator_attribution/20260904T132820Z/onnx_node_inventory.csv`
4. `results/phase4a_operator_attribution/20260904T132820Z/mapping.csv`

The output directory was created only after verifying that
`results/phase4a_operator_attribution/20260904T134556Z` did not already exist.

Generated artifacts:

- `runtime_nvtx_kernel_mapping.csv`
- `runtime_nvtx_range_summary.csv`
- `runtime_capability_summary.json`
- `phase4a2_runtime_evidence_correlation_report.md`

## Validation

```powershell
git status --short --branch
git diff --check
```

Expected final state:

- No tracked-file modification.
- No deletion.
- Only new Phase 4-A.2 artifacts in the timestamped directory.
- No commit.
