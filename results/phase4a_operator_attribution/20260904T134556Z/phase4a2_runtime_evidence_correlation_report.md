# Phase 4-A.2 Runtime Evidence Correlation Report

## Environment

- Device: Jetson Orin Nano Engineering Reference Developer Kit Super, host `nvidia-desktop`
- TensorRT: `10.3.0`
- CUDA: `12.6`
- Branch: `phase/04a-tensorrt-operator-attribution-recovery`
- Starting HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Mixed Decode engine: `/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine`, `650,285,868` bytes
- Engine SHA-256: `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c`

No engine rebuild, inference, benchmark, Nsight Systems profile, Nsight Compute
profile, ONNX export, plugin, or kernel development was performed in this phase.
The only Jetson-side probe was a read-only Python introspection of the installed
TensorRT 10.3 API surface.

## Existing evidence

The study used existing Phase 3-C NSYS summaries, Phase 4-A.0 EngineInspector
JSON, Phase 4-A.1 ONNX inventory, and Phase 4-A.1 mapping artifacts. Input
hashes are recorded in `runtime_capability_summary.json`.

Key summary evidence:

| Source | Observed value |
| --- | ---: |
| `mixed_persistent_nvtx_kern_sum.csv` rows | `1,115` |
| Unique kernel names in that summary | `144` |
| `mixed_persistent_cuda_api_sum.csv` rows / unique APIs | `28` / `28` |
| `myelin-exec:` ranges in the NVTX kernel summary | `117` |
| `myelin-exec:/...` source-path ranges | `86` |
| EngineInspector layers | `699` |
| EngineInspector layers with metadata | `498` |

The committed NSYS evidence is a derived aggregate, not the raw `.nsys-rep`
timeline. It is therefore suitable for correlation feasibility, not for a new
performance claim.

## Tested correlation methods

| Method | Can provide | Cannot provide | Status |
| --- | --- | --- | --- |
| Existing NSYS NVTX-to-kernel aggregate | NVTX execution-range to CUDA-kernel-name pairs | Full raw launch ordering; complete 699-layer coverage | `PASS / BOUNDED` |
| Exact `myelin-exec:/...` range-name matching | Runtime execution range to exact ONNX node path | Operator identity without the ONNX/Inspector chain | `PASS` for the 86 observed source-path ranges |
| EngineInspector metadata | Exact ONNX node references to TensorRT layer id/name | Runtime CUDA kernel name | `PASS / BOUNDED` |
| ONNX node inventory | Validation that the referenced node exists and its op type | Proof that a kernel alone belongs to a Transformer operator | `PASS` |
| TensorRT `IProfiler` | Python API surface is present | Untested kernel-to-layer mapping | `NOT TESTED` |
| TensorRT execution callbacks / debug state | API methods are present | Untested kernel-to-layer mapping | `NOT TESTED` |

The TensorRT 10.3 Python surface exposed `IProfiler`, `ProfilingVerbosity`,
`IExecutionContext.execute_async_v3`, `IExecutionContext.profiler`,
`report_to_profiler`, `set_debug_listener`, and related debug-state methods.
Their presence was recorded as capability evidence only; no untested API was
used to infer a mapping.

## Kernel -> Layer feasibility

A partial runtime mapping was established from the existing evidence chain:

```text
NSYS kernel name
  -> myelin-exec NVTX range
  -> exact ONNX node name
  -> exact EngineInspector metadata reference
  -> TensorRT layer id/name
```

All `86` observed `myelin-exec:/...` source-path ranges matched the ONNX graph.
Each also matched an EngineInspector metadata reference. The filtered mapping
table contains `120` rows across `6` unique CUDA kernel names.

Confidence of the mapped rows:

| Confidence | Mapping rows |
| --- | ---: |
| `HIGH` | `63` |
| `MEDIUM` | `57` |
| `LOW` | `0` |
| `UNKNOWN` | `0` |

Confidence of the `86` unique mapped ranges:

| Confidence | Ranges |
| --- | ---: |
| `HIGH` | `30` |
| `MEDIUM` | `56` |
| `LOW` | `0` |
| `UNKNOWN` | `0` |

The `30` `HIGH` ranges decompose as:

| Runtime range family | Confidence | Count |
| --- | --- | ---: |
| `up_proj` | `HIGH` | `28` |
| `gate_proj_27` | `HIGH` | `1` |
| fused `k_proj;q_proj;v_proj` at layer 55 | `HIGH` | `1` |

One concrete `HIGH` example is:

```text
myelin-exec:/v_proj_2/MatMul+/k_proj_2/MatMul+/q_proj_2/MatMul
  -> /v_proj_2/MatMul+/k_proj_2/MatMul+/q_proj_2/MatMul_myl0_55
  -> TensorRT layer id 55
  -> k_proj + q_proj + v_proj
```

The kernels observed in this `HIGH` subset include:

- `trt_ampere_h16816gemm_128x64_ldg8_nn_v1`
- `trt_ampere_h16816gemm_256x64_ldg8_tn_v1`
- four `sm80_xmma_gemm_*` GEMM tactic kernels recorded in
  `runtime_nvtx_kernel_mapping.csv`

## Layer -> Operator feasibility

The runtime evidence supports a bounded partial Layer -> Operator mapping:

- `28` `up_proj` ranges are `HIGH`.
- `1` `gate_proj` range is `HIGH`.
- `1` fused `k_proj/q_proj/v_proj` range is `HIGH`.
- `56` generic `/MatMul_*` ranges remain `MEDIUM`, because the ONNX node exists
  but the node name does not identify a Qwen3 operator.
- `down_proj` and `o_proj` are `UNKNOWN` in this runtime subset because the
  committed NVTX-to-kernel summary did not provide matching high-evidence
  execution ranges.

No operator identity was inferred from tactic names, kernel names, or layer
types alone.

## Evidence gaps

- The committed NSYS artifacts are summaries, not raw timelines.
- The `myelin-exec` ranges cover only a subset of the `699` EngineInspector
  layers.
- Internal `__myl_*` kernels without an exact ONNX node path remain `UNKNOWN`.
- Untested `IProfiler`, execution callback, and debug-state interfaces may
  provide stronger evidence, but no capability claim is made here.
- The result is correlation feasibility, not a proven optimization target.

## Gate

Phase 4-A.2: `PASS / BOUNDED`

The `PASS` component is justified because at least a partial
kernel -> TensorRT execution-layer mapping is supported by existing runtime
evidence. The `BOUNDED` component is required because the mapping is partial
and does not prove a complete or unique optimization target.

## Recommendation

Do not start CUDA kernel development or optimization from this report alone.
A separate, explicitly authorized and narrowly bounded target-selection study
may use the new partial mapping as one input, but it must revalidate coverage,
frequency, and operator attribution before selecting any CUDA target.
