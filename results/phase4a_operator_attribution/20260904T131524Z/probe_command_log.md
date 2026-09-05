# Phase 4-A.0 Probe Command Log

Timestamp: `2026-09-04T13:15:24Z`

## Preflight

```powershell
git status --short --branch
git branch --show-current
git rev-parse HEAD
git log -1 --oneline --decorate
git switch -c phase/04a-tensorrt-operator-attribution-recovery
```

Recorded preflight state:

- Starting branch: `phase/03e-tensorrt-kernel-attribution`
- Starting HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- New branch: `phase/04a-tensorrt-operator-attribution-recovery`
- Windows pre-existing untracked directory was preserved.

## Jetson State And Engine Check

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline --decorate
date -u +%Y%m%dT%H%M%SZ
stat -c %s /tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine
sha256sum /tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine
```

Result:

- Jetson branch at probe start: `phase/03e-tensorrt-kernel-attribution`
- Jetson HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Engine SHA-256: `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c`

## Inspector Probe

A single remote Python process was run on Jetson with:

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(blob)
inspector = engine.create_engine_inspector()
info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
```

The process:

1. Verified the target directory did not exist.
2. Read and SHA-256 hashed the existing Mixed Decode engine.
3. Deserialized the existing engine without rebuilding it.
4. Exported `LayerInformationFormat.JSON`.
5. Wrote the raw JSON, CSV summary, analysis JSON, and environment JSON to a new Jetson `/tmp` directory.

No build, inference, benchmark, Nsight profiling, engine rebuild, ONNX export, or repository cleanup was executed.

## Artifact Transfer

```powershell
scp -o BatchMode=yes \
  jetson:/tmp/phase4a_operator_attribution_20260904T131524Z/mixed_decode_engine_inspector.json \
  jetson:/tmp/phase4a_operator_attribution_20260904T131524Z/mixed_decode_layer_summary.csv \
  jetson:/tmp/phase4a_operator_attribution_20260904T131524Z/phase4a0_analysis.json \
  jetson:/tmp/phase4a_operator_attribution_20260904T131524Z/environment.json \
  results/phase4a_operator_attribution/20260904T131524Z/
```

## Validation

```powershell
$rows = Import-Csv results/phase4a_operator_attribution/20260904T131524Z/mixed_decode_layer_summary.csv
$rows.Count
($rows | Where-Object tactic -ne 'UNKNOWN').Count
git status --short --branch
git diff --check
```

Result:

- CSV rows: `699`
- Layers with non-empty tactic values: `498`
- Unique tactic values: `68`
- Windows tracked diff: none
