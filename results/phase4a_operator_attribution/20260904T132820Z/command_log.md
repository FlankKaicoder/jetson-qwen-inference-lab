# Phase 4-A.1 Command Log

Timestamp: `2026-09-04T13:28:20Z`

## Git Preflight

```powershell
git status --short --branch
git branch --show-current
git log -1 --oneline --decorate
```

Recorded state:

- Branch: `phase/04a-tensorrt-operator-attribution-recovery`
- HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Windows pre-existing untracked directory preserved.

## Input Verification

On Jetson:

```bash
python3.10 -c 'import onnx; print(onnx.__version__)' || true
/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools/bin/python -c 'import onnx; print(onnx.__version__)'
ls -l /tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.onnx
sha256sum /tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.onnx
```

Result:

- System Python lacked ONNX.
- Jetson tools venv ONNX version: `1.22.0`.
- ONNX size: `631,017,438` bytes.
- ONNX SHA-256: `8dccf65451c548762896280b498358069eaba100824b0acb92b7cdc9e4c51d9a`.

## Mapping Generation

A single remote Python process in `/home/nvidia/.venvs/jetson-qwen-phase2-trt-tools`:

1. Loaded the existing Phase 4-A.0 EngineInspector JSON.
2. Loaded the existing Mixed Decode ONNX graph with `load_external_data=False`.
3. Extracted `[ONNX Layer: ...]` references from TensorRT metadata.
4. Validated every referenced node name against the ONNX graph.
5. Derived a candidate operator only from recognized Qwen3 projection or normalization node names.
6. Wrote `layer_inventory.csv`, `onnx_node_inventory.csv`, `mapping.csv`, `gemm_candidates.csv`, and `analysis.json` to a new Jetson `/tmp` directory.

No build, execution, benchmark, profiling, engine rebuild, ONNX export, or repository cleanup was performed.

## Artifact Transfer

```powershell
scp -o BatchMode=yes \
  jetson:/tmp/phase4a_operator_attribution_20260904T132820Z/layer_inventory.csv \
  jetson:/tmp/phase4a_operator_attribution_20260904T132820Z/onnx_node_inventory.csv \
  jetson:/tmp/phase4a_operator_attribution_20260904T132820Z/mapping.csv \
  jetson:/tmp/phase4a_operator_attribution_20260904T132820Z/gemm_candidates.csv \
  jetson:/tmp/phase4a_operator_attribution_20260904T132820Z/analysis.json \
  results/phase4a_operator_attribution/20260904T132820Z/
```

## Validation

```powershell
git status --short --branch
git diff --stat
git diff --check
```

The directory previously did not exist. Only new files were added. Phase 3-E and Phase 4-A.0 artifacts were not modified.
