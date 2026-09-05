# Phase 4-C Command Log

All commands ran from `E:\nvidia-qwen` on the Windows analysis host. No Jetson
side command was run in this phase. No engine, ONNX, benchmark, profiler, or
runtime artifact was modified or rerun.

## Repository preflight

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
git remote -v
```

Observed:

- Branch: `phase/04a-tensorrt-operator-attribution-recovery`
- HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Last commit: `phase3e: close compute efficiency audit`
- Tracked diff: none
- Deleted files: none
- Existing untracked artifact directories were preserved.

## Read-only evidence inspection

Read and cross-checked:

- `results/phase3e_kernel_attribution/20260904T121007Z/analysis/analysis_summary.json`
- `results/phase3e_kernel_attribution/20260904T121007Z/analysis/e3_ncu_summary.json`
- `results/phase4a_operator_attribution/20260904T132820Z/phase4a1_layer_operator_mapping_report.md`
- `results/phase4a_operator_attribution/20260904T132820Z/layer_inventory.csv`
- `results/phase4a_operator_attribution/20260904T132820Z/mapping.csv`
- `results/phase4a_operator_attribution/20260904T134556Z/phase4a2_runtime_evidence_correlation_report.md`
- `results/phase4a_operator_attribution/20260904T134556Z/runtime_capability_summary.json`
- `results/phase4b_target_selection/20260904T140726Z/phase4b_target_selection_report.md`
- `results/phase3c_residual_runtime/20260904T090803Z_bench/benchmark_result.json`
- `experiments/Exp04-gemm/docs/experiment_design.md`
- `experiments/Exp04-gemm/docs/correctness_analysis.md`
- `experiments/Exp04-gemm/docs/benchmark_analysis.md`

Read-only PowerShell queries included filtering `mapping.csv` for `up_proj`,
inspecting representative EngineInspector rows with layer IDs 22, 223, 372 and
696, and reading the Exp04 correctness/benchmark definitions.

## Artifact creation

Created a new directory only after checking that it did not already exist:

```powershell
if (Test-Path -LiteralPath results/phase4c_up_proj_baseline/20260904T152617Z) { throw 'artifact directory already exists' }
New-Item -ItemType Directory -Path results/phase4c_up_proj_baseline/20260904T152617Z -Force
```

Added Phase 4-C files with `apply_patch`. No existing file was overwritten and
no file was deleted.

## Validation

Parsed the four JSON artifacts with `ConvertFrom-Json`; all passed. Parsed
`shape_audit.csv`; it contains exactly 28 rows. The mechanical CSV-format
validation found that array-like shape strings containing commas needed a
delimiter-safe representation, so those newly created shape fields were
rewritten as `1x1x1024`, `1024x3072`, `1x1024x3072` and `1x1x3072`. This did
not alter any historical artifact or shape semantics.

Final validation observed:

- Shape rows: 28.
- Uniform expected-shape rows: 28.
- Tactic count `f16f16_f16`: 25.
- Tactic count `f16f32_f32`: 3.
- Tracked diff: none.
- Deleted tracked files: none.
