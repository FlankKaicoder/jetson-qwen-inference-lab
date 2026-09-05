# Phase 4-B Command Log

## Scope

- Branch: `phase/04a-tensorrt-operator-attribution-recovery`
- Starting HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Mode: read-only metadata analysis and target selection.
- No Jetson execution, engine rebuild, ONNX export, benchmark, Nsight profiling, CUDA kernel, plugin, quantization change, or runtime modification was performed.
- No historical artifact was deleted or overwritten.

## Commands

The following repository-state and evidence checks were run in PowerShell from `E:\nvidia-qwen`:

```powershell
git status --porcelain=v1
git branch --show-current
git log -1 --oneline
git remote -v

Get-Content -Raw README.md
Get-Content -Raw ROADMAP.md
Get-Content -Raw AGENTS.md
Get-Content -Raw docs/project_management.md
Get-Content -Raw docs/PROJECT_STATE.md
Get-Content -Raw docs/experiment_index.md

Get-Content -Raw results/phase4b_target_selection/20260904T140726Z/candidate_operator_summary.json
Get-Content -Raw results/phase3e_kernel_attribution/20260904T121007Z/analysis/analysis_summary.json
Get-Content -Raw results/phase3e_kernel_attribution/20260904T121007Z/analysis/e3_ncu_summary.json
Get-Content -Raw results/phase4a_operator_attribution/20260904T132820Z/phase4a1_layer_operator_mapping_report.md
Get-Content -Raw results/phase4a_operator_attribution/20260904T134556Z/phase4a2_runtime_evidence_correlation_report.md
Get-Content results/phase4a_operator_attribution/20260904T132820Z/gemm_candidates.csv -TotalCount 8
Get-Content results/phase4a_operator_attribution/20260904T134556Z/runtime_nvtx_kernel_mapping.csv -TotalCount 10
Get-ChildItem -Recurse results/phase4a_operator_attribution
Get-ChildItem -Recurse results/phase3e_kernel_attribution/20260904T121007Z
```

A read-only Python aggregation was run against
`runtime_nvtx_kernel_mapping.csv` to validate per-operator row counts,
confidence counts, mapped layer counts, kernel-name sets, and `total_time_ns`
sums. The bundled workspace Python interpreter was used because `python` was
not present on PATH. No source file, historical artifact, or input CSV was
modified.

## Aggregation result

| Operator | Rows | Unique range/kernel pairs | All-trace mapped time (ms) | Confidence |
| --- | ---: | ---: | ---: | --- |
| `UNKNOWN` / `unknown_attention_matmul` | 57 | 57 | 61.815776 | MEDIUM=57 |
| `up_proj` | 56 | 56 | 37.554112 | HIGH=56 |
| `v_proj` | 2 | 2 | 1.480960 | HIGH=2 |
| `k_proj` | 2 | 2 | 1.480960 | HIGH=2 |
| `q_proj` | 2 | 2 | 1.480960 | HIGH=2 |
| `gate_proj` | 1 | 1 | 0.648512 | HIGH=1 |

`down_proj` and `o_proj` had no rows in the Phase 4-A.2 runtime mapping.
The `q/k/v` total is one fused runtime range viewed from three operator
perspectives and is not additive.
