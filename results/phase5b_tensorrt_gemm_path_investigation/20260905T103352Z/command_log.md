# Phase 5-B Step 1 Command Log

Repository-side analysis was run from `E:\nvidia-qwen` with Python 3:

```powershell
py experiments/Phase5-cuda-feasibility/scripts/phase5b_tactic_attribution.py `
  --inspector results/phase4a_operator_attribution/20260904T131524Z/mixed_decode_engine_inspector.json `
  --mapping results/phase4a_operator_attribution/20260904T132820Z/mapping.csv `
  --runtime-kernel-csv results/phase5a_cuda_feasibility_baseline/20260905T072100Z/kernel_breakdown.csv `
  --csv-out results/phase5b_tensorrt_gemm_path_investigation/20260905T103352Z/tactic_attribution.csv `
  --json-out results/phase5b_tensorrt_gemm_path_investigation/20260905T103352Z/tactic_attribution_analysis.json
```

No remote execution, engine inspection, engine build, benchmark, or profiler
command was run.
