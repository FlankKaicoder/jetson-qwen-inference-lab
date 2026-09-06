# Phase 6-H Validation

## Artifact Validation

- All Phase 6-H artifacts are compact text files intended to remain under
  1 MiB.
- `evidence_manifest.csv` records SHA-256 hashes for every external artifact
  used as evidence.
- `av_feasibility_evidence.csv` records `UNKNOWN`, `NOT_PROVEN`,
  `PARTIALLY_SUPPORTED`, or an explicitly bounded `YES` instead of inventing
  missing fields.
- The exact AV kernel-name search over
  `results/phase3e_kernel_attribution/20260904T121007Z/ncu` and
  `results/phase5b_tensorrt_gemm_path_investigation/20260905T112513Z/ncu`
  returned zero files.
- No NCU report contains one of the 112 Phase 6-G AV correlation IDs, so no
  NCU metric is treated as direct AV efficiency evidence.
- No raw `.ncu-rep`, `.qdrep`, engine, model weight, credential, or protected
  untracked directory is staged.

## Programmatic Validation

Python 3.14.0 is used only to parse CSVs, check expected row counts and column
consistency, calculate hashes/sizes, and verify the Phase 6-G duration total.
It does not create or modify evidence content.

Expected checks are recorded in `validation_results.json`.
