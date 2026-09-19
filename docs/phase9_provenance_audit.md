# Phase9 Provenance Audit

Audit date: 2026-09-19 (Asia/Shanghai)

Mode: read-only. No checkout, pull, fetch of Git objects, merge, reset, clean, code modification, or artifact modification was performed.

## Conclusion

**A: Phase9 confirmed executed on Jetson.**

The current Jetson repository checkout does not contain Phase9 files because it remains intentionally frozen on `phase/08-rmsnorm-optimization`. That does not imply Phase9 was not executed. Windows-tracked Phase9 artifacts provide direct runtime evidence that the Phase9 experiments ran on the Jetson host, in isolated `/tmp/phase9_*` run directories, against Jetson-local model and engine paths.

The strongest runtime evidence is:

- hostname `nvidia-desktop`
- platform `Linux-5.15.148-tegra-aarch64-with-glibc2.35`
- CUDA device `Orin`
- CUDA capability `[8, 7]`
- NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`
- CUDA `12.6`
- TensorRT `10.3.0`
- Jetson repository path `/home/nvidia/projects/jetson-qwen-inference-lab`
- Jetson frozen branch `phase/08-rmsnorm-optimization` and HEAD `6fb18773014a43f51c82b91338f2972131a4ce86`
- Jetson-local model path `/home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203`
- Jetson-local engine path `/tmp/phase9_2c1_20260916T034956Z/engine/qwen3_vl_vision_fp16.trt`

The final Nsight Systems report and SQLite export were recorded as Jetson-local with SHA-256 values in the Phase9.3-B2 artifact manifest. They are no longer present under `/tmp` in the current audit, but their provenance metadata is committed.

## GitHub phase/09 Commit

The remote branch was checked by read-only HTTPS because the earlier `git ls-remote origin` SSH query failed:

```text
refs/heads/phase/09-qwen3vl-migration = 5c092f6e25df9d5717495af2bd4ce2b39673978e
```

The Windows local `refs/heads/phase/09-qwen3vl-migration` points to the same commit.

| Field | Value |
| --- | --- |
| Commit hash | `5c092f6e25df9d5717495af2bd4ce2b39673978e` |
| Commit message | `phase9.4: finalize qwen3-vl ai infra project closure` |
| Author date | `2026-09-16T21:29:10+08:00` |
| Committer date | `2026-09-16T21:29:10+08:00` |

Modified files in the branch-head commit:

| Status | File |
| --- | --- |
| M | `CHANGELOG.md` |
| M | `README.md` |
| A | `docs/FINAL_STATUS.md` |
| A | `docs/PROJECT_FINAL_REPORT.md` |
| M | `docs/PROJECT_STATE.md` |
| M | `docs/handoff/current_state.md` |
| A | `docs/handoff/phase9_4_project_closure.md` |

The Phase9 branch also contains a continuous execution history from the Phase9.0 startup audit through Phase9.3-B2 attribution and the Phase9.4 closure. The first Phase9-specific commit after Phase8 closure is:

| Commit | Date | Subject |
| --- | --- | --- |
| `72c41b8` | `2026-09-15T20:50:36+08:00` | `phase9: plan qwen3-vl migration startup` |

The final experiment commit before closure is:

| Commit | Date | Subject |
| --- | --- | --- |
| `42d77ec` | `2026-09-16T18:55:28+08:00` | `phase9.3-b2: attribute decoder runtime bottleneck` |

## Windows Repository

- Current branch: `final-analysis`
- Current HEAD: `fa6b2ce720f1ab8cb623dff59b153271b567e73e`
- The audit inspected `refs/heads/phase/09-qwen3vl-migration` without checking it out.
- `refs/heads/phase/09-qwen3vl-migration` matches GitHub.

### Phase9 Directory

The current Windows filesystem contains:

```text
experiments/Phase9-qwen3-vl-migration
```

Filesystem audit:

| Metric | Value |
| --- | ---: |
| Files | 136 |
| Size | 3.564 MiB |

The Phase9 branch tracks 124 files matching Phase9, including experiment code, artifact metadata, result JSON/CSV/TXT evidence, command logs, and three handoff documents. The working tree contains some additional local untracked files such as `__pycache__`, consistent with the repository-wide local state.

Representative tracked artifact directories:

```text
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_0_20260915T125952Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_0B_20260915T132321Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_1A_20260915T134400Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_1B_20260915T234500Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2A_20260915T161337Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2B1_20260915T164731Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2B2_20260915T171236Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2C1_20260916T034956Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2C2_20260916T042306Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2C2R1_20260916T071026Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2C2R2_20260916T080511Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2D1_20260916T083418Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_3A_20260916T085442Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_3B1_20260916T094011Z
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_3B2_20260916T102808Z
```

### Results

- `results/` exists, but the current Windows filesystem has no `phase9` match under it.
- The Phase9 branch has no tracked file under `results/` matching Phase9.
- This is not an execution gap: Phase9 compact evidence is intentionally under `experiments/Phase9-qwen3-vl-migration/artifacts/`, and large raw Nsight files were kept Jetson-local by policy.

### Experiments

The Phase9 implementation exists under:

```text
experiments/Phase9-qwen3-vl-migration/src/
experiments/Phase9-qwen3-vl-migration/docs/
experiments/Phase9-qwen3-vl-migration/artifacts/
```

Representative source entries:

```text
experiments/Phase9-qwen3-vl-migration/src/phase9_1A/run_fp16_smoke.py
experiments/Phase9-qwen3-vl-migration/src/phase9_1B/run_fp16_baseline.py
experiments/Phase9-qwen3-vl-migration/src/phase9_2B1/export_vision_onnx.py
experiments/Phase9-qwen3-vl-migration/src/phase9_2B2/parse_vision_onnx.py
experiments/Phase9-qwen3-vl-migration/src/phase9_2C1/build_vision_onnx_fp16.py
experiments/Phase9-qwen3-vl-migration/src/phase9_2C2/run_vision_correctness.py
experiments/Phase9-qwen3-vl-migration/src/phase9_2D1/run_vision_latency_benchmark.py
experiments/Phase9-qwen3-vl-migration/src/phase9_3A/run_end_to_end_vision_integration.py
experiments/Phase9-qwen3-vl-migration/src/phase9_3B1/run_end_to_end_stage_breakdown.py
experiments/Phase9-qwen3-vl-migration/src/phase9_3B2/run_decoder_bottleneck_profile.py
```

### Logs

There is no top-level `logs/` directory. Phase9 execution logs are inside artifact directories, for example:

```text
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_1A_20260915T134400Z/attempt1_sdpa_failure/smoke_stdout.txt
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_1B_20260915T234500Z/tegrastats.log
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2C1_20260916T034956Z/build_log_selected.jsonl
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2C2R1_20260916T071026Z/console_attempt2.log
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_3B1_20260916T094011Z/console.log
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_3B2_20260916T102808Z/latency_console.log
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_3B2_20260916T102808Z/profile_console.log
```

### Reports

There is no top-level `reports/` directory. Phase9 reports exist under the experiment and handoff trees:

```text
experiments/Phase9-qwen3-vl-migration/docs/
docs/handoff/phase9_3B1_checkpoint.md
docs/handoff/phase9_4_project_closure.md
docs/handoff/phase9_qwen3vl_migration_plan.md
```

The registry also records Phase9 experiments:

```text
results/experiment_registry.csv
```

Rows include Phase9.0-B through Phase9.3-B2 and explicitly identify `Jetson Orin Nano Super` as the execution platform for the runtime experiments.

## Jetson Repository

- Path: `/home/nvidia/projects/jetson-qwen-inference-lab`
- Current branch: `phase/08-rmsnorm-optimization`
- Current HEAD: `6fb18773014a43f51c82b91338f2972131a4ce86`
- Tracked working tree: clean; only untracked Phase3/legacy artifact paths were present.
- Upstream status: `phase/08-rmsnorm-optimization` is behind `origin/phase/08-rmsnorm-optimization` by 1 commit.
- Current tree Phase9 files: none.
- Local Phase9 refs: none.
- Local `origin/phase/09-qwen3vl-migration` remote-tracking ref: none.
- Filesystem Phase9 hits in the repository: none.
- Top-level `experiments/` contains Phase1, Phase2, Phase5, Phase6, and Phase8, but no Phase9 directory.
- Top-level `results/` contains prior phase evidence, but no Phase9 directory.
- Current `/tmp` contains no `phase9_*` directories.
- Reflog records prior work on Phase3/Phase8 branches and a Phase8 fast-forward, but no Phase9 checkout.

The absence of Phase9 files on the current Jetson checkout is consistent with the recorded Phase9 protocol: the Jetson repository was intentionally frozen at `phase/08-rmsnorm-optimization` and `6fb1877...`, while Phase9 scripts were copied into isolated `/tmp/phase9_*` directories for execution. The command logs explicitly state that no Jetson repository fetch, pull, branch switch, reset, clean, or sync was performed during those runs.

## Runtime Provenance Evidence

### Phase9.1-A Smoke Test

Evidence file:

```text
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_1A_20260915T134400Z/run_manifest.json
```

It records:

```json
{
  "jetson": {
    "hostname": "nvidia-desktop",
    "repository_path": "/home/nvidia/projects/jetson-qwen-inference-lab",
    "repository_branch": "phase/08-rmsnorm-optimization",
    "repository_head": "6fb18773014a43f51c82b91338f2972131a4ce86",
    "runtime_venv": "/home/nvidia/.venvs/jetson-qwen-phase1-hf"
  },
  "checkpoint": {
    "local_path": "/home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203"
  }
}
```

The companion `runtime_dependency_probe.json` records:

```json
{
  "hostname": "nvidia-desktop",
  "linux": "Linux-5.15.148-tegra-aarch64-with-glibc2.35",
  "torch": "2.5.0a0+872d972e41.nv24.08",
  "torch_cuda": "12.6",
  "cuda_device_name": "Orin",
  "cuda_capability": [8, 7]
}
```

### Phase9.1-B PyTorch Baseline

Evidence file:

```text
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_1B_20260915T234500Z/benchmark_result.json
```

It records the same Jetson host, Orin CUDA device, NVIDIA PyTorch/CUDA stack, model path, CUDA allocator evidence, and `tegrastats`-based power sampling.

### Phase9.2-D1 Vision Latency Benchmark

Evidence file:

```text
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_2D1_20260916T083418Z/benchmark_result.json
```

It records:

- `hostname: nvidia-desktop`
- `device: cuda:0`
- `device_name: Orin`
- CUDA capability `[8, 7]`
- NVIDIA PyTorch and CUDA versions
- Jetson-local model path
- Jetson-local TensorRT engine path
- `tegrastats` sampling and 25W power mode

The `benchmark_protocol.json` also records Jetson branch `phase/08-rmsnorm-optimization` and HEAD `6fb1877...`.

### Phase9.3-B2 Decoder Attribution

Evidence file:

```text
experiments/Phase9-qwen3-vl-migration/artifacts/phase9_3B2_20260916T102808Z/artifact_manifest.json
```

It records:

```json
{
  "remote_runtime": {
    "host_alias": "jetson",
    "host_directory": "/tmp/phase9_3b2_20260916T102808Z",
    "model_path": "/home/nvidia/models/qwen3-vl-2b-instruct-89644892e4d85e24eaac8bacfd4f463576704203",
    "engine_path": "/tmp/phase9_2c1_20260916T034956Z/engine/qwen3_vl_vision_fp16.trt",
    "nsys_report_path": "/tmp/phase9_3b2_20260916T102808Z/profile/phase9_3B2_profile.nsys-rep",
    "sqlite_path": "/tmp/phase9_3b2_20260916T102808Z/profile/phase9_3B2_profile.sqlite"
  }
}
```

The same manifest records SHA-256 values for the Jetson-local Nsight report and SQLite export. The result JSON additionally records TensorRT `10.3.0`, Orin, Tegra Linux, and NVIDIA PyTorch.

The command log records the remote command sequence, failed attempt preservation, corrected latency run, final Nsight profile run, and the explicit restriction that the Jetson repository remained frozen without branch switch or sync.

## Judgment

Classification: **A — Phase9 confirmed executed on Jetson.**

Basis:

1. Windows-tracked artifacts include multiple independent evidence files with Jetson-specific host, kernel, model path, runtime path, device, CUDA, and TensorRT metadata.
2. The recorded model and engine hashes are internally consistent across Phase9 protocols and results.
3. Command logs describe execution in `/tmp/phase9_*` while keeping the Jetson Git repository frozen on Phase8.
4. `results/experiment_registry.csv` records Phase9.0-B through Phase9.3-B2 as Jetson Orin Nano Super experiments.
5. The current absence of Phase9 files in the Jetson checkout is expected and does not contradict historical execution.

Limitations:

- The raw Nsight `.nsys-rep` and SQLite files recorded in the Phase9.3-B2 manifest are no longer present under Jetson `/tmp`; only their committed compact artifacts and hashes remain.
- This report does not re-run Phase9 or verify hashes against currently absent Jetson-local raw files.
