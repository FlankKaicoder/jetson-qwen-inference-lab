#!/usr/bin/env bash
set -euo pipefail

script_path="${1:?usage: run_phase8_0_baseline.sh RMSNORM_SCRIPT [OUTPUT_DIR]}"
output_dir="${2:-/home/nvidia/projects/jetson-qwen-inference-lab/experiments/Phase8-rmsnorm-optimization/artifacts/phase8_0_$(date -u +%Y%m%dT%H%M%SZ)}"
python_bin="${PYTHON_BIN:-/home/nvidia/.venvs/jetson-qwen-phase1-hf/bin/python}"
model_dir="${MODEL_DIR:-/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca}"
repo_dir="${REPO_DIR:-/home/nvidia/projects/jetson-qwen-inference-lab}"

mkdir -p "$(dirname "$output_dir")"
"$python_bin" "$script_path" \
  --model-dir "$model_dir" \
  --output-dir "$output_dir" \
  --warmup 50 \
  --repetitions 200 \
  --trials 5

{
  echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "repo_dir=$repo_dir"
  echo "git_branch=$(git -C "$repo_dir" branch --show-current)"
  echo "git_head=$(git -C "$repo_dir" rev-parse HEAD)"
  echo "git_status_short_start=$(git -C "$repo_dir" status --short | wc -l)"
  echo "script_sha256=$(sha256sum "$script_path" | awk '{print $1}')"
  echo "model_sha256=$(sha256sum "$model_dir/model.safetensors" | awk '{print $1}')"
  echo "model_config_sha256=$(sha256sum "$model_dir/config.json" | awk '{print $1}')"
} > "$output_dir/run_manifest.txt"

sha256sum "$output_dir"/rmsnorm_baseline_results.json \
  "$output_dir"/rmsnorm_baseline_summary.csv \
  "$output_dir"/rmsnorm_baseline_trials.csv \
  "$output_dir"/environment.json > "$output_dir/artifact_sha256sums.txt"

echo "$output_dir"
