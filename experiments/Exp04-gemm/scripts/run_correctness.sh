#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP04_BUILD_DIR:-/tmp/jetson-qwen-exp04-build}"
binary="${build_dir}/gemm"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw_dir="${experiment_dir}/benchmark/raw"; mkdir -p "${raw_dir}"
raw="${raw_dir}/correctness_${timestamp}.csv"; summary="${raw_dir}/correctness_summary_${timestamp}.txt"; envf="${raw_dir}/environment_${timestamp}.txt"
for f in "$raw" "$summary" "$envf"; do [[ ! -e "$f" ]] || { echo "Refusing to overwrite $f" >&2; exit 2; }; done
[[ -x "$binary" ]] || bash "${script_dir}/build.sh"
{ echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; hostname; uname -a; echo "git_branch=$(git -C "${experiment_dir}/../.." branch --show-current)"; echo "git_head=$(git -C "${experiment_dir}/../.." rev-parse HEAD)"; "${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" --version; "$binary" --device-info; echo "compute-sanitizer=N/A (not checked in initial stage)"; } | tee "$envf"
"$binary" --correctness --raw "$raw" | tee "$summary"
echo "raw=$raw"; echo "summary=$summary"; echo "environment=$envf"
