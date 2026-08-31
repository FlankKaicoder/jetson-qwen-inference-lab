#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP03_BUILD_DIR:-/tmp/jetson-qwen-exp03-build}"
binary="${build_dir}/transpose"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw_dir="${experiment_dir}/benchmark/raw"
mkdir -p "${raw_dir}"
raw="${raw_dir}/correctness_${timestamp}.csv"; summary="${raw_dir}/correctness_summary_${timestamp}.csv"; envf="${raw_dir}/environment_${timestamp}.txt"
for f in "$raw" "$summary" "$envf"; do [[ ! -e "$f" ]] || { echo "Refusing to overwrite $f" >&2; exit 2; }; done
[[ -x "$binary" ]] || "${script_dir}/build.sh"
{ echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; hostname; uname -a; "${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" --version; "$binary" --device-info; } | tee "$envf"
"$binary" --correctness --raw "$raw" --summary "$summary"
echo "raw=$raw"; echo "summary=$summary"; echo "environment=$envf"; echo "compute-sanitizer=N/A (not installed; not run)"
