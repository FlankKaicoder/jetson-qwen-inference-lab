#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP04_BUILD_DIR:-/tmp/jetson-qwen-exp04-build}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw="${experiment_dir}/benchmark/raw/wmma_correctness_dual_reference_${timestamp}.csv"
[[ ! -e "$raw" ]] || { echo "refusing overwrite" >&2; exit 2; }
"${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" -O3 -std=c++17 -arch=sm_87 "${experiment_dir}/src/wmma_dual_correctness.cu" -o "${build_dir}/wmma_dual_correctness"
"${build_dir}/wmma_dual_correctness" "$raw"
echo "raw=$raw"
