#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP01_BUILD_DIR:-/tmp/jetson-qwen-exp01-build}"
binary="${build_dir}/vector_add"

mkdir -p "${build_dir}"

nvcc \
  -O3 \
  -lineinfo \
  -std=c++17 \
  -Xcompiler=-Wall,-Wextra \
  "${experiment_dir}/src/vector_add.cu" \
  -o "${binary}"

echo "binary=${binary}"
