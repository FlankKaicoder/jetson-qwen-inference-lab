#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP02_BUILD_DIR:-/tmp/jetson-qwen-exp02-build}"
binary="${build_dir}/reduction"
mkdir -p "${build_dir}"

${NVCC:-/usr/local/cuda-12.6/bin/nvcc} -O3 -lineinfo -std=c++17 -Xcompiler=-Wall,-Wextra \
  "${experiment_dir}/src/reduction.cu" -o "${binary}"
echo "binary=${binary}"

