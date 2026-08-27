#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP01_BUILD_DIR:-/tmp/jetson-qwen-exp01-build}"
binary="${build_dir}/vector_add"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw_dir="${experiment_dir}/benchmark/raw"

environment_raw="${raw_dir}/environment_${timestamp}.txt"
correctness_raw="${raw_dir}/correctness_${timestamp}.csv"
benchmark_raw="${raw_dir}/vector_add_benchmark_${timestamp}.csv"
correctness_log="${experiment_dir}/notes/correctness_${timestamp}.txt"
benchmark_log="${experiment_dir}/notes/benchmark_${timestamp}.txt"

environment_canonical="${experiment_dir}/benchmark/environment.txt"
correctness_canonical="${experiment_dir}/benchmark/correctness_results.csv"
benchmark_canonical="${experiment_dir}/benchmark/vector_add_benchmark.csv"

for raw_output in \
  "${environment_raw}" \
  "${correctness_raw}" \
  "${benchmark_raw}" \
  "${correctness_log}" \
  "${benchmark_log}"; do
  if [[ -e "${raw_output}" ]]; then
    echo "Refusing to overwrite timestamped output: ${raw_output}" >&2
    exit 2
  fi
done

publish_canonical() {
  local raw_output="$1"
  local canonical_output="$2"
  if [[ -e "${canonical_output}" ]]; then
    echo "Preserving existing canonical result: ${canonical_output}"
  else
    cp "${raw_output}" "${canonical_output}"
  fi
}

mkdir -p "${raw_dir}" "${experiment_dir}/notes"

if [[ ! -x "${binary}" ]]; then
  "${script_dir}/build.sh"
fi

{
  echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "## hostname"
  hostname
  echo
  echo "## uname -a"
  uname -a
  echo
  echo "## /etc/os-release"
  cat /etc/os-release
  echo
  echo "## nvcc --version"
  nvcc --version
  echo
  echo "## nvidia-smi"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi: NOT FOUND"
  fi
  echo
  echo "## tegrastats --help"
  if command -v tegrastats >/dev/null 2>&1; then
    timeout 5s tegrastats --help 2>&1 | head -n 20
  else
    echo "tegrastats: NOT FOUND"
  fi
  echo
  echo "## gcc --version"
  gcc --version | head -n 1
  echo
  echo "## g++ --version"
  g++ --version | head -n 1
  echo
  echo "## CUDA runtime device properties"
  "${binary}" --device-info
} | tee "${environment_raw}"

publish_canonical "${environment_raw}" "${environment_canonical}"

set +e
"${binary}" --correctness --csv "${correctness_raw}" \
  2>&1 | tee "${correctness_log}"
correctness_exit="${PIPESTATUS[0]}"
set -e
publish_canonical "${correctness_raw}" "${correctness_canonical}"
if [[ "${correctness_exit}" -ne 0 ]]; then
  echo "Correctness sweep failed with exit ${correctness_exit}; benchmark skipped." >&2
  exit "${correctness_exit}"
fi

set +e
"${binary}" \
  --benchmark \
  --csv "${benchmark_raw}" \
  --warmup 20 \
  --repetitions 200 \
  2>&1 | tee "${benchmark_log}"
benchmark_exit="${PIPESTATUS[0]}"
set -e
publish_canonical "${benchmark_raw}" "${benchmark_canonical}"
if [[ "${benchmark_exit}" -ne 0 ]]; then
  echo "Benchmark sweep failed with exit ${benchmark_exit}." >&2
  exit "${benchmark_exit}"
fi

echo "environment=${environment_canonical}"
echo "correctness=${correctness_canonical}"
echo "benchmark=${benchmark_canonical}"
