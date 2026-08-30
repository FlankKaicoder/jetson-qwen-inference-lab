#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP02_BUILD_DIR:-/tmp/jetson-qwen-exp02-build}"
binary="${build_dir}/reduction"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw_dir="${experiment_dir}/benchmark/raw"
mkdir -p "${raw_dir}"

environment="${raw_dir}/environment_${timestamp}.txt"
correctness_raw="${raw_dir}/correctness_${timestamp}.csv"
correctness_summary="${raw_dir}/correctness_summary_${timestamp}.csv"
sanitizer_discovery="${raw_dir}/sanitizer_discovery_${timestamp}.txt"
for output in "${environment}" "${correctness_raw}" "${correctness_summary}" "${sanitizer_discovery}"; do
  [[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 2; }
done

if [[ ! -x "${binary}" ]]; then "${script_dir}/build.sh"; fi
{
  echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  hostname
  uname -a
  nvcc --version
  "${binary}" --device-info
} | tee "${environment}"

"${binary}" --correctness --raw "${correctness_raw}" --summary "${correctness_summary}"

{
  echo "sanitizer_discovery_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if command -v compute-sanitizer >/dev/null 2>&1; then
    compute-sanitizer --version
    compute-sanitizer --help | head -n 80
  else
    echo "compute-sanitizer=N/A (command not found)"
  fi
} | tee "${sanitizer_discovery}"

echo "environment=${environment}"
echo "correctness_raw=${correctness_raw}"
echo "correctness_summary=${correctness_summary}"
echo "sanitizer_discovery=${sanitizer_discovery}"

