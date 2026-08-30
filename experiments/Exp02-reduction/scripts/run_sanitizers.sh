#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP02_BUILD_DIR:-/tmp/jetson-qwen-exp02-build}"
binary="${build_dir}/reduction"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${experiment_dir}/benchmark/raw/sanitizer_${timestamp}"
summary="${output_dir}/summary.csv"
mkdir -p "${output_dir}"
echo "tool,status,version,N,block_size,exit_code,log" > "${summary}"

if ! command -v compute-sanitizer >/dev/null 2>&1; then
  for tool in memcheck racecheck synccheck; do
    echo "${tool},N/A,,,,,compute-sanitizer not found" >> "${summary}"
  done
  exit 0
fi

[[ -x "${binary}" ]] || "${script_dir}/build.sh"
overall=0
for tool in memcheck racecheck synccheck; do
  for version in 2 4 6 7; do
    for n in 1025 65549; do
      block=256
      log="${output_dir}/${tool}_V${version}_N${n}_B${block}.txt"
      set +e
      compute-sanitizer --tool "${tool}" --error-exitcode 86 "${binary}" \
        --single --version "${version}" --pattern signed --n "${n}" \
        --block "${block}" > "${log}" 2>&1
      status=$?
      set -e
      result=PASS
      if [[ "${status}" -ne 0 ]]; then result=FAIL; overall=1; fi
      echo "${tool},${result},V${version},${n},${block},${status},${log}" >> "${summary}"
    done
  done
done
exit "${overall}"

