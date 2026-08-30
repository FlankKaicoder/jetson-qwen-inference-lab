#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP02_BUILD_DIR:-/tmp/jetson-qwen-exp02-build}"
binary="${build_dir}/reduction"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw_dir="${experiment_dir}/benchmark/raw"
mkdir -p "${raw_dir}"
b1_raw="${raw_dir}/block_survey_${timestamp}.csv"
b2_raw="${raw_dir}/scaling_${timestamp}.csv"
b3_raw="${raw_dir}/stability_${timestamp}.csv"
b1_summary="${raw_dir}/block_survey_summary_${timestamp}.csv"
b2_summary="${raw_dir}/scaling_summary_${timestamp}.csv"
b3_summary="${raw_dir}/stability_summary_${timestamp}.csv"
choices="${raw_dir}/block_candidates_${timestamp}.csv"
[[ -x "${binary}" ]] || "${script_dir}/build.sh"
run_case() {
  local raw="$1" version="$2" n="$3" block="$4" round="$5" reps="$6"
  "${binary}" --benchmark-single --raw "${raw}" --version "${version}" --pattern signed --n "${n}" --block "${block}" --warmup 20 --repetitions "${reps}" --round "${round}"
}
versions=(1 2 3 4 5 6 7)
blocks=(32 64 128 256 512 1024)
N_B1=$((2**24 + 13))
for round in 1 2 3; do
  case "${round}" in
    1) vorder=(1 2 3 4 5 6 7); border=(32 64 128 256 512 1024) ;;
    2) vorder=(7 6 5 4 3 2 1); border=(1024 512 256 128 64 32) ;;
    3) vorder=(4 5 6 7 1 2 3); border=(128 256 512 1024 32 64) ;;
  esac
  for version in "${vorder[@]}"; do
    for block in "${border[@]}"; do run_case "${b1_raw}" "${version}" "${N_B1}" "${block}" "${round}" 50; done
  done
done
python3 "${script_dir}/analyze_benchmark.py" summary "${b1_raw}" "${b1_summary}"
python3 "${script_dir}/analyze_benchmark.py" choose "${b1_raw}" "${choices}"
for round in 1 2 3; do
  case "${round}" in
    1) vorder=(1 2 3 4 5 6 7) ;;
    2) vorder=(7 6 5 4 3 2 1) ;;
    3) vorder=(4 5 6 7 1 2 3) ;;
  esac
  for version in "${vorder[@]}"; do
    block="$(awk -F, -v v="V${version}" '$1==v {print $2}' "${choices}")"
    for n in 1048576 4194304 16777216 16777229; do run_case "${b2_raw}" "${version}" "${n}" "${block}" "${round}" 100; done
  done
done
python3 "${script_dir}/analyze_benchmark.py" summary "${b2_raw}" "${b2_summary}"
for round in 1 2 3 4 5; do
  case "${round}" in
    1) vorder=(1 2 3 4 5 6 7) ;;
    2) vorder=(4 5 6 7 1 2 3) ;;
    3) vorder=(7 6 5 4 3 2 1) ;;
    4) vorder=(2 3 4 5 6 7 1) ;;
    5) vorder=(6 7 1 2 3 4 5) ;;
  esac
  for version in "${vorder[@]}"; do
    block="$(awk -F, -v v="V${version}" '$1==v {print $2}' "${choices}")"
    run_case "${b3_raw}" "${version}" "${N_B1}" "${block}" "${round}" 200
  done
done
python3 "${script_dir}/analyze_benchmark.py" summary "${b3_raw}" "${b3_summary}"
printf 'block_survey_raw=%s\nscaling_raw=%s\nstability_raw=%s\n' "${b1_raw}" "${b2_raw}" "${b3_raw}"
printf 'block_survey_summary=%s\nscaling_summary=%s\nstability_summary=%s\nchoices=%s\n' "${b1_summary}" "${b2_summary}" "${b3_summary}" "${choices}"