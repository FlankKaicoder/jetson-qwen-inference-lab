#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
cuda_bin="/usr/local/cuda-12.6/bin"
build_dir="${EXP01_BUILD_DIR:-/tmp/jetson-qwen-exp01-build}"
binary="${build_dir}/vector_add"
ncu="${cuda_bin}/ncu"
timestamp="${EXP01_PROFILE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${experiment_dir}/benchmark/profiler/${timestamp}"
report_dir="${EXP01_NCU_REPORT_DIR:-/tmp/jetson-qwen-exp01-ncu/${timestamp}}"

n=16777216
warmup=20
repetitions=200
blocks=(32 128 256 1024)
sections=(
  LaunchStats
  Occupancy
  SpeedOfLight
  MemoryWorkloadAnalysis_Tables
  SchedulerStats
  WarpStateStats
  SourceCounters
)

export PATH="${cuda_bin}:${PATH}"

if [[ -n "${EXP01_PROFILE_BLOCKS:-}" ]]; then
  read -r -a blocks <<<"${EXP01_PROFILE_BLOCKS}"
fi

if [[ -e "${output_dir}" || -e "${report_dir}" ]]; then
  echo "Refusing to overwrite profiler output: ${output_dir} or ${report_dir}" >&2
  exit 2
fi

if [[ ! -x "${ncu}" ]]; then
  echo "Nsight Compute not executable: ${ncu}" >&2
  exit 3
fi
if [[ ! -x "${cuda_bin}/nvcc" ]]; then
  echo "NVCC not executable: ${cuda_bin}/nvcc" >&2
  exit 3
fi

sudo -n "${ncu}" --version >/dev/null
sudo -n "${ncu}" --list-sections >/dev/null

mkdir -p "${output_dir}" "${report_dir}"
"${script_dir}/build.sh"

section_args=()
for section in "${sections[@]}"; do
  section_args+=(--section "${section}")
done

{
  echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "repository=$(git -C "${experiment_dir}" rev-parse --show-toplevel)"
  echo "branch=$(git -C "${experiment_dir}" branch --show-current)"
  echo "head=$(git -C "${experiment_dir}" rev-parse HEAD)"
  echo "n=${n}"
  echo "warmup=${warmup}"
  echo "repetitions=${repetitions}"
  echo "profiled_launch=first measured launch after ${warmup} warmup launches"
  echo "kernel_filter=regex:vectorAddKernel"
  echo "launch_skip=${warmup}"
  echo "launch_count=1"
  echo "sections=$(IFS=,; echo "${sections[*]}")"
  echo "ncu_report_dir=${report_dir}"
  "${ncu}" --version
} >"${output_dir}/environment.txt"

printf 'block_size,profile_status,correctness,report_path,details_path,raw_csv_path\n' \
  >"${output_dir}/manifest.csv"

for block in "${blocks[@]}"; do
  report_base="${report_dir}/block${block}"
  profile_log="${output_dir}/block${block}_profile.txt"
  details_path="${output_dir}/block${block}_details.txt"
  raw_csv_path="${output_dir}/block${block}_raw.csv"

  set +e
  sudo -n "${ncu}" \
    "${section_args[@]}" \
    --kernel-name-base function \
    --kernel-name 'regex:vectorAddKernel' \
    --launch-skip "${warmup}" \
    --launch-count 1 \
    --export "${report_base}" \
    "${binary}" \
      --single \
      --n "${n}" \
      --block "${block}" \
      --warmup "${warmup}" \
      --repetitions "${repetitions}" \
    2>&1 | tee "${profile_log}"
  profile_exit="${PIPESTATUS[0]}"
  set -e

  if [[ "${profile_exit}" -ne 0 ]]; then
    printf '%s,FAIL,UNKNOWN,%s,%s,%s\n' \
      "${block}" "${report_base}.ncu-rep" "${details_path}" "${raw_csv_path}" \
      >>"${output_dir}/manifest.csv"
    echo "NCU profiling failed for block ${block} with exit ${profile_exit}" >&2
    exit "${profile_exit}"
  fi

  if ! grep -q '^correctness=PASS$' "${profile_log}"; then
    printf '%s,FAIL,FAIL,%s,%s,%s\n' \
      "${block}" "${report_base}.ncu-rep" "${details_path}" "${raw_csv_path}" \
      >>"${output_dir}/manifest.csv"
    echo "Correctness did not pass for block ${block}" >&2
    exit 4
  fi

  sudo -n "${ncu}" \
    --import "${report_base}.ncu-rep" \
    --page details \
    --print-details all \
    >"${details_path}"
  sudo -n "${ncu}" \
    --import "${report_base}.ncu-rep" \
    --page raw \
    --csv \
    >"${raw_csv_path}"

  printf '%s,PASS,PASS,%s,%s,%s\n' \
    "${block}" "${report_base}.ncu-rep" "${details_path}" "${raw_csv_path}" \
    >>"${output_dir}/manifest.csv"
done

echo "profile_output=${output_dir}"
echo "local_ncu_reports=${report_dir}"
