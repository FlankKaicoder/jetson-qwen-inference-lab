#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP01_BUILD_DIR:-/tmp/jetson-qwen-exp01-build}"
binary="${build_dir}/vector_add"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

environment_output="${experiment_dir}/benchmark/raw/environment_exp01_1_${timestamp}.txt"
raw_output="${experiment_dir}/benchmark/stability_raw_${timestamp}.csv"
summary_output="${experiment_dir}/benchmark/stability_summary.csv"
csv_test_output="${experiment_dir}/notes/csv_escape_test_${timestamp}.txt"
console_output="${experiment_dir}/notes/stability_${timestamp}.txt"

for output in \
  "${environment_output}" \
  "${raw_output}" \
  "${summary_output}" \
  "${csv_test_output}" \
  "${console_output}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing output: ${output}" >&2
    exit 2
  fi
done

mkdir -p "${experiment_dir}/benchmark/raw" "${experiment_dir}/notes"
"${script_dir}/build.sh"

{
  echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "measurement_scope=Exp01.1 stability pre-run read-only state"
  echo
  echo "## nvpmodel -q"
  if nvpmodel -q; then
    echo "nvpmodel_exit=0"
  else
    echo "nvpmodel_exit=$?"
  fi
  echo
  echo "## jetson_clocks --show (non-root; no state change)"
  if jetson_clocks --show; then
    echo "jetson_clocks_show_exit=0"
  else
    echo "jetson_clocks_show_exit=$?"
  fi
  echo
  echo "## GPU devfreq sysfs"
  gpu_devfreq=/sys/class/devfreq/17000000.gpu
  for field in cur_freq min_freq max_freq available_frequencies; do
    if [[ -r "${gpu_devfreq}/${field}" ]]; then
      printf '%s=' "${field}"
      cat "${gpu_devfreq}/${field}"
    fi
  done
  echo
  echo "## EMC / memory frequency candidates"
  emc_found=0
  for emc_path in \
    /sys/kernel/debug/bpmp/debug/clk/emc/rate \
    /sys/kernel/debug/clk/emc/clk_rate; do
    if [[ -r "${emc_path}" ]]; then
      printf '%s=' "${emc_path}"
      cat "${emc_path}"
      emc_found=1
    fi
  done
  if [[ "${emc_found}" -eq 0 ]]; then
    echo "EMC frequency unavailable to the non-root user"
  fi
  echo
  echo "## tegrastats pre-run (3 samples)"
  if timeout 4s tegrastats --interval 1000; then
    echo "tegrastats_pre_exit=0"
  else
    status="$?"
    echo "tegrastats_pre_exit=${status} (124 means expected timeout)"
  fi
  echo
  echo "## CUDA runtime device properties"
  "${binary}" --device-info
} 2>&1 | tee "${environment_output}"

"${binary}" --test-csv-escape 2>&1 | tee "${csv_test_output}"

set +e
"${binary}" \
  --stability \
  --csv "${raw_output}" \
  --summary "${summary_output}" \
  --warmup 20 \
  --repetitions 200 \
  2>&1 | tee "${console_output}"
stability_exit="${PIPESTATUS[0]}"
set -e

{
  echo
  echo "## tegrastats post-run (3 samples)"
  if timeout 4s tegrastats --interval 1000; then
    echo "tegrastats_post_exit=0"
  else
    status="$?"
    echo "tegrastats_post_exit=${status} (124 means expected timeout)"
  fi
  echo "post_run_collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "${environment_output}"

if [[ "${stability_exit}" -ne 0 ]]; then
  echo "Stability benchmark failed with exit ${stability_exit}." >&2
  exit "${stability_exit}"
fi

echo "environment=${environment_output}"
echo "stability_raw=${raw_output}"
echo "stability_summary=${summary_output}"
echo "csv_escape_test=${csv_test_output}"
echo "console_log=${console_output}"
