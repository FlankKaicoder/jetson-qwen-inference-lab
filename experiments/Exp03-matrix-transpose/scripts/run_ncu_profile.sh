#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${experiment_dir}/../.." && pwd)"
timestamp="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
out_dir="${2:-/tmp/jetson-qwen-exp03-ncu-${timestamp}}"
build_dir="${EXP03_BUILD_DIR:-/tmp/jetson-qwen-exp03-build}"
binary="${build_dir}/transpose_benchmark"
ncu="/usr/local/cuda-12.6/bin/ncu"
mkdir -p "$out_dir"
[[ -z "$(find "$out_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "refusing non-empty output directory: $out_dir" >&2; exit 2; }
bash "${script_dir}/build.sh"
"$ncu" --version > "${out_dir}/ncu_version.txt"
sudo -n "$ncu" --version > /dev/null
sudo -n "$ncu" --list-sections > "${out_dir}/ncu_sections.txt"
{
  echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  hostname
  uname -a
  echo "git_branch=$(git -C "$repo_dir" branch --show-current)"
  echo "git_head=$(git -C "$repo_dir" rev-parse HEAD)"
  echo "git_status=$(git -C "$repo_dir" status --short | tr '\n' ' ')"
  "${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" --version
  cat "${out_dir}/ncu_version.txt"
  "$binary" --device-info
  echo "width=4096"
  echo "height=4096"
  echo "TILE_DIM=32"
  echo "BLOCK_ROWS=8"
  echo "launch_skip=1"
  echo "launch_count=1"
} > "${out_dir}/environment.txt"
echo "version,status" > "${out_dir}/sanity.txt"
for version in V1 V2 V3 V4; do
  "$binary" --version "$version" --width 4096 --height 4096 --sanity-only >/dev/null
  echo "${version},PASS" >> "${out_dir}/sanity.txt"
done
echo "version,report,details,raw_csv,stderr" > "${out_dir}/manifest.csv"
for version in V1 V2 V3 V4; do
  lower="$(echo "$version" | tr '[:upper:]' '[:lower:]')"
  base="ncu_${lower}_4096_${timestamp}"
  report="${out_dir}/${base}.ncu-rep"
  details="${out_dir}/${base}_details.txt"
  raw="${out_dir}/${base}_raw.csv"
  stderr="${out_dir}/${base}_stderr.txt"
  printf 'sudo -n %q --launch-skip 1 --launch-count 1 --export %q [sections] %q --version %q --width 4096 --height 4096 --single\n' "$ncu" "$report" "$binary" "$version" >> "${out_dir}/commands.txt"
  sudo -n "$ncu" --force-overwrite --kernel-name-base function \
    --launch-skip 1 --launch-count 1 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables --section Occupancy \
    --section WarpStateStats --section SchedulerStats \
    --section LaunchStats --section SourceCounters \
    --export "$report" "$binary" --version "$version" --width 4096 --height 4096 --single \
    > "${out_dir}/${base}_stdout.txt" 2> "$stderr"
  sudo -n "$ncu" --import "$report" --page details > "$details"
  sudo -n "$ncu" --import "$report" --csv --page raw > "$raw"
  echo "${version},${base}.ncu-rep,${base}_details.txt,${base}_raw.csv,${base}_stderr.txt" >> "${out_dir}/manifest.csv"
done
for version in V1 V2 V3 V4; do
  "$binary" --version "$version" --width 4096 --height 4096 --sanity-only >/dev/null
  echo "${version},PASS_POST" >> "${out_dir}/sanity.txt"
done
(
  cd "$out_dir"
  sha256sum *.ncu-rep *_details.txt *_raw.csv *_stderr.txt environment.txt manifest.csv sanity.txt commands.txt
) > "${out_dir}/SHA256SUMS"
echo "output_dir=$out_dir"
