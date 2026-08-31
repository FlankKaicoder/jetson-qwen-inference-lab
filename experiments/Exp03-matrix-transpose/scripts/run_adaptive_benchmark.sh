#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
repo_dir="$(cd "${experiment_dir}/../.." && pwd)"
kind="${1:?kind: diagnostic_A, diagnostic_B or benchmark_v2}"
out_dir="${2:-/tmp/jetson-qwen-exp03-stability-$(date -u +%Y%m%dT%H%M%SZ)}"
build_dir="${EXP03_BUILD_DIR:-/tmp/jetson-qwen-exp03-build}"
binary="${build_dir}/transpose_benchmark"
mkdir -p "$out_dir"
timestamp="$(basename "$out_dir" | sed 's/.*-//')"

case "$kind" in
  diagnostic_A)
    warm_target=1000; measure_target=500; raw_prefix=stability_diagnostic_A
    summary_prefix=stability_diagnostic_A_summary; telemetry_prefix=stability_telemetry_A
    calibration_prefix=stability_calibration_A; order_prefix=stability_order_A
    dims=("1024 1024" "4096 4096")
    ;;
  diagnostic_B)
    warm_target=2000; measure_target=1000; raw_prefix=stability_diagnostic_B
    summary_prefix=stability_diagnostic_B_summary; telemetry_prefix=stability_telemetry_B
    calibration_prefix=stability_calibration_B; order_prefix=stability_order_B
    dims=("1024 1024" "4096 4096")
    ;;
  benchmark_v2)
    warm_target=1000; measure_target=500; raw_prefix=benchmark_v2
    summary_prefix=benchmark_v2_summary; telemetry_prefix=benchmark_v2_telemetry
    calibration_prefix=benchmark_v2_calibration; order_prefix=benchmark_v2_order
    dims=("1024 1024" "2048 2048" "4096 4096" "4093 4093" "4096 2048" "2048 4096")
    ;;
  *)
    echo "unknown kind: $kind" >&2
    exit 2
    ;;
esac

bash "${script_dir}/build.sh"
raw_tmp="${out_dir}/${raw_prefix}_measurements_${timestamp}.csv"
raw="${out_dir}/${raw_prefix}_${timestamp}.csv"
summary="${out_dir}/${summary_prefix}_${timestamp}.csv"
telemetry="${out_dir}/${telemetry_prefix}_${timestamp}.csv"
calibration="${out_dir}/${calibration_prefix}_${timestamp}.csv"
order="${out_dir}/${order_prefix}_${timestamp}.csv"
environment="${out_dir}/environment_${timestamp}.txt"
sanity="${out_dir}/${raw_prefix}_sanity_${timestamp}.txt"
for file in "$raw_tmp" "$raw" "$summary" "$telemetry" "$calibration" "$order" "$environment" "$sanity"; do
  [[ ! -e "$file" ]] || { echo "refusing to overwrite $file" >&2; exit 2; }
done

{
  echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  hostname
  uname -a
  echo "git_branch=$(git -C "$repo_dir" branch --show-current)"
  echo "git_head=$(git -C "$repo_dir" rev-parse HEAD)"
  echo "git_status=$(git -C "$repo_dir" status --short | tr '\n' ' ')"
  "${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" --version
  /usr/local/cuda-12.6/bin/ncu --version 2>/dev/null || echo "ncu=unavailable"
  "$binary" --device-info
  echo "kind=$kind"
  echo "warmup_target_ms=$warm_target"
  echo "measurement_target_ms=$measure_target"
  echo "measurement_safety_factor=2.0"
  echo "calibration_iterations=50"
  echo "min_warmup=20"
  echo "max_warmup=20000"
  echo "min_measurement=100"
  echo "max_measurement=10000"
  echo "trials=7"
  echo "TILE_DIM=32"
  echo "BLOCK_ROWS=8"
} > "$environment"

for version in V1 V2 V3 V4; do
  "$binary" --version "$version" --width 1024 --height 1024 --sanity-only >> "$sanity"
done

for dimension in "${dims[@]}"; do
  read -r width height <<< "$dimension"
  for version in V1 V2 V3 V4; do
    "$binary" --version "$version" --width "$width" --height "$height" \
      --calibrate --calibration-warmup 20 --calibration-iterations 50 \
      --warmup-target-ms "$warm_target" --max-warmup 20000 \
      --calibration-output "$calibration" > "${out_dir}/cal_${width}x${height}_${version}.txt"
  done
done

echo "dimension,trial,order" > "$order"
orders=(
  "V1 V2 V3 V4"
  "V2 V3 V4 V1"
  "V3 V4 V1 V2"
  "V4 V1 V2 V3"
  "V1 V3 V2 V4"
  "V2 V4 V3 V1"
  "V4 V2 V1 V3"
)
for dimension in "${dims[@]}"; do
  read -r width height <<< "$dimension"
  for trial in {1..7}; do
    order_value="${orders[$((trial-1))]}"
    echo "${width}x${height},${trial},${order_value}" >> "$order"
    for version in $order_value; do
      calibration_file="${out_dir}/cal_${width}x${height}_${version}.txt"
      calibrated_ms="$(tail -n 1 "$calibration_file")"
      "$binary" --version "$version" --width "$width" --height "$height" \
        --adaptive --trial "$trial" --calibrated-ms "$calibrated_ms" \
        --warmup-target-ms "$warm_target" --measurement-target-ms "$measure_target" \
        --measurement-safety-factor 2.0 \
        --min-warmup 20 --max-warmup 20000 --min-measurement 100 --max-measurement 10000 \
        --raw "$raw_tmp" --telemetry-script "${script_dir}/capture_tegrastats.sh" \
        --telemetry-output "$telemetry"
    done
  done
done

for version in V1 V2 V3 V4; do
  "$binary" --version "$version" --width 1024 --height 1024 --sanity-only >> "$sanity"
done

python3 "${script_dir}/merge_stability_telemetry.py" "$raw_tmp" "$telemetry" "$raw"
python3 "${script_dir}/summarize_stability.py" "$raw" > "$summary"
(
  cd "$out_dir"
  sha256sum "$(basename "$raw")" "$(basename "$summary")" \
    "$(basename "$telemetry")" "$(basename "$calibration")" \
    "$(basename "$order")" "$(basename "$environment")" "$(basename "$sanity")"
) > "${out_dir}/SHA256SUMS"

echo "raw=$raw"
echo "summary=$summary"
echo "telemetry=$telemetry"
echo "calibration=$calibration"
echo "order=$order"
echo "environment=$environment"
