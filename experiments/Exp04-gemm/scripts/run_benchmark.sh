#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; experiment_dir="$(cd "${script_dir}/.." && pwd)"; build_dir="${EXP04_BUILD_DIR:-/tmp/jetson-qwen-exp04-build}"; binary="${build_dir}/gemm"; timestamp="$(date -u +%Y%m%dT%H%M%SZ)"; raw_dir="${experiment_dir}/benchmark/raw"; mkdir -p "$raw_dir"
raw="${raw_dir}/benchmark_v1_${timestamp}.csv"; summary="${raw_dir}/benchmark_v1_summary_${timestamp}.csv"; envf="${raw_dir}/benchmark_environment_${timestamp}.txt"
for f in "$raw" "$summary" "$envf"; do [[ ! -e "$f" ]] || { echo "Refusing to overwrite $f" >&2; exit 2; }; done
[[ -x "$binary" ]] || bash "${script_dir}/build.sh"
{ echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; hostname; uname -a; echo "git_branch=$(git -C "${experiment_dir}/../.." branch --show-current)"; echo "git_head=$(git -C "${experiment_dir}/../.." rev-parse HEAD)"; "${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" --version; "$binary" --device-info; echo "warmup_target_ms=1000"; echo "measurement_target_ms=500"; echo "measurement_safety_factor=2.0"; echo "trials=7"; echo "timing=CUDA Event kernel-only"; echo "FLOPs=2*M*N*K"; } > "$envf"
echo 'timestamp,version,M,K,N,trial,warmup_iterations,measurement_iterations,calibrated_ms,actual_window_ms,latency_ms,gflops' > "$raw"
shapes=("256 256 256" "512 512 512" "1024 1024 1024" "512 384 640")
for shape in "${shapes[@]}"; do read -r m k n <<< "$shape"; calibrated="$("$binary" --m "$m" --k "$k" --n "$n" --calibrate)"; echo "shape=${m}x${k}x${n},calibrated_ms=${calibrated}" >> "$envf"; for trial in {1..7}; do "$binary" --m "$m" --k "$k" --n "$n" --benchmark-single --trial "$trial" --calibrated-ms "$calibrated" --raw "$raw"; done; done
python3 "${script_dir}/summarize_benchmark.py" "$raw" "$summary"
echo "raw=$raw"; echo "summary=$summary"; echo "environment=$envf"
