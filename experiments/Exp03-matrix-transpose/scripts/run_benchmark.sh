#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_dir="$(cd "${script_dir}/.." && pwd)"
build_dir="${EXP03_BUILD_DIR:-/tmp/jetson-qwen-exp03-build}"
binary="${build_dir}/transpose_benchmark"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
raw_dir="${experiment_dir}/benchmark/raw"
mkdir -p "${raw_dir}"
raw="${raw_dir}/benchmark_${timestamp}.csv"
order="${raw_dir}/benchmark_order_${timestamp}.csv"
envf="${raw_dir}/environment_${timestamp}.txt"
for f in "$raw" "$order" "$envf"; do [[ ! -e "$f" ]] || { echo "Refusing to overwrite $f" >&2; exit 2; }; done
[[ -x "$binary" ]] || bash "${script_dir}/build.sh"
{ echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; hostname; uname -a; "${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" --version; /usr/local/cuda-12.6/bin/ncu --version 2>/dev/null || echo "ncu=unavailable"; "$binary" --device-info; echo "warmup=20"; echo "repetitions=100"; echo "trials=5"; echo "TILE_DIM=32"; echo "BLOCK_ROWS=8"; } | tee "$envf"
echo "version,width,height,trial,warmup,repetitions,mean_ms,effective_gb_s" > "$raw"
echo "dimension,trial,order" > "$order"
dims=("1024 1024" "2048 2048" "4096 4096" "4093 4093" "4096 2048" "2048 4096")
orders=("V1 V2 V3 V4" "V2 V3 V4 V1" "V3 V4 V1 V2" "V4 V1 V2 V3" "V1 V3 V2 V4")
for idx in "${!dims[@]}"; do read -r w h <<< "${dims[$idx]}"; for trial in {1..5}; do ord="${orders[$(( (trial-1) % 5 ))]}"; echo "${w}x${h},${trial},${ord}" >> "$order"; for v in $ord; do "$binary" --version "$v" --width "$w" --height "$h" --warmup 20 --repetitions 100 --trials 1 --output "$raw"; done; done; done
summary="${raw_dir}/benchmark_summary_${timestamp}.csv"; python3 "${script_dir}/summarize_benchmark.py" "$raw" > "$summary"; echo "raw=$raw"; echo "summary=$summary"; echo "order=$order"; echo "environment=$envf"
