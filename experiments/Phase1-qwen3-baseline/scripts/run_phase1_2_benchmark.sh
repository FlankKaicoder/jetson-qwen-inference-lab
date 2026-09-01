#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "usage: $0 SNAPSHOT OUTPUT_DIR MODE ISL [WARMUPS] [TRIALS]" >&2
    exit 2
fi

snapshot=$1
output_dir=$2
mode=$3
isl=$4
warmups=${5:-2}
trials=${6:-10}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "$script_dir/../src" && pwd)
repo_dir=${PHASE1_REPO_DIR:-$(cd -- "$script_dir/../../.." && pwd)}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
json="$output_dir/phase1_2_${mode}_isl${isl}_${timestamp}.json"
telemetry="$output_dir/phase1_2_tegrastats_${mode}_isl${isl}_${timestamp}.log"
stdout_log="$output_dir/phase1_2_${mode}_isl${isl}_${timestamp}.log"

mkdir -p "$output_dir"
python "$source_dir/tegrastats_sampler.py" --output "$telemetry" --interval-ms 200 &
sampler_pid=$!

cleanup() {
    if kill -0 "$sampler_pid" 2>/dev/null; then
        kill "$sampler_pid"
        wait "$sampler_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

python "$source_dir/hf_bf16_benchmark.py" \
    --snapshot "$snapshot" \
    --output "$json" \
    --mode "$mode" \
    --isl "$isl" \
    --warmups "$warmups" \
    --trials "$trials" \
    --git-commit "$(git -C "$repo_dir" rev-parse HEAD)" \
    --branch "$(git -C "$repo_dir" branch --show-current)" 2>&1 | tee "$stdout_log"

printf 'json=%s\ntegrastats=%s\nstdout=%s\n' "$json" "$telemetry" "$stdout_log"
