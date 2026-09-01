#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 SNAPSHOT ARTIFACT_DIR" >&2
    exit 2
fi

snapshot=$1
artifact_dir=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd -- "$script_dir/../src" && pwd)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
functional="$artifact_dir/phase1_1_functional_${timestamp}.json"
tegrastats_log="$artifact_dir/phase1_1_tegrastats_${timestamp}.log"
stdout_log="$artifact_dir/phase1_1_reference_${timestamp}.log"

mkdir -p "$artifact_dir"
tegrastats --interval 200 > "$tegrastats_log" &
tegra_pid=$!

cleanup() {
    if kill -0 "$tegra_pid" 2>/dev/null; then
        kill "$tegra_pid"
        wait "$tegra_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

python "$source_dir/hf_bf16_reference.py" \
    --snapshot "$snapshot" \
    --output "$functional" 2>&1 | tee "$stdout_log"

printf 'functional=%s\ntegrastats=%s\nstdout=%s\n' \
    "$functional" "$tegrastats_log" "$stdout_log"
