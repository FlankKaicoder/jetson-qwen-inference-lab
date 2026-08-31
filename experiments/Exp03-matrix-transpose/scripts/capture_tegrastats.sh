#!/usr/bin/env bash
set -euo pipefail
output="$1"
phase="$2"
dimension="$3"
version="$4"
trial="$5"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
line=""
if command -v tegrastats >/dev/null 2>&1; then
  line="$(timeout 1s tegrastats --interval 100 2>/dev/null | head -n 1 || true)"
fi
python3 "${script_dir}/parse_tegrastats.py" "$line" "$output" "$phase" "$dimension" "$version" "$trial"
