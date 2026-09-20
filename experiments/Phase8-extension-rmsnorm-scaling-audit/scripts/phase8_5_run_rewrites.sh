#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
for kind in prefill decode; do
  for scale in 1 4 8 28 56; do
    echo "rewrite ${kind} scale=${scale}"
    "$PY" scripts/phase8_5_rewrite_onnx.py \
      --source "work/${kind}_28layer.onnx" \
      --output "rewritten/${kind}_plugin_scale${scale}.onnx" \
      --kind "${kind}" \
      --scale "${scale}" \
      --summary "rewritten/rewrite_${kind}_scale${scale}.json" \
      > "rewritten/rewrite_${kind}_scale${scale}.log" 2>&1
  done
done
