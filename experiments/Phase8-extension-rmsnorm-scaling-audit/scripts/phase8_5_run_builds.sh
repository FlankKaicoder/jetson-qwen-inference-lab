#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
PLUGIN_SO=${PLUGIN_SO:?PLUGIN_SO is required}

build_engine() {
  local onnx=$1
  local engine=$2
  local kind=$3
  local name=$4
  local plugin_arg=""
  if [[ -n "${PLUGIN_SO}" && "${name}" == *plugin* ]]; then
    plugin_arg="--plugin-so ${PLUGIN_SO}"
  fi
  if [[ -f "${engine}" ]]; then
    echo "skip existing ${engine}"
    return
  fi
  echo "build ${name}"
  "$PY" scripts/phase8_5_build_engine.py \
    --onnx "${onnx}" \
    --engine "${engine}" \
    --kind "${kind}" \
    ${plugin_arg} \
    --out "artifacts_local/build_${name}.json" \
    > "artifacts_local/build_${name}.log" 2>&1
}

build_engine rewritten/prefill_baseline_scale56.onnx engines/prefill_baseline.engine prefill baseline_prefill
build_engine rewritten/decode_baseline_scale56.onnx engines/decode_baseline.engine decode baseline_decode

for kind in prefill decode; do
  for scale in 1 4 8 28 56; do
    build_engine \
      "rewritten/${kind}_plugin_scale${scale}.onnx" \
      "engines/${kind}_plugin_scale${scale}.engine" \
      "${kind}" \
      "${kind}_plugin_scale${scale}"
  done
done
