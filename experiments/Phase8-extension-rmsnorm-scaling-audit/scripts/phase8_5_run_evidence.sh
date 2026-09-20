#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?mode is required: compare|measure|generation|profile}
PY=${PY:-python}
PLUGIN_SO=${PLUGIN_SO:?PLUGIN_SO is required}
MODEL_DIR=${MODEL_DIR:?MODEL_DIR is required}

target_outputs() {
  local kind=$1
  local scale=$2
  if [[ "${scale}" == "0" ]]; then
    echo ""
  else
    "$PY" - "$kind" "$scale" <<'PYEOF'
import json
import sys
kind, scale = sys.argv[1], int(sys.argv[2])
data = json.load(open(f"rewritten/rewrite_{kind}_scale{scale}.json"))
print(",".join(row["target_output"] for row in data["plugin"]["replacements"]))
PYEOF
  fi
}

for kind in prefill decode; do
  for scale in 0 1 4 8 28 56; do
    targets=$(target_outputs "${kind}" "${scale}")
    engine="engines/${kind}_baseline.engine"
    plugin_args=()
    case "${MODE}" in
      compare) out="artifacts_local/evidence/compare_${kind}_scale${scale}.json" ;;
      measure) out="artifacts_local/evidence/measure_${kind}_scale${scale}.json" ;;
      generation) out="artifacts_local/evidence/generation_${kind}_scale${scale}.json" ;;
      profile) out="artifacts_local/evidence/profile_event_${kind}_scale${scale}.json" ;;
    esac
    if [[ -f "${out}" ]]; then
      echo "${MODE} ${kind} scale=${scale} skipped: output exists"
      continue
    fi
    if [[ "${scale}" -gt 0 ]]; then
      engine="engines/${kind}_plugin_scale${scale}.engine"
      plugin_args+=(--plugin-engine "${engine}" --plugin-so "${PLUGIN_SO}")
    fi
    case "${MODE}" in
      compare)
        "$PY" scripts/phase8_5_audit.py \
          --mode compare --kind "${kind}" --scale "${scale}" \
          --baseline-engine "engines/${kind}_baseline.engine" \
          "${plugin_args[@]}" \
          --target-outputs "${targets}" \
          --out "artifacts_local/evidence/compare_${kind}_scale${scale}.json" \
          > "artifacts_local/evidence/compare_${kind}_scale${scale}.log" 2>&1
        ;;
      measure)
        "$PY" scripts/phase8_5_audit.py \
          --mode measure --kind "${kind}" --scale "${scale}" \
          --engine "${engine}" "${plugin_args[@]}" \
          --warmup 10 --repetitions 30 \
          --out "${out}" \
          > "artifacts_local/evidence/measure_${kind}_scale${scale}.log" 2>&1
        ;;
      generation)
        "$PY" scripts/phase8_5_audit.py \
          --mode generation --kind "${kind}" --scale "${scale}" \
          --baseline-engine "engines/${kind}_baseline.engine" \
          --baseline-prefill-engine "engines/prefill_baseline.engine" \
          --baseline-decode-engine "engines/decode_baseline.engine" \
          --plugin-prefill-engine "engines/prefill_plugin_scale${scale}.engine" \
          --plugin-decode-engine "engines/decode_plugin_scale${scale}.engine" \
          "${plugin_args[@]}" \
          --model-dir "${MODEL_DIR}" \
          --out "${out}" \
          > "artifacts_local/evidence/generation_${kind}_scale${scale}.log" 2>&1
        ;;
      profile)
        nsys profile \
          --trace cuda,nvtx --sample=none --cpuctxsw=none \
          --output "artifacts_local/profile_${kind}_scale${scale}" \
          "$PY" scripts/phase8_5_audit.py \
            --mode profile --kind "${kind}" --scale "${scale}" \
            --engine "${engine}" "${plugin_args[@]}" \
            --profile-warmup 1 --profile-runs 3 \
            --out "${out}" \
            > "artifacts_local/evidence/profile_event_${kind}_scale${scale}.log" 2>&1
        ;;
      *)
        echo "unsupported mode: ${MODE}" >&2
        exit 2
        ;;
    esac
    echo "${MODE} ${kind} scale=${scale} done"
  done
done
