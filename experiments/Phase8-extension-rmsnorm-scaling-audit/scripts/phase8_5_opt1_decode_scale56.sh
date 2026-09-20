#!/usr/bin/env bash
set -euo pipefail

PY=${PY:?PY is required}
PLUGIN_SO=${PLUGIN_SO:?PLUGIN_SO is required}

TARGETS=$("$PY" - <<'PYEOF'
import json
data = json.load(open("rewritten/rewrite_decode_scale56.json"))
print(",".join(row["target_output"] for row in data["plugin"]["replacements"]))
PYEOF
)

"$PY" scripts/phase8_5_audit.py \
  --mode compare --kind decode --scale 56 \
  --baseline-engine engines/decode_baseline.engine \
  --plugin-engine engines/decode_plugin_scale56_opt1.engine \
  --plugin-so "${PLUGIN_SO}" \
  --target-outputs "${TARGETS}" \
  --out artifacts_local/evidence/compare_decode_scale56_opt1_attempt2.json \
  > artifacts_local/evidence/compare_decode_scale56_opt1_attempt2.log 2>&1

echo "OPT1_COMPARE_ATTEMPT2_PASS"
