# Phase 8.5 RMSNorm Scaling Audit Handoff (2026-09-20)

## Recovery facts

- Branch: `phase/08-extension-fusion-analysis`
- Starting HEAD: `c4c702cc8accd10010133c9439c67f354d7d51e0`
- Main experiment commit: `7cd5c7e99fa871f8c57cdf06fa4ab4b7bbec59b7`
  (`phase8-ext: add RMSNorm scaling audit`)
- Gate:
  `PASS / BOUNDED / NO_PROVEN_SCALABLE_PLUGIN_PATH`
- Primary report:
  `experiments/Phase8-extension-rmsnorm-scaling-audit/docs/phase8_5_rmsnorm_scaling_audit_report.md`
- Compact artifacts:
  `experiments/Phase8-extension-rmsnorm-scaling-audit/artifacts/phase8_5_20260920T075327Z/`
- Raw Jetson ONNX, engines, and `.nsys-rep` files:
  `/tmp/phase8_5_20260920T075327Z/` (Jetson-local, intentionally not committed)

Recover state from Git, the primary report, and the timestamped artifacts first.
This handoff is auxiliary and is committed after the evidence commit.

## Evidence map

- Replaced-norm error:
  `summary_replaced_norm_relative_l2.csv`
- Per-layer final propagation:
  `summary_hidden_relative_l2.csv`
- Numerical gate summary:
  `summary_numerical.csv`
- Generation consistency:
  `summary_generation.csv`
- Performance:
  `summary_performance.csv`
- Nsight attribution:
  `summary_nsys.csv`

All paths above are relative to the compact artifacts directory.

## Main findings

- The audit covered scales `0/1/4/8/28/56` for both prefill and decode on
  reconstructed isolated FP16 28-layer Qwen3 graphs. It used the unchanged
  Phase 8.4-A RMSNorm plugin.
- All compared outputs were finite, and final `hidden_l27` relative-L2 stayed
  below `0.05` for every scale.
- The predeclared replaced-RMSNorm tolerance failed from scale 4:
  prefill maxima were `0.0031608739`, `0.0072243474`, `0.0221709777`, and
  `0.0232615210` at scales `4/8/28/56`; decode maxima were `0.0026417261`,
  `0.0046953182`, `0.0138255768`, and `0.0148195559`.
- Generation consistency failed at prefill scale 1 and every scale 4 or larger.
  Decode scale 1 passed. Final hidden tolerance therefore did not imply
  generation consistency.
- CUDA-event mean latency ranged from `1244.775277` to `2231.409619` ms for
  prefill and `2007.754590` to `2058.818229` ms for decode. Kernel count stayed
  near `5.1k` for prefill and `5.4k-5.5k` for decode; it did not monotonically
  scale with replaced-node count or explain the larger prefill scale 28/56
  event latency.
- Decode scale 56 used a separately marked `opt1` engine after OOM-killed
  default/low-memory retries. The failed attempts and the original `opt0`
  evidence were preserved and are not silently replaced.

## Gate recommendation

Stop after Phase 8.5. Do not begin further scaling, fusion, optimization,
deployment, or plugin/runtime changes without a fresh explicit owner gate.
Any later attempt must treat `NO_PROVEN_SCALABLE_PLUGIN_PATH` as the standing
bounded result until new evidence changes it.
