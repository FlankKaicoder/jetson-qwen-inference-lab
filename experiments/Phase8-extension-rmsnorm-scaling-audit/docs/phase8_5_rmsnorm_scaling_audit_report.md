# Phase 8.5 RMSNorm Scaling and Numerical Stability Audit

Date: 2026-09-20  
Branch: `phase/08-extension-fusion-analysis`  
Starting HEAD: `c4c702cc8accd10010133c9439c67f354d7d51e0`  
Gate: `PASS / BOUNDED / NO_PROVEN_SCALABLE_PLUGIN_PATH`

## Result

The predeclared numerical gate failed. The plugin output stayed finite and the
final `hidden_l27` relative-L2 remained below `0.05`, but the maximum replaced
RMSNorm relative-L2 exceeded `1e-3` from scale 4 upward. Generation consistency
also failed before all scales passed. Therefore there is no proven scalable
plugin path in this bounded experiment. This is an attribution result, not an
optimization or deployment conclusion.

## Objective and boundary

The audit replaced only selected Qwen3 RMSNorm subgraphs with the unchanged
Phase 8.4-A RMSNorm TensorRT plugin and measured numerical error propagation,
CUDA-event latency, deterministic graph generation, and Nsight Systems
attribution. No new kernel, fusion, plugin modification, or optimization was
performed. Existing Phase 8 files and artifacts were not changed.

## Environment

| Item | Value |
| --- | --- |
| Device | Jetson Orin Nano Super (`nvidia-desktop`) |
| Driver | 540.4.0 |
| TensorRT | 10.3.0 |
| NVIDIA PyTorch | 2.5.0a0+872d972e41.nv24.08 |
| CUDA stack | 12.6 series |
| Nsight Systems | 2024.5.4.34 |
| Transformers used for generation | 4.57.3 |
| Model | Pinned Qwen3 checkpoint, `model.safetensors` SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Isolated Jetson workspace | `/tmp/phase8_5_20260920T075327Z/` |

## Engine source

The historical Jetson `/tmp` ONNX/engine path was absent, so two new FP16
28-layer source graphs were reconstructed in the isolated workspace from the
pinned Qwen3 checkpoint. The frozen checkpoint hash was verified before graph
construction. Each source graph was rewritten once per scale for plugin use and
once for an output-exposure baseline. The plugin source was copied unchanged,
compiled successfully, and registered as `RMSNormPlugin`.

Scales:

| Scale | Replaced nodes |
| ---: | --- |
| 0 | None; deterministic baseline self-check |
| 1 | Layer 0 `input_layernorm` |
| 4 | Layers 0-3 `input_layernorm` |
| 8 | Layers 0-7 `input_layernorm` |
| 28 | Layers 0-27 `input_layernorm` |
| 56 | Layers 0-27 `input_layernorm` and `post_attention_layernorm` |

Q/K RMSNorm and the final model RMSNorm were not replaced. TensorRT 10.3.0
built 11/12 required engines with the normal builder configuration. The first
`decode_plugin_scale56` build and a 1 GiB low-memory retry were killed by the
Jetson OOM killer; both failures remain in logs. A separate
`decode_plugin_scale56_opt1.engine` was then built with 1 GiB workspace/tactic
pools and `builder_optimization_level=0`. It was measured separately and is
marked `opt1` in the summary. The original failed and successful attempts were
not deleted.

## Predeclared gate

For every nonzero scale:

1. all compared outputs must be finite;
2. every replaced RMSNorm output must have relative-L2 <= `1e-3`;
3. `hidden_l27` must have relative-L2 <= `0.05`;
4. all 8 generated tokens must match the baseline.

## Numerical results

All compared outputs were finite and all `hidden_l27` checks passed, but the
replaced-RMSNorm gate failed from scale 4 upward:

| Kind | Scale | Replaced-norm max rel-L2 | Final hidden rel-L2 | Final hidden max abs | Norm gate | Hidden gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Prefill | 0 | 0.0000000000 | 0.0000000000 | 0.0000 | PASS | PASS |
| Prefill | 1 | 0.0005070585 | 0.0250545554 | 5.6875 | PASS | PASS |
| Prefill | 4 | 0.0031608739 | 0.0256094895 | 7.9375 | FAIL | PASS |
| Prefill | 8 | 0.0072243474 | 0.0250120796 | 6.3750 | FAIL | PASS |
| Prefill | 28 | 0.0221709777 | 0.0255153142 | 7.5625 | FAIL | PASS |
| Prefill | 56 | 0.0232615210 | 0.0267501809 | 6.3750 | FAIL | PASS |
| Decode | 0 | 0.0000000000 | 0.0000000000 | 0.0000 | PASS | PASS |
| Decode | 1 | 0.0005001227 | 0.0164169539 | 1.59375 | PASS | PASS |
| Decode | 4 | 0.0026417261 | 0.0151888235 | 1.43262 | FAIL | PASS |
| Decode | 8 | 0.0046953182 | 0.0154183619 | 1.46875 | FAIL | PASS |
| Decode | 28 | 0.0138255768 | 0.0153234769 | 1.21875 | FAIL | PASS |
| Decode | 56 | 0.0148195559 | 0.0171932895 | 1.43750 | FAIL | PASS |

Every per-layer hidden relative-L2 and max-abs value is recorded in
[summary_hidden_relative_l2.csv](../artifacts/phase8_5_20260920T075327Z/summary_hidden_relative_l2.csv).
Every replaced-norm output metric is recorded in
[summary_replaced_norm_relative_l2.csv](../artifacts/phase8_5_20260920T075327Z/summary_replaced_norm_relative_l2.csv).
The compact gate summary is
[summary_numerical.csv](../artifacts/phase8_5_20260920T075327Z/summary_numerical.csv).

## Generation consistency

Generation used fixed greedy argmax over a final-position RMSNorm and embedding
head. Prefill generation reran the no-KV prefill graph as the token sequence
grew. Decode generation used the corresponding prefill engine once to create
initial KV, then the corresponding decode engine for eight steps. Baseline and
plugin paths were kept in the same graph family and precision.

| Kind | Scale | Match |
| --- | ---: | --- |
| Prefill | 0 | PASS |
| Prefill | 1 | FAIL |
| Prefill | 4 | FAIL |
| Prefill | 8 | FAIL |
| Prefill | 28 | FAIL |
| Prefill | 56 | FAIL |
| Decode | 0 | PASS |
| Decode | 1 | PASS |
| Decode | 4 | FAIL |
| Decode | 8 | FAIL |
| Decode | 28 | FAIL |
| Decode | 56 | FAIL |

The exact sequences are in
[summary_generation.csv](../artifacts/phase8_5_20260920T075327Z/summary_generation.csv).
Scale 0 was a baseline self-check. The prefill scale-1 mismatch plus the
decode scale-4 mismatch independently falsify the generation condition, even
though the final hidden state remained under its predeclared tolerance.

## Performance and kernel attribution

Each scale used 10 CUDA-event warmups and 30 measured executions. Nsight
Systems profiled one warmup and three measured executions. The decode scale-56
row uses the separately built `opt1` engine; its raw `opt0` attempt is preserved
but not used as the comparable result.

| Kind | Scale | Mean latency (ms) | Min/max (ms) | Kernels (3 profile runs) | GPU kernel time (ms, 3 runs) | CUDA launches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 0 | 1248.762337 | 1240.109 / 1257.382 | 5111 | 468.322432 | 1822 |
| Prefill | 1 | 1244.775277 | 1234.475 / 1255.769 | 5111 | 473.817248 | 1835 |
| Prefill | 4 | 1523.931197 | 1515.445 / 1532.042 | 5111 | 475.804576 | 1146 |
| Prefill | 8 | 1421.027425 | 1414.719 / 1427.135 | 5215 | 464.437408 | 1926 |
| Prefill | 28 | 2231.409619 | 2226.856 / 2238.498 | 5098 | 470.656576 | 1458 |
| Prefill | 56 | 2226.714632 | 2218.643 / 2238.958 | 5098 | 494.151168 | 1822 |
| Decode | 0 | 2033.236080 | 2021.772 / 2050.329 | 5504 | 501.425120 | 70 |
| Decode | 1 | 2056.906649 | 2042.202 / 2074.067 | 5504 | 496.061600 | 83 |
| Decode | 4 | 2029.839380 | 2015.662 / 2049.689 | 5478 | 503.310272 | 200 |
| Decode | 8 | 2058.818229 | 2049.587 / 2075.558 | 5426 | 493.824448 | 356 |
| Decode | 28 | 2238.059489 | 2230.962 / 2248.749 | 5153 | 490.380160 | 1136 |
| Decode | 56 | 2007.754590 | 2001.633 / 2013.401 | 5517 | 496.992736 | 1864 |

The compact CSV is
[summary_performance.csv](../artifacts/phase8_5_20260920T075327Z/summary_performance.csv);
the Nsight summary is
[summary_nsys.csv](../artifacts/phase8_5_20260920T075327Z/summary_nsys.csv).

Three attribution observations are supported:

1. Plugin scale changes do not cause a large monotonic kernel-count change.
   Kernel count remains near 5.1k for prefill and 5.4-5.5k for decode across
   all scales.
2. Kernel count does not explain the larger scale-28/56 prefill CUDA-event
   latency. Kernel count slightly decreases while mean latency increases.
3. CUDA launch counts vary non-monotonically and cannot be interpreted as a
   launch-overhead cause from these aggregate API sums. Exact per-kernel launch
   origin and tactic difference remain `UNKNOWN`.

No performance optimization is claimed. The plugin path is slower at multiple
scales, and the isolated workspace/builder differences prevent a deployment
causal claim.

## Conclusion

`NO_PROVEN_SCALABLE_PLUGIN_PATH`.

The plugin remains bounded to one node by the predeclared evidence. Multi-node
replacement does not preserve the required numerical contract and loses
generation consistency. The final hidden-state tolerance hides this failure;
therefore hidden-state tolerance alone must not be used to declare scale
success. No Phase 8.5 follow-up is authorized without a new explicit gate.

## Evidence map

- Compact evidence: `artifacts/phase8_5_20260920T075327Z/remote/artifacts_local/`
- Numerical summaries: `artifacts/phase8_5_20260920T075327Z/summary_*.csv`
- Scripts: `scripts/`
- Raw ONNX, TensorRT engines, and raw `.nsys-rep` files remain Jetson-local under
  `/tmp/phase8_5_20260920T075327Z/` and are intentionally not committed.
