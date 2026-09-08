# Phase 8.4-A Qwen3 End-to-End RMSNorm Plugin Impact Evaluation

Date: 2026-09-09 (Asia/Shanghai)

## Objective

Evaluate one real Qwen3 Layer 0 `input_layernorm` replacement in isolated
full 28-layer TensorRT FP16 Prefill and Decode graphs. The original ONNX,
checkpoint, and historical engines were not modified. The experiment replaces
1 of 113 RMSNorm nodes.

## Environment and method

- Jetson Orin Nano Super, CUDA 12.6.68, TensorRT 10.3.0, SM87.
- Source graphs: `/tmp/phase2_2b4_20260902T082326Z/prefill_28layer.onnx` and
  `decode_28layer.onnx`.
- Prefill input `[1,8,1024]`; Decode input `[1,1,1024]` with past KV length 8.
- FP16, seed `20260909`, 10 warmups and 30 CUDA Event repetitions.
- Baseline and plugin graphs/engines were separate files. The plugin uses the
  Phase 8.1 V2 CUDA kernel and runtime token count `B*S`.

## Build results

| Variant | Graph bytes | Engine bytes | Build seconds | Result |
| --- | ---: | ---: | ---: | --- |
| Prefill baseline | 881,567,467 | 887,464,492 | 108.780 | PASS |
| Prefill plugin | 881,566,253 | 887,446,972 | 104.791 | PASS |
| Decode baseline | 881,548,745 | 892,739,268 | 113.969 | PASS |
| Decode plugin | 881,547,531 | 892,527,348 | 127.930 | PASS |

TensorRT parser errors were empty for all four builds. The custom
`trt.plugins:RMSNormPlugin` node was accepted after adding the required ONNX
domain opset and preserving a graph-shared constant.

## Correctness

Layer 0 exposed output passed the bounded single-node reference of relative
L2 <= 1e-3: Prefill `0.0006377380`, Decode `0.0008134234` (max absolute error
`0.001953125` in both). Numerical differences accumulate through the full
stack: `hidden_l27` relative L2 was Prefill `0.0221897` and Decode `0.0143852`.
Therefore the full-model output gate is `INCONCLUSIVE`; this run does not
support an end-to-end numerical equivalence claim.

## Runtime results

| Mode | Baseline mean (ms) | Plugin mean (ms) | Change | Baseline throughput | Plugin throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill | 38.5183 | 39.6601 | +2.96% | 25.9617/s | 25.2142/s |
| Decode | 41.7803 | 42.6711 | +2.13% | 23.9347/s | 23.4351/s |

Observed CUDA free memory after runs was 4.777 GB baseline / 4.767 GB plugin
for Prefill and 4.644 GB baseline / 4.670 GB plugin for Decode. These snapshots
are process-end observations, not a peak allocation trace.

## Analysis and limitations

The isolated plugin is accepted in the complete 28-layer graph, but it does
not improve latency under this fixed-shape protocol. The small slowdown is
consistent with plugin launch/dispatch overhead and lack of TensorRT native
fusion, but the experiment does not isolate either cause. The accumulating
FP16 numerical difference means only the Layer 0 node contract is bounded;
model-level correctness remains `INCONCLUSIVE`. Only one RMSNorm was replaced,
so no claim is made for replacing all 113 nodes.

## Phase 8.4-A Gate

**BOUNDED / NO_END_TO_END_SPEEDUP**. Build and integration gates pass; the
plugin is not a demonstrated model-level optimization and full-model numerical
equivalence is inconclusive. Further work would require explicit authorization
for a separately designed accuracy/fusion investigation.

## Evidence

Compact raw JSON, logs, hashes, and run metadata are in
`experiments/Phase8-rmsnorm-optimization/artifacts/phase8_4A_20260909T/`.
Large ONNX and engine files remain only in the Jetson `/tmp` experiment area.
