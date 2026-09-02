# Phase 2.2-B4.2 — Real Qwen3 28-Layer TensorRT FP16 Decoder Stack

## Scope

This bounded feasibility experiment evaluates the real Qwen3-0.6B decoder layers 0–27 with FP16 TensorRT prefill and dynamic decode graphs. Embedding, final RMSNorm, LM head, sampling, token generation, benchmark, Nsight, and quantization are explicitly out of scope.

The implementation consumes the Phase 2.2-B4.1 streaming handoff (`layer_00.pt` … `layer_27.pt`) and loads each file into its layer, releasing the temporary state dictionary before proceeding. No monolithic HF checkpoint payload is retained.

## Architecture

The primary design is one 28-layer prefill engine and one 28-layer decode engine. Each layer carries independent GQA K/V tensors with layout `[B, 8, L, 128]`; decode inputs contain the 28 past-K tensors followed by 28 past-V tensors. A fixed seven-partition (4 layers each) fallback is permitted by the experiment specification only after a primary resource failure; it is not attempted automatically.

## Gates

| Gate | Definition |
| --- | --- |
| B4.2-1 | 28-layer mapping and streaming handoff hashes pass |
| B4.2-2 | FP16 opset-17 prefill/decode ONNX export and checker pass |
| B4.2-3 | TensorRT FP16 parser/build pass without resource failure |
| B4.2-4 | B=1,S=8 prefill: finite CUDA hidden/K/V for all 28 layers |
| B4.2-5 | Decode 8→9→10→11→12 advances all 28 caches |
| B4.2-6 | Per-layer cache isolation and prefix invariants pass |
| B4.2-7 | Layer 0/3/7/15/27 numerical propagation is acceptable or explicitly reviewed |
| B4.2-8 | Memory/build feasibility is PASS, RISK, or BLOCKED with raw snapshots |

## Evidence policy

All claims require raw JSON/CSV/log evidence under the timestamped B4.2 artifact directory. Exit 137, OOM, or device instability is recorded as a blocker and is not retried automatically. ONNX and engine binaries remain outside Git.

## Explicit stop conditions

No embedding integration, final RMSNorm, LM head, sampling runtime, full token generation, benchmark, Nsight, INT8, INT4, TensorRT-LLM, Phase 2.2-C, Phase 2.3, or Phase 3 work is performed here.

## Results

Frozen identity checks passed for `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`; checkpoint SHA256 is `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. All 28 handoff files and their tensor hashes matched the B4.1 manifest. Raw FP16 decoder tensor bytes are 880,932,864.

The primary `ONE_28_LAYER_STACK_ENGINE` architecture succeeded; the 7x4 fallback was not used. Both opset-17 graphs passed ONNX checker:

| Graph | Bytes | Nodes | Initializers |
| --- | ---: | ---: | ---: |
| Prefill | 881,567,403 | 6,371 | 308 |
| Decode | 881,548,681 | 6,146 | 308 |

TensorRT 10.3 parsed both graphs with zero parser errors and built FP16 engines of 892,483,860 bytes (prefill) and 890,363,012 bytes (decode). Filesystem timestamps provide a coarse 227-second end-to-end build estimate (116 seconds to prefill completion and a further 110 seconds to decode completion); this includes parser/build and shell overhead and is not a benchmark.

Prefill `B=1,S=8,H=1024` produced finite CUDA hidden outputs and independent `[1,8,8,128]` K/V tensors for all 28 layers. Four dynamic decode calls advanced every cache `8→9→10→11→12`. Every old K/V prefix remained bitwise identical, and each step retained 28 unique K pointers plus 28 unique V pointers.

Numerical propagation was compared against the source-faithful portable FP16 graph at layers 0, 3, 7, 15 and 27. Across the recorded selected hidden/K/V comparisons, the maximum relative-L2 was approximately 0.03 (Layer 27 decode hidden); all values were finite and the minimum cosine exceeded the predeclared 0.99 bound. The decision is `ACCEPTABLE_FOR_FULL_FP16_RUNTIME_STEP`. This is a bounded decoder execution criterion, not a full-model accuracy or generation-quality claim.

Measured runtime memory snapshots remained stable: minimum `MemAvailable` was 4,122,578,944 bytes, maximum process RSS was approximately 1,441,280 KiB, process swap stayed 0, and system swap usage stayed 1,938,653,184 bytes. The theoretical batch-1 FP16 KV footprint is 114,688 bytes/token (917,504 bytes at L=8; 1,376,256 bytes at L=12). No runtime capacity claim is made.

TensorRT retained its warnings about Int64 `position_ids`, DLA profile fallback, FP16 layernorm Reduce/Pow overflow risk, and use of the current default CUDA stream. Correctness remained finite and within the predefined bound, but these warnings remain engineering risks for a later runtime stage.

## Gate Decision

| Gate | Result |
| --- | --- |
| B4.2-1 mapping | PASS |
| B4.2-2 ONNX export/checker | PASS |
| B4.2-3 TensorRT parse/build | PASS |
| B4.2-4 prefill | PASS |
| B4.2-5 dynamic decode | PASS |
| B4.2-6 KV integrity | PASS |
| B4.2-7 numerical propagation | PASS — `ACCEPTABLE_FOR_FULL_FP16_RUNTIME_STEP` |
| B4.2-8 resource feasibility | PASS for this bounded B=1,S<=12 run |

Overall: `PASS / BOUNDED`. Decision: `REAL_28_LAYER_TRT_DECODER_STACK_FEASIBLE`.

Evidence is in `artifacts/phase2_2b4_2_20260902T082326Z/`. ONNX and engine binaries are intentionally excluded from Git and retained only under Jetson `/tmp/phase2_2b4_2_20260902T082326Z/`.
