# Phase 2.2-B4.1 — 28-Layer Oracle Memory Recovery

Date: 2026-09-02

## Why B4 stopped

The original B4 monolithic oracle and its predefined 7x4 fallback were both killed with exit code 137 before producing a handoff. Neither attempt reached ONNX or TensorRT. Static inspection found that both runners first materialized all 28 decoder-layer state dictionaries on CPU (880,932,864 BF16 bytes), then retained those tensors while loading transient CUDA layers and accumulating HF and portable reference trees. The final giant `torch.save` payload included the all-layer states and both reference trees, creating another high-risk serialization phase.

## Phase 1 contradiction and regression probe

Phase 1 had already loaded the exact frozen Qwen3-0.6B model on `cuda:0` and completed forward/generation, so B4 exit 137 could not be treated as proof of an intrinsic model limit. The B4.1 regression probe repeated only the bounded known-good operation: exact local revision, BF16, eager attention, one short two-token forward, no generation and no timing.

The probe passed. `MemAvailable` was 5,040,418,816 bytes before load, 4,054,503,424 after load and 3,576,070,144 after forward. Swap used remained 1,925,337,088 bytes. Torch allocated/reserved after load were 1,192,638,976/1,509,949,440 bytes and logits were finite on `cuda:0`. This rules out current inability to load the intrinsic model under the measured state.

## Streaming weight strategy and handoff

`streaming_extract.py` opens the safetensors checkpoint for one layer at a time, reads only the 11 required tensors, normalizes them to contiguous CPU tensors, saves one `layer_XX.pt`, computes tensor and file SHA256 values, releases all references and runs garbage collection before advancing. It never constructs an all-layer Python state dictionary.

All 28 independent files were produced under Jetson-local `/tmp/phase2_2b4_stream_20260902T070000Z/`. Total serialized size was 881,044,080 bytes; tensor payload was 880,932,864 bytes; all file hashes are present. `MemAvailable` moved from 5,062,651,904 to a recorded minimum of 4,995,117,056 bytes. The 67,534,848-byte system-level delta is an observation, not a process-exclusive peak-RSS claim. The `.pt` files remain outside Git.

## Recovered oracle architecture

The recovered oracle uses one Phase 1-validated `Qwen3ForCausalLM` load and directly invokes `model.model.layers` in order. It does not instantiate duplicate 28-layer HF or portable stacks. Only the persistent per-layer K/V required for decode is retained; selected hidden states 0, 3, 7, 15 and 27 are represented by small hashes in evidence rather than full histories. Every scale was run in a fresh Python process.

The 4-layer diagnostic passed the B3 execution contract: BF16 eager attention, `B=1,S=8` prefill, sequential layers, per-layer DynamicCache and one `8->9` decode. The 8-layer diagnostic passed the same lifecycle checks. Layer 0 and layer 3 hidden hashes were identical across the 4-, 8- and 28-layer processes. For both bounded diagnostics, torch reserved memory remained fixed at 1,509,949,440 bytes and swap was flat within the run.

The single permitted 28-layer recovery attempt passed. Prefill produced finite `[1,8,1024]` hidden state and all 28 caches. One decode step produced finite `[1,1,1024]` hidden state and all caches advanced to `[1,8,9,128]`. Minimum `MemAvailable` was 3,669,630,976 bytes; maximum torch allocated/reserved were 1,202,212,352/1,509,949,440 bytes. Across prefill layers 0 through 27, allocated growth was 884,736 bytes, exactly 27 times the 32,768-byte per-layer prefill K/V increment. Across decode layers it grew 110,592 bytes, exactly 27 times the 4,096-byte new-token K/V increment. Reserved memory and swap did not grow across layers.

## Root-cause conclusion

`IMPLEMENTATION_MEMORY_LIFETIME_CONFIRMED` is the most precise supported conclusion. The failed runners combined all-layer CPU state materialization, source/destination overlap during CUDA loads, retained reference trees and giant handoff construction. The same frozen model and the recovered single-model oracle complete 28-layer prefill and decode under current system conditions. This does not establish general capacity at longer sequence lengths and does not erase the original exit-137 evidence.

## Remaining resource risks

The measurement is limited to batch 1, prefill length 8 and one decode token. Unified CPU/GPU memory leaves less headroom at longer sequence lengths, during ONNX export or during TensorRT build. `MemAvailable` is system-wide rather than process-exclusive, and the extraction trace does not provide exact process peak RSS. TensorRT continuation therefore still requires a separately authorized, bounded plan.

## Gates

| Gate | Result |
| --- | --- |
| Phase 1 known-good full-model load and forward | PASS |
| Static duplication/lifetime audit | PASS |
| Streaming 28-file handoff | PASS |
| Recovered 4-layer memory diagnostic | PASS |
| Recovered 8-layer memory diagnostic | PASS |
| Recovered 28-layer prefill | PASS |
| Recovered 28-layer decode `8->9` | PASS |
| No exit 137 | PASS |

Phase 2.2-B4.1: **PASS / CLOSED**. Decision: `B4_ORACLE_MEMORY_PATH_RECOVERED`.

## Explicit stop

No 28-layer ONNX was exported. No 28-layer TensorRT engine was built. No INT8 or INT4 was performed. No benchmark or Nsight profiler was run. Phase 2.2-B4 TensorRT continuation, Phase 2.2-C, Phase 2.2-D, Phase 2.3 and Phase 3 have not started. The Phase 1 BF16 reference remains unchanged, and the pre-existing Jetson stash remains preserved.
