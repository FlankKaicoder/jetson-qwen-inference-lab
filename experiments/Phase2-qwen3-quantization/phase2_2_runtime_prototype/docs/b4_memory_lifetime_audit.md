# B4.1 Memory Lifetime Audit

## Failed B4 lifecycle

The B4 runners first enumerate and materialize all 28 layers into `states` using `safetensors.safe_open().get_tensor()`. This leaves roughly 880,932,864 BF16 bytes alive on CPU before execution. Each iteration then creates a HF or portable layer on CUDA and calls `load_state_dict`, so a source CPU tensor and a CUDA parameter copy overlap during loading. The CPU state dictionary remains alive for the whole run.

The HF path retains CPU copies of every per-layer hidden/K/V output for prefill and four decode steps. The portable path retains another set and attention outputs. Finally, `torch.save` receives a nested object containing all layer states plus references, creating a large serialization/materialization phase. The fallback reduces simultaneous CUDA layer modules but still retains the all-layer CPU `states` dictionary and full reference trees.

## Lifetime table

| Stage | CPU checkpoint tensors | CUDA weights | BF16/FP16 duplicate | References retained |
| --- | --- | --- | --- | --- |
| B4 state extraction | all 28 layers | none | no | `states` and manifest |
| HF prefill/decode | all 28 layers | one HF layer at a time | CPU source overlaps CUDA load | all per-layer CPU outputs |
| Portable prefill/decode | all 28 layers | one portable layer at a time | CPU source overlaps CUDA load | portable hidden/K/V/attention trees |
| handoff construction | all 28 layers | transient layer/cache allocations | serialized nested copy risk | HF + portable references |

## Root-cause conclusion

The failure is not evidence that the intrinsic Qwen3 model cannot fit: Phase 1 loaded the exact full model successfully with a recorded model-load allocator delta of 1,192,638,976 bytes and minimum MemAvailable of 2,746,023,936 bytes. B4 instead combines all-layer CPU materialization, repeated CUDA copies, retained reference trees and handoff construction. Before dynamic recovery, the evidence supported `ROOT_CAUSE_PARTIALLY_LOCALIZED`; a true streaming design had to read, use, hash, and release one layer at a time and never construct a giant nested handoff.

## B4.1 dynamic confirmation

The Phase 1 known-good exact-model load and short forward passed. A 28-file streaming extraction then completed with only one layer state live at a time, followed by fresh-process 4-layer, 8-layer and 28-layer oracle runs. The 28-layer run completed prefill and one `8->9` decode with flat allocator reservation and only expected KV growth. This dynamic contrast confirms `IMPLEMENTATION_MEMORY_LIFETIME_CONFIRMED`; it does not invalidate or overwrite the original exit-137 evidence.
