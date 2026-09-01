"""Deterministic CPU-only validation for the Phase 2.2 cache ownership contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kv_cache_manager import KVCacheConfig, KVCacheManager
from runtime_interface import DecodeRequest, KVCacheState, PrefillRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Small synthetic dimensions make position semantics testable without a model-sized allocation.
    manager = KVCacheManager(KVCacheConfig(num_layers=2, num_kv_heads=2, head_dim=4))
    manager.allocate_cache(batch_size=1, capacity_tokens=4)
    payload_bytes = 1 * 2 * 2 * 4 * 2  # [B=1,H=2,T=2,D=4] FP16/BF16 byte size
    key = bytes(range(payload_bytes))
    value = bytes((index + 17) % 256 for index in range(payload_bytes))

    layer0_end = manager.append_kv(0, key, value, token_count=2, start_position=0)
    visible_after_one_layer = manager.sequence_length
    layer1_end = manager.append_kv(1, key, value, token_count=2, start_position=0)
    visible_after_all_layers = manager.sequence_length
    cache = manager.get_layer_cache(0)
    element_bytes = 4 * 2
    reconstructed = bytearray()
    for head in range(2):
        base = (head * manager.capacity_tokens) * element_bytes
        reconstructed.extend(cache.key[base : base + 2 * element_bytes])
    first_key_bytes_match = bytes(reconstructed) == key

    state = KVCacheState(manager=manager, sequence_id="synthetic-sequence")
    state.validate_prefill(PrefillRequest(input_ids=((10, 11),)))
    decode_position = state.validate_decode(DecodeRequest(input_ids=((12,),)))
    manager.reset_sequence()

    result = {
        "status": "PASS",
        "scope": "CPU-only synthetic byte-layout validation; no CUDA, TensorRT or Qwen3 checkpoint",
        "allocation": {
            "batch_size": manager.batch_size,
            "capacity_tokens": manager.capacity_tokens,
            "bytes_allocated": manager.bytes_allocated,
        },
        "append": {
            "layer0_end": layer0_end,
            "layer1_end": layer1_end,
            "visible_after_one_layer": visible_after_one_layer,
            "visible_after_all_layers": visible_after_all_layers,
            "first_key_bytes_match": first_key_bytes_match,
        },
        "interface": {"decode_position_before_reset": decode_position},
        "reset": {"sequence_length_after_reset": manager.sequence_length},
        "pass_conditions": {
            "partial_layer_not_visible": visible_after_one_layer == 0,
            "all_layers_visible": visible_after_all_layers == 2,
            "payload_round_trip": first_key_bytes_match,
            "decode_uses_visible_position": decode_position == 2,
            "reset_invalidates_logical_sequence": manager.sequence_length == 0,
        },
    }
    result["status"] = "PASS" if all(result["pass_conditions"].values()) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
