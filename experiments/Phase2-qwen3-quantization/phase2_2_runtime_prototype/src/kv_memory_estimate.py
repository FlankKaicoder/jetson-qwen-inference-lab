"""Emit theoretical Qwen3 KV-cache storage, without allocating a model or cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
DTYPE_SIZE = 2  # FP16/BF16
TOKEN_COUNTS = (1024, 4096, 8192, 32768, 40960)


def estimate_bytes(batch_size: int, tokens: int) -> int:
    return 2 * LAYERS * KV_HEADS * HEAD_DIM * DTYPE_SIZE * batch_size * tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "THEORETICAL_ESTIMATE",
        "formula": "2 * layers * kv_heads * head_dim * dtype_size * batch_size * tokens",
        "config": {
            "layers": LAYERS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "dtype_size": DTYPE_SIZE,
            "dtypes": ["fp16", "bf16"],
        },
        "bytes_per_token_batch_1": estimate_bytes(1, 1),
        "estimates": {
            f"batch_{batch}": {str(tokens): estimate_bytes(batch, tokens) for tokens in TOKEN_COUNTS}
            for batch in (1, 2)
        },
        "limitation": "KV cache only; excludes engine weights, activations, workspace, metadata and allocator overhead. This is not a runtime capacity claim.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
