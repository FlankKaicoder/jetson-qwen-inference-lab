from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPTS = {
    "hello": "Hello",
    "en_8tok": "What is the capital of France?",
    "en_10tok": "Please tell me the capital city of France today.",
    "zh_short": "你好，请介绍一下你自己。",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    out = {}
    for name, text in PROMPTS.items():
        ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        out[name] = {"text": text, "input_ids": ids, "token_count": len(ids)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
