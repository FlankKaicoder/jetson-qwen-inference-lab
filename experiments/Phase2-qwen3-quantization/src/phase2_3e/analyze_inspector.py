from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    args = ap.parse_args()
    data = json.loads(args.summary.read_text())
    insp = data["engine_inspector"]
    for kind in ("prefill", "decode"):
        text = insp[kind]
        counts = collections.Counter()
        for m in re.finditer(r'"TacticName"\s*:\s*"([^"]+)"', text):
            counts[m.group(1)] += 1
        int8_tactics = {k: v for k, v in counts.items() if "i8" in k.lower() or "int8" in k.lower()}
        fp16_tactics = {k: v for k, v in counts.items() if "half" in k.lower() or "fp16" in k.lower() or "h884" in k.lower() or "h1688" in k.lower()}
        print(f"=== {kind} ===")
        print(f"total tactic entries={sum(counts.values())} distinct={len(counts)}")
        print(f"int8 tactic entries={sum(int8_tactics.values())} distinct={len(int8_tactics)}")
        print("int8 tactics:")
        for k, v in sorted(int8_tactics.items(), key=lambda kv: -kv[1]):
            print(f"  {v:4d}  {k}")
        print("fp16 tactics:")
        for k, v in sorted(fp16_tactics.items(), key=lambda kv: -kv[1]):
            print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
