from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    data = json.loads(args.path.read_text())
    names = [n["name"] for n in data["matmul_nodes"]]
    counts = collections.Counter(names)
    print(f"total matmul={len(names)} unique={len(counts)}")
    for name, cnt in sorted(counts.items()):
        print(f"{cnt:3d}  {name}")


if __name__ == "__main__":
    main()
