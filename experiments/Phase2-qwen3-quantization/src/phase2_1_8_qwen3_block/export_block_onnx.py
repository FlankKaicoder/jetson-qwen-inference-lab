import argparse
import json
from pathlib import Path
import torch
from qwen3_block import CONFIG, make_block


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config-out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    block = make_block()
    x = torch.randn((1, CONFIG["sequence_length"], CONFIG["hidden_size"]), device="cuda", dtype=torch.float16)
    torch.onnx.export(
        block, x, args.out, opset_version=17, do_constant_folding=True,
        input_names=["hidden_states"], output_names=["output"],
        dynamic_axes={"hidden_states": {0: "batch"}, "output": {0: "batch"}},
    )
    args.config_out.write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "EXPORT_PASS", "onnx": str(args.out), "config": str(args.config_out), "opset": 17}))


if __name__ == "__main__":
    main()
