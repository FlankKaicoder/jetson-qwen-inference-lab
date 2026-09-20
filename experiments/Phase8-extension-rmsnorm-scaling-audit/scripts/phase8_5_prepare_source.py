#!/usr/bin/env python3
"""Reconstruct isolated 28-layer Qwen3 FP16 source graphs from safetensors."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
LAYER_KEYS = [
    "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight",
    "self_attn.k_norm.weight", "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size)), 8 + size


def load_tensor(path: Path, name: str) -> torch.Tensor:
    header, data_start = safetensors_header(path)
    if name not in header:
        raise KeyError(f"missing safetensors tensor: {name}")
    item = header[name]
    start, end = item["data_offsets"]
    raw = np.memmap(path, dtype=np.uint8, mode="r",
                    offset=data_start + start, shape=(end - start,))
    return torch.frombuffer(memoryview(raw), dtype=torch.uint8)


def load_layer_state(model_file: Path, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}."
    state = {}
    header, data_start = safetensors_header(model_file)
    for suffix in LAYER_KEYS:
        name = prefix + suffix
        if name not in header:
            raise KeyError(f"missing safetensors tensor: {name}")
        item = header[name]
        start, end = item["data_offsets"]
        raw = np.memmap(model_file, dtype=np.uint8, mode="r",
                        offset=data_start + start, shape=(end - start,))
        uint8 = torch.frombuffer(memoryview(raw), dtype=torch.uint8)
        state[suffix] = (uint8.view(torch.bfloat16)
                         .view(item["shape"]).to(device="cuda", dtype=torch.float16))
    return state


def load_head_weights(model_dir: Path) -> dict[str, torch.Tensor]:
    model_file = model_dir / "model.safetensors"
    wanted = ["model.embed_tokens.weight", "model.norm.weight"]
    header, data_start = safetensors_header(model_file)
    result = {}
    for name in wanted:
        if name not in header:
            raise KeyError(f"missing safetensors tensor: {name}")
        item = header[name]
        start, end = item["data_offsets"]
        raw = np.memmap(model_file, dtype=np.uint8, mode="r",
                        offset=data_start + start, shape=(end - start,))
        uint8 = torch.frombuffer(memoryview(raw), dtype=torch.uint8)
        result[name] = (uint8.view(torch.bfloat16)
                        .view(item["shape"]).to(device="cuda", dtype=torch.float16))
    return result


class PrefillExport(torch.nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, hidden, pos):
        hs, ks, vs = self.stack.forward_prefill(hidden, pos)
        return tuple(hs + ks + vs)


class DecodeExport(torch.nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, hidden, pos, *past):
        pks, pvs = list(past[:28]), list(past[28:])
        hs, ks, vs = self.stack.forward_decode(hidden, pos, pks, pvs)
        return tuple(hs + ks + vs)


def export_graph(stack, path: Path, kind: str, hidden, pos, past=None) -> dict:
    import onnx

    module = PrefillExport(stack) if kind == "prefill" else DecodeExport(stack)
    if kind == "prefill":
        inputs = (hidden, pos)
        input_names = ["hidden_states", "position_ids"]
        dynamic = {
            "hidden_states": {0: "batch", 1: "seq"},
            "position_ids": {0: "batch", 1: "seq"},
        }
    else:
        inputs = (hidden, pos, *past)
        input_names = ["hidden_states", "position_ids"]
        input_names += [f"past_k{i}" for i in range(28)]
        input_names += [f"past_v{i}" for i in range(28)]
        dynamic = {"hidden_states": {0: "batch"}, "position_ids": {0: "batch"}}
        for name in input_names[2:]:
            dynamic[name] = {0: "batch", 2: "past_len"}
    outputs = [f"hidden_l{i}" for i in range(28)]
    outputs += [f"present_k{i}" for i in range(28)]
    outputs += [f"present_v{i}" for i in range(28)]
    for name in outputs:
        dynamic[name] = ({0: "batch", 1: "seq"} if name.startswith("hidden_")
                         else {0: "batch", 2: "seq" if kind == "prefill" else "present_len"})
    torch.onnx.export(module, inputs, str(path), opset_version=17,
                      input_names=input_names, output_names=outputs,
                      dynamic_axes=dynamic)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    result = {
        "status": "PASS", "bytes": path.stat().st_size,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operators": dict(Counter(node.op_type for node in model.graph.node)),
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "sha256": sha256(path),
    }
    del model
    gc.collect()
    return result


def main() -> None:
    import onnx

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--portable-src", type=Path, required=True)
    parser.add_argument("--tmp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.portable_src))
    from portable_qwen3_28 import PortableTwentyEightLayerStack

    args.tmp.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    model_file = args.model_dir / "model.safetensors"
    model_sha = sha256(model_file)
    if model_sha != MODEL_SHA256:
        raise RuntimeError("FROZEN_MODEL_SHA256_MISMATCH")

    stack = PortableTwentyEightLayerStack(device="cuda", dtype=torch.float16).eval()
    for layer in range(28):
        state = load_layer_state(model_file, layer)
        stack.layers[layer].load_state_dict(state, strict=True)
        del state
        gc.collect()
        torch.cuda.empty_cache()

    torch.manual_seed(0)
    hidden = torch.randn((1, 8, 1024), device="cuda", dtype=torch.float16)
    pos = torch.arange(8, device="cuda", dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        _, ks, vs = stack.forward_prefill(hidden, pos)
    prefill_result = export_graph(stack, args.tmp / "prefill_28layer.onnx", "prefill", hidden, pos)
    decode_result = export_graph(stack, args.tmp / "decode_28layer.onnx", "decode", hidden[:, :1], pos[:, :1], ks + vs)

    stack_cpu = None
    del stack
    gc.collect()
    torch.cuda.empty_cache()
    summary = {
        "status": "PASS",
        "model": str(args.model_dir),
        "model_sha256": model_sha,
        "checkpoint_dtype": "bfloat16",
        "graph_dtype": "float16",
        "prefill": prefill_result,
        "decode": decode_result,
    }
    (args.out / "source_graph_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
