#!/usr/bin/env python3
"""Build and measure isolated Qwen3 28-layer baseline/plugin ONNX copies on Jetson."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch
import tensorrt as trt


def mem(stage: str) -> dict[str, object]:
    free, total = torch.cuda.mem_get_info()
    return {"stage": stage, "cuda_free_bytes": free, "cuda_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def build(onnx_path: Path, engine_path: Path, plugin_so: Path | None, kind: str) -> dict[str, object]:
    if plugin_so:
        ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    started = time.time()
    ok = parser.parse(onnx_path.read_bytes())
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not ok:
        raise RuntimeError(f"parser failed {kind}: {errors}")
    config = builder.create_builder_config(); config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    if kind == "prefill":
        profile.set_shape("hidden_states", (1, 1, 1024), (1, 8, 1024), (1, 16, 1024))
        profile.set_shape("position_ids", (1, 1), (1, 8), (1, 16))
    else:
        profile.set_shape("hidden_states", (1, 1, 1024), (1, 1, 1024), (1, 1, 1024))
        profile.set_shape("position_ids", (1, 1), (1, 1), (1, 1))
        for i in range(28):
            for prefix in ("past_k", "past_v"):
                profile.set_shape(f"{prefix}{i}", (1, 8, 1, 128), (1, 8, 8, 128), (1, 8, 16, 128))
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"build failed {kind}")
    engine_path.write_bytes(bytes(blob))
    return {"status": "PASS", "kind": kind, "onnx_bytes": onnx_path.stat().st_size,
            "engine_bytes": engine_path.stat().st_size, "build_seconds": time.time() - started,
            "parser_errors": errors}


class Runner:
    def __init__(self, path: Path, plugin_so: Path | None):
        if plugin_so:
            ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None: raise RuntimeError(f"deserialize failed: {path}")
        self.stream = torch.cuda.current_stream()

    def run(self, inputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        ctx = self.engine.create_execution_context()
        for name, tensor in inputs.items():
            if not ctx.set_input_shape(name, tuple(tensor.shape)): raise RuntimeError(f"shape rejected {name}")
            ctx.set_tensor_address(name, tensor.data_ptr())
        outputs: dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name)); dtype = self.engine.get_tensor_dtype(name)
                if dtype != trt.DataType.HALF: raise RuntimeError(f"output {name} dtype {dtype}")
                outputs[name] = torch.empty(shape, device="cuda", dtype=torch.float16)
                ctx.set_tensor_address(name, outputs[name].data_ptr())
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record(self.stream)
        if not ctx.execute_async_v3(self.stream.cuda_stream): raise RuntimeError("execute failed")
        end.record(self.stream); end.synchronize()
        return outputs, float(begin.elapsed_time(end))


def inputs(kind: str, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn((1, 8 if kind == "prefill" else 1, 1024), device="cuda", dtype=torch.float16)
    p = torch.arange(8, device="cuda", dtype=torch.long).unsqueeze(0) if kind == "prefill" else torch.tensor([[8]], device="cuda", dtype=torch.long)
    out = {"hidden_states": x, "position_ids": p}
    if kind == "decode":
        for i in range(28):
            out[f"past_k{i}"] = torch.zeros((1, 8, 8, 128), device="cuda", dtype=torch.float16)
            out[f"past_v{i}"] = torch.zeros((1, 8, 8, 128), device="cuda", dtype=torch.float16)
    return out


def measure(path: Path, plugin_so: Path | None, kind: str, reps: int = 30) -> dict[str, object]:
    run = Runner(path, plugin_so); inp = inputs(kind, 20260909)
    for _ in range(10): run.run(inp)
    samples = [run.run(inp)[1] for _ in range(reps)]
    outputs, _ = run.run(inp)
    return {"engine": str(path), "kind": kind, "warmup": 10, "repetitions": reps,
            "mean_ms": sum(samples) / len(samples),
            "min_ms": min(samples), "max_ms": max(samples),
            "throughput_items_per_s": 1000.0 / (sum(samples) / len(samples)),
            "output_names": sorted(outputs), "memory": mem(f"after_{kind}")}


def compare_engines(baseline: Path, plugin: Path, plugin_so: Path, kind: str) -> dict[str, object]:
    base = Runner(baseline, None)
    candidate = Runner(plugin, plugin_so)
    inp = inputs(kind, 20260909)
    for _ in range(3):
        base.run(inp)
        candidate.run(inp)
    base_outputs, _ = base.run(inp)
    plugin_outputs, _ = candidate.run(inp)
    common = sorted(set(base_outputs).intersection(plugin_outputs))
    metrics = {}
    for name in common:
        a = base_outputs[name].float()
        b = plugin_outputs[name].float()
        delta = b - a
        metrics[name] = {
            "relative_l2": float(torch.linalg.vector_norm(delta) /
                                  torch.clamp(torch.linalg.vector_norm(a), min=1.0e-30)),
            "max_abs": float(delta.abs().max()),
        }
    return {"kind": kind, "baseline": str(baseline), "plugin": str(plugin),
            "common_outputs": common, "metrics": metrics}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["build", "measure", "compare"], required=True)
    p.add_argument("--kind", choices=["prefill", "decode"], required=True)
    p.add_argument("--onnx", type=Path); p.add_argument("--engine", type=Path)
    p.add_argument("--plugin-so", type=Path); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline-engine", type=Path); p.add_argument("--plugin-engine", type=Path)
    a = p.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "build": result = build(a.onnx, a.engine, a.plugin_so, a.kind)
    elif a.mode == "measure": result = measure(a.engine, a.plugin_so, a.kind)
    else:
        result = compare_engines(a.baseline_engine, a.plugin_engine, a.plugin_so, a.kind)
    a.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
