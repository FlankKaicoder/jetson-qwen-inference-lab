#!/usr/bin/env python3
"""Build one FP16 TensorRT engine for the Phase 8.5 audit."""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import tensorrt as trt


def build(onnx_path: Path, engine_path: Path, plugin_so: Path | None, kind: str,
          workspace_bytes: int | None = None, tactic_dram_bytes: int | None = None,
          builder_optimization_level: int | None = None) -> dict:
    if plugin_so:
        global PLUGIN_HANDLE
        PLUGIN_HANDLE = ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    plugin_creator_check = None
    if plugin_so:
        plugin_creator_check = bool(
            trt.get_plugin_registry().get_creator("RMSNormPlugin", "1", ""))
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    started = time.time()
    ok = parser.parse(onnx_path.read_bytes())
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not ok:
        raise RuntimeError(f"parser failed {kind}: {errors}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    if workspace_bytes is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if tactic_dram_bytes is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, tactic_dram_bytes)
    if builder_optimization_level is not None:
        config.builder_optimization_level = builder_optimization_level
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
    return {
        "status": "PASS", "kind": kind, "onnx": str(onnx_path),
        "engine": str(engine_path), "onnx_bytes": onnx_path.stat().st_size,
        "engine_bytes": engine_path.stat().st_size,
        "build_seconds": time.time() - started, "parser_errors": errors,
        "plugin_creator_check": plugin_creator_check,
        "workspace_bytes": workspace_bytes,
        "tactic_dram_bytes": tactic_dram_bytes,
        "builder_optimization_level": builder_optimization_level,
    }


PLUGIN_HANDLE = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--kind", choices=["prefill", "decode"], required=True)
    parser.add_argument("--plugin-so", type=Path)
    parser.add_argument("--workspace-bytes", type=int)
    parser.add_argument("--tactic-dram-bytes", type=int)
    parser.add_argument("--builder-optimization-level", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.onnx, args.engine, args.plugin_so, args.kind,
                   args.workspace_bytes, args.tactic_dram_bytes,
                   args.builder_optimization_level)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
