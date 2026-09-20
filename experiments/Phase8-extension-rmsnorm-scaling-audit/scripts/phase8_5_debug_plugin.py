#!/usr/bin/env python3
"""Check plugin registration without building an engine."""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import tensorrt as trt


parser = argparse.ArgumentParser()
parser.add_argument("--plugin-so", required=True)
args = parser.parse_args()
handle = ctypes.CDLL(args.plugin_so, mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(None, "")
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
registry = trt.get_plugin_registry()
print("get_creator", registry.get_creator("RMSNormPlugin", "1", ""))
for creator in registry.plugin_creator_list:
    if "RMS" in creator.name or "rms" in creator.name:
        print(creator.name, creator.plugin_version, creator.plugin_namespace)
print("handle", handle)
print("builder", builder)
print("loaded_libraries")
for line in Path("/proc/self/maps").read_text().splitlines():
    if "nvinfer" in line or "tensorrt" in line:
        print(line)
