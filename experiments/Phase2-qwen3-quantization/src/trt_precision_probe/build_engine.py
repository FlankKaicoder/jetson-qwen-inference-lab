"""Build synthetic direct TensorRT FP16 and explicit-Q/DQ INT8 Linear engines."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.WARNING)


def constant(network, values, name):
    layer = network.add_constant(values.shape, trt.Weights(values))
    layer.name = name
    return layer.get_output(0)


def build_engine(out_features, qdq, engine_path):
    builder = trt.Builder(LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    if qdq:
        config.set_flag(trt.BuilderFlag.INT8)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 1024), (32, 1024), (32, 1024))
    config.add_optimization_profile(profile)

    rng = np.random.default_rng(0)
    x = network.add_input("input", trt.float16, (-1, 1024))
    weight = rng.standard_normal((1024, out_features)).astype(np.float16) * np.float16(0.02)
    weight_tensor = constant(network, weight, "weight")
    if qdq:
        scale = constant(network, np.array([0.02], dtype=np.float32), "scale")
        xq = network.add_quantize(x, scale)
        xq.name = "input_quantize"
        xdq = network.add_dequantize(xq.get_output(0), scale)
        xdq.name = "input_dequantize"
        wq = network.add_quantize(weight_tensor, scale)
        wq.name = "weight_quantize"
        wdq = network.add_dequantize(wq.get_output(0), scale)
        wdq.name = "weight_dequantize"
        left, right = xdq.get_output(0), wdq.get_output(0)
    else:
        left, right = x, weight_tensor
    matmul = network.add_matrix_multiply(left, trt.MatrixOperation.NONE, right, trt.MatrixOperation.NONE)
    matmul.name = "linear_matmul"
    output = matmul.get_output(0)
    output.name = "output"
    network.mark_output(output)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized engine")
    Path(engine_path).write_bytes(bytes(serialized))
    return weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-features", required=True, type=int, choices=[1024, 2048, 3072])
    parser.add_argument("--qdq", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        weight = build_engine(args.out_features, args.qdq, args.out)
        print(json.dumps({
            "status": "BUILD_PASS",
            "engine": str(args.out),
            "engine_bytes": args.out.stat().st_size,
            "out_features": args.out_features,
            "qdq": args.qdq,
            "weight_shape": list(weight.shape),
        }, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "BUILD_FAIL", "error": repr(exc), "qdq": args.qdq}, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
