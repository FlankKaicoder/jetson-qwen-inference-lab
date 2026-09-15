#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import onnx


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value_info):
    shape = value_info.type.tensor_type.shape
    dims = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    initializer_counts = Counter(init.data_type for init in graph.initializer)

    checker_result = "PASS"
    checker_error = None
    try:
        onnx.checker.check_model(str(onnx_path), full_check=False)
    except Exception as exc:
        checker_result = "FAIL"
        checker_error = {"type": type(exc).__name__, "message": str(exc)}

    result = {
        "phase": "Phase 9.2-B1",
        "audit_mode": "ONNX_GRAPH_INSPECTION_ONLY",
        "onnx_file": {
            "path": str(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "sha256": sha256_file(onnx_path),
        },
        "onnx_model": {
            "onnx_version": onnx.__version__,
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "opset_import": [
                {"domain": entry.domain, "version": entry.version} for entry in model.opset_import
            ],
        },
        "checker": {"result": checker_result, "error": checker_error},
        "graph": {
            "node_count": len(graph.node),
            "initializer_count": len(graph.initializer),
            "input_count": len(graph.input),
            "output_count": len(graph.output),
            "value_info_count": len(graph.value_info),
            "sparse_initializer_count": len(graph.sparse_initializer),
            "function_count": len(model.functions),
        },
        "inputs": [
            {"name": value.name, "shape": tensor_shape(value), "dtype": value.type.tensor_type.elem_type}
            for value in graph.input
        ],
        "outputs": [
            {"name": value.name, "shape": tensor_shape(value), "dtype": value.type.tensor_type.elem_type}
            for value in graph.output
        ],
        "operator_counts": dict(sorted(op_counts.items())),
        "initializer_dtype_counts": {str(key): value for key, value in initializer_counts.items()},
        "dynamic_shape_indicators": [],
        "tensorrt_parser_attempted": False,
        "prohibited_operations_performed": {
            "tensorrt_build": False,
            "benchmark": False,
            "quantization": False,
            "optimization": False,
            "cuda_modification": False,
            "environment_modification": False,
        },
    }

    for value in list(graph.input) + list(graph.output):
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") and dim.dim_param:
                result["dynamic_shape_indicators"].append(
                    {"name": value.name, "dim_param": dim.dim_param}
                )
    result["dynamic_shape_indicators"] = list(
        {json.dumps(item, sort_keys=True) for item in result["dynamic_shape_indicators"]}
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
