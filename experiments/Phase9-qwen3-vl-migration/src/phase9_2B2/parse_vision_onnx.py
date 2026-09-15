#!/usr/bin/env python3
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import tensorrt as trt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CollectingLogger(trt.ILogger):
    def __init__(self):
        super().__init__()
        self.records = []

    def log(self, severity, message):
        try:
            severity_name = severity.name
        except AttributeError:
            severity_name = str(severity)
        record = {"severity": severity_name, "message": message}
        self.records.append(record)
        print(f"[TRT:{severity_name}] {message}", flush=True)


def enum_name(value):
    return getattr(value, "name", str(value))


def tensor_record(tensor):
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "dtype": enum_name(tensor.dtype),
    }


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--onnx-path", required=True)
    argument_parser.add_argument("--result-path", required=True)
    args = argument_parser.parse_args()

    onnx_path = Path(args.onnx_path).resolve()
    result_path = Path(args.result_path).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX graph not found: {onnx_path}")

    started_utc = datetime.now(timezone.utc).isoformat()
    logger = CollectingLogger()
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    onnx_parser = trt.OnnxParser(network, logger)

    parse_returned = False
    parse_exception = None
    try:
        parse_returned = bool(onnx_parser.parse_from_file(str(onnx_path)))
    except Exception as exc:
        parse_exception = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    parser_errors = []
    for index in range(onnx_parser.num_errors):
        error = onnx_parser.get_error(index)
        parser_errors.append(
            {
                "index": index,
                "code": enum_name(error.code),
                "description": error.desc,
                "file": error.file(),
                "line": error.line(),
                "function": error.func(),
                "node": error.node(),
            }
        )

    inputs = [tensor_record(network.get_input(index)) for index in range(network.num_inputs)]
    network_outputs = [
        tensor_record(network.get_output(index)) for index in range(network.num_outputs)
    ]

    layer_type_counts = Counter()
    layer_summary = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        layer_type = enum_name(layer.type)
        layer_type_counts[layer_type] += 1
        outputs = [
            tensor_record(layer.get_output(output_index))
            for output_index in range(layer.num_outputs)
        ]
        layer_summary.append(
            {
                "index": index,
                "name": layer.name,
                "type": layer_type,
                "num_inputs": layer.num_inputs,
                "num_outputs": layer.num_outputs,
            "outputs": network_outputs,
            }
        )

    severity_counts = Counter(record["severity"] for record in logger.records)
    result = {
        "phase": "Phase 9.2-B2",
        "audit_mode": "TENSORRT_ONNX_PARSER_INSPECTION_ONLY",
        "hostname": platform.node(),
        "tensorrt_version": trt.__version__,
        "onnx_file": {
            "path": str(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "sha256": sha256_file(onnx_path),
        },
        "parser": {
            "invoked": True,
            "invocation_method": "OnnxParser.parse_from_file",
            "parse_returned": parse_returned,
            "parse_exception": parse_exception,
            "num_errors": onnx_parser.num_errors,
            "errors": parser_errors,
            "success": parse_returned and onnx_parser.num_errors == 0,
        },
        "network": {
            "creation_flags": ["EXPLICIT_BATCH"],
            "num_inputs": network.num_inputs,
            "num_outputs": network.num_outputs,
            "num_layers": network.num_layers,
            "inputs": [tensor_record(network.get_input(index)) for index in range(network.num_inputs)],
            "outputs": [tensor_record(network.get_output(index)) for index in range(network.num_outputs)],
        },
        "layer_type_counts": dict(sorted(layer_type_counts.items())),
        "layer_summary": layer_summary,
        "logger": {
            "record_count": len(logger.records),
            "severity_counts": dict(sorted(severity_counts.items())),
            "records": logger.records,
        },
        "prohibited_operations_performed": {
            "builder_config_created": False,
            "engine_build_called": False,
            "engine_serialized": False,
            "benchmark_run": False,
            "tactic_selection_forced": False,
            "quantization_performed": False,
            "cuda_kernel_modified": False,
            "environment_modified": False,
        },
        "started_utc_hostname": started_utc,
    }

    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PHASE9_2B2_PARSER_AUDIT_DONE", flush=True)
    if not result["parser"]["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
