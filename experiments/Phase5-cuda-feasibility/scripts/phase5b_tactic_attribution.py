#!/usr/bin/env python3
"""Build Phase 5-B tactic attribution from frozen Phase 4/5-A evidence."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


OPERATOR_PATTERN = re.compile(r"^L(\d+):up_proj$")
LAYER_ID_PATTERN = re.compile(r"_myl0_(\d+)$")
INPUT_SHAPE = "[1,1,1024]"
WEIGHT_SHAPE = "[1,1024,3072]"
OUTPUT_SHAPE = "[1,1,3072]"

CSV_FIELDS = [
    "decoder_layer",
    "trt_layer_id",
    "trt_layer_name",
    "trt_layer_type",
    "onnx_node",
    "operator",
    "attribution_confidence",
    "input_shape",
    "input_precision",
    "weight_shape",
    "weight_precision",
    "output_shape",
    "output_precision",
    "alpha_beta_precision",
    "alpha_beta_values",
    "inspector_implementation_evidence",
    "tactic_name",
    "tactic_id",
    "runtime_kernel_name_counts",
    "kernel_name",
    "kernel_family",
    "tensor_core_usage_tactic_label",
    "tensor_core_usage_direct_measurement",
    "workspace",
    "cuBLASLt_algorithm",
    "cuBLASLt_workspace_bytes",
    "backend_identity",
    "comparison_status",
    "evidence",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_text(tensor):
    precision = tensor.get("Format/Datatype", "UNKNOWN")
    dimensions = tensor.get("Dimensions")
    shape = "x".join(str(value) for value in dimensions) if dimensions else "UNKNOWN"
    return f"{shape}|{precision}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspector", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--runtime-kernel-csv", required=True, type=Path)
    parser.add_argument("--csv-out", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    with args.inspector.open("r", encoding="utf-8") as stream:
        inspector = json.load(stream)
    inspector_by_id = {
        int(LAYER_ID_PATTERN.search(layer["Name"]).group(1)): layer
        for layer in inspector["Layers"]
    }

    with args.mapping.open("r", encoding="utf-8", newline="") as stream:
        mappings = list(csv.DictReader(stream))
    up_proj_rows = {}
    for row in mappings:
        operator = row.get("candidate_operator", "")
        match = OPERATOR_PATTERN.fullmatch(operator)
        if not match:
            continue
        decoder_layer = int(match.group(1))
        layer_id = int(row["layer_id"])
        if layer_id in up_proj_rows:
            raise RuntimeError(f"Duplicate up_proj mapping for layer {decoder_layer}")
        up_proj_rows[decoder_layer] = row

    if len(up_proj_rows) != 28:
        raise RuntimeError(f"Expected 28 up_proj mappings, got {len(up_proj_rows)}")

    runtime_by_layer = {}
    runtime_observations = 0
    with args.runtime_kernel_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            layer_id = int(row["layer_index"])
            runtime_by_layer.setdefault(layer_id, []).append(row)
            runtime_observations += 1

    if runtime_observations != 196 or len(runtime_by_layer) != 28:
        raise RuntimeError(
            "Expected 196 runtime observations across 28 layers, got "
            f"{runtime_observations} across {len(runtime_by_layer)}"
        )

    output_rows = []
    summaries = {}
    for decoder_layer in sorted(up_proj_rows):
        mapping = up_proj_rows[decoder_layer]
        layer_id = int(mapping["layer_id"])
        inspector_layer = inspector_by_id[layer_id]
        inputs = inspector_layer.get("Inputs", [])
        outputs = inspector_layer.get("Outputs", [])

        runtime_counts = Counter()
        runtime_name_counts = Counter()
        for runtime_row in runtime_by_layer[decoder_layer]:
            runtime_counts[runtime_row["kernel_family"]] += 1
            runtime_name_counts[runtime_row["kernel_name"]] += 1

        kernel_name_summary = ";".join(
            f"{name}={count}" for name, count in sorted(runtime_name_counts.items())
        )
        family_summary = ";".join(
            f"{family}={count}" for family, count in sorted(runtime_counts.items())
        )
        tactic = inspector_layer.get("TacticName", "UNKNOWN")
        summaries[decoder_layer] = {
            "trt_layer_id": layer_id,
            "trt_layer_name": inspector_layer.get("Name", "UNKNOWN"),
            "tactic_name": tactic,
            "runtime_kernel_name_counts": kernel_name_summary,
            "runtime_kernel_family_counts": family_summary,
        }
        output_rows.append({
            "decoder_layer": decoder_layer,
            "trt_layer_id": layer_id,
            "trt_layer_name": inspector_layer.get("Name", "UNKNOWN"),
            "trt_layer_type": inspector_layer.get("LayerType", "UNKNOWN"),
            "onnx_node": mapping["candidate_onnx_node"],
            "operator": "up_proj",
            "attribution_confidence": mapping["confidence"],
            "input_shape": INPUT_SHAPE,
            "input_precision": tensor_text(inputs[0]) if inputs else "UNKNOWN",
            "weight_shape": WEIGHT_SHAPE,
            "weight_precision": tensor_text(inputs[1]) if len(inputs) > 1 else "UNKNOWN",
            "output_shape": OUTPUT_SHAPE,
            "output_precision": tensor_text(outputs[0]) if outputs else "UNKNOWN",
            "alpha_beta_precision": (
                "Float;Float"
                if len(inputs) >= 4
                and inputs[2].get("Format/Datatype") == "Float"
                and inputs[3].get("Format/Datatype") == "Float"
                else "UNKNOWN"
            ),
            "alpha_beta_values": "UNKNOWN",
            "inspector_implementation_evidence": "TACTIC_NAME_ONLY",
            "tactic_name": tactic,
            "tactic_id": "NOT_AVAILABLE",
            "runtime_kernel_name_counts": kernel_name_summary,
            "kernel_name": "OBSERVED_MULTIPLE_BY_INVOCATION",
            "kernel_family": family_summary,
            "tensor_core_usage_tactic_label": (
                "TENSOR16X8X16_LABEL_PRESENT"
                if "tensor16x8x16" in tactic
                else "UNKNOWN"
            ),
            "tensor_core_usage_direct_measurement": (
                "NOT_AVAILABLE_FOR_ALL_28_ROWS; PHASE_3E_SELECTED_KERNEL_SAMPLES_ONLY"
            ),
            "workspace": "NOT_AVAILABLE",
            "cuBLASLt_algorithm": "21",
            "cuBLASLt_workspace_bytes": "0",
            "backend_identity": "UNKNOWN",
            "comparison_status": "NOT_PAIRED_BENCHMARK; BOUNDARY_AND_CONTEXT_MISMATCH",
            "evidence": (
                "PHASE4A_INSPECTOR_METADATA+ONNX_NODE_MATCH; "
                "PHASE5A_RUNTIME_KERNEL_BREAKDOWN"
            ),
        })

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    tactic_counts = Counter(row["tactic_name"] for row in output_rows)
    kernel_family_counts = Counter()
    for row in output_rows:
        for family_count in row["kernel_family"].split(";"):
            family, count = family_count.rsplit("=", 1)
            kernel_family_counts[family] += int(count)

    analysis = {
        "phase": "Phase5B Step1 TensorRT GEMM Path Investigation",
        "mode": "READ_ONLY_ANALYSIS_OF_FROZEN_ARTIFACTS",
        "coverage": {
            "up_proj_layers": len(output_rows),
            "runtime_observations": runtime_observations,
            "runtime_observations_per_layer": 7,
        },
        "source_files": {
            "inspector": {
                "path": str(args.inspector),
                "sha256": sha256(args.inspector),
            },
            "mapping": {
                "path": str(args.mapping),
                "sha256": sha256(args.mapping),
            },
            "runtime_kernel_csv": {
                "path": str(args.runtime_kernel_csv),
                "sha256": sha256(args.runtime_kernel_csv),
            },
        },
        "tactic_name_counts": dict(sorted(tactic_counts.items())),
        "runtime_kernel_family_counts": dict(sorted(kernel_family_counts.items())),
        "per_layer": summaries,
        "not_available_fields": [
            "numeric_tactic_id",
            "runtime_workspace",
            "alpha_beta_values",
            "direct_tensor_core_measurement_for_all_28_rows",
        ],
        "unknown_fields": [
            "tensorrt_backend_identity",
            "accumulator_dtype",
            "physical_layout_semantics",
            "tactic_to_runtime_kernel_family_mapping",
        ],
    }
    with args.json_out.open("w", encoding="utf-8") as stream:
        json.dump(analysis, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
