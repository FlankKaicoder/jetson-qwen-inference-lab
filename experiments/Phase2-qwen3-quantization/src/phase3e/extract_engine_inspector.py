from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
from collections import defaultdict
from pathlib import Path


LINEAR_RE = re.compile(
    r"/(?:layers_(\d+)/)?(?:self_attn\.)?(q_proj|k_proj|v_proj|o_proj|"
    r"gate_proj|up_proj|down_proj)(?:_(\d+))?/MatMul"
)
ATTENTION_RE = re.compile(r"/(?:layers_(\d+)/)?MatMul(?:_(\d+))?(?=/|$)")
GEMM_RE = re.compile(r"(xmma_gemm|h16816gemm|cutlass.*gemm)", re.I)
MYL_RE = re.compile(r"^__myl_")


def onnx_layers(metadata: str) -> list[str]:
    return re.findall(r"\[ONNX Layer: ([^\]]+)\]", metadata)


def tensor_strings(items: list[dict]) -> tuple[str, str]:
    shapes = []
    precisions = []
    for item in items:
        dims = item.get("Dimensions", [])
        shapes.append("x".join(str(value) for value in dims) or "SCALAR")
        precisions.append(str(item.get("Format/Datatype", "UNKNOWN")))
    return ";".join(shapes), ";".join(precisions)


def operator_for(onnx_names: list[str]) -> tuple[str, str]:
    for name in onnx_names:
        match = LINEAR_RE.search(name)
        if match:
            layer = match.group(3) or match.group(1) or 0
            return f"L{int(layer)}:{match.group(2)}", "LINEAR"
    for name in onnx_names:
        match = ATTENTION_RE.search(name)
        if match:
            layer = match.group(2) or match.group(1) or 0
            if name == "/MatMul" or name.endswith("/MatMul"):
                return f"L{int(layer)}:ATTENTION_SCORE", "ATTENTION_GEMM"
            return f"L{int(layer)}:ATTENTION_VALUE", "ATTENTION_GEMM"
    return "UNKNOWN", "UNKNOWN"


def classify(tactic: str) -> str:
    if not tactic:
        return "NO_KERNEL"
    if MYL_RE.match(tactic):
        return "TENSORRT_INTERNAL"
    if GEMM_RE.search(tactic):
        return "GEMM"
    if "_gemm_mha_" in tactic:
        return "ATTENTION_GEMM"
    return "TENSORRT_OTHER"


def effective_m(shape: str, assumed_m: int | None) -> str:
    parts = shape.split("x")
    values = [int(value) for value in parts if value.isdigit()]
    if len(parts) >= 2 and parts[-2] == "-1" and assumed_m is not None:
        return str(assumed_m)
    if values and len(parts) >= 2 and parts[-2] != "-1":
        return str(values[-2])
    return "UNKNOWN"


def linear_mnk(operator: str, inputs: str, outputs: str, assumed_m: int | None) -> tuple[str, str, str]:
    in_shapes = inputs.split(";")
    out_shapes = outputs.split(";")
    if operator == "UNKNOWN" or ":ATTENTION_" in operator or not in_shapes or not out_shapes:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"
    output = out_shapes[0]
    output_n = output.split("x")[-1]
    candidates = [
        shape for shape in in_shapes
        if len(shape.split("x")) == 3
        and shape.split("x")[-1] == output_n
        and "-" not in shape.split("x")
    ]
    weight = candidates[-1] if candidates else "UNKNOWN"
    weight_parts = weight.split("x")
    if len(weight_parts) != 3:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"
    m = effective_m(output, assumed_m)
    return m, weight_parts[-1], weight_parts[-2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", action="append", required=True, help="label=path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assume-m", action="append", default=[], help="label=M")
    parser.add_argument("--save-raw-inspector", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import tensorrt as trt

    assumed = {}
    for item in args.assume_m:
        label, value = item.split("=", 1)
        assumed[label] = int(value)

    layer_rows: list[dict[str, object]] = []
    gemm_rows: list[dict[str, object]] = []
    engine_summaries: dict[str, dict[str, object]] = {}

    for spec in args.engine:
        label, engine_path_text = spec.split("=", 1)
        engine_path = Path(engine_path_text)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        blob = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError(f"ENGINE_DESERIALIZE_FAILED:{engine_path}")
        info = engine.create_engine_inspector().get_engine_information(
            trt.LayerInformationFormat.JSON
        )
        if args.save_raw_inspector:
            (args.out / f"engine_inspector_{label}.json").write_text(info)
        payload = json.loads(info)

        counts = defaultdict(int)
        engine_gemm_rows: list[dict[str, object]] = []
        for index, original_layer in enumerate(payload["Layers"]):
            if isinstance(original_layer, str):
                layer = {"Name": original_layer}
                names = [original_layer] if not MYL_RE.match(original_layer) else []
                compact = True
            else:
                layer = original_layer
                names = onnx_layers(layer.get("Metadata", ""))
                compact = False
            tactic = layer.get("TacticName", "")
            operator, operator_kind = operator_for(names)
            if compact:
                if operator != "UNKNOWN":
                    tactic = "UNKNOWN_NO_DETAILED_INSPECTOR"
                    category = "GEMM"
                else:
                    tactic = "UNKNOWN_NO_DETAILED_INSPECTOR"
                    category = "UNKNOWN"
            else:
                category = classify(tactic)
            inputs, input_precisions = tensor_strings(layer.get("Inputs", []))
            outputs, output_precisions = tensor_strings(layer.get("Outputs", []))
            row = {
                "engine": label,
                "layer_index": index,
                "layer_name": layer.get("Name", ""),
                "layer_type": layer.get("LayerType", ""),
                "operator": operator,
                "operator_kind": operator_kind,
                "category": category,
                "tactic": tactic or "NONE",
                "input_shapes": inputs or "NONE",
                "input_precisions": input_precisions or "NONE",
                "output_shapes": outputs or "NONE",
                "output_precisions": output_precisions or "NONE",
                "onnx_layers": ";".join(names) or "UNKNOWN",
            }
            layer_rows.append(row)
            counts[category] += 1
            if category in ("GEMM", "ATTENTION_GEMM"):
                if compact:
                    precision = "FP16_ENGINE_NO_LAYER_TACTIC_DATA"
                else:
                    precision = "INT8" if "i8" in tactic.lower() else (
                        "FP16" if "f16" in tactic.lower() or "h16816" in tactic.lower() else "UNKNOWN"
                    )
                m, n, k = linear_mnk(operator, inputs, outputs, assumed.get(label))
                gemm_row = {
                    "engine": label,
                    "layer_index": index,
                    "operator": operator,
                    "operator_kind": operator_kind,
                    "matrix_m": m,
                    "matrix_n": n,
                    "matrix_k": k,
                    "precision": precision,
                    "tactic": tactic,
                    "input_shapes": inputs,
                    "output_shapes": outputs,
                    "onnx_layers": ";".join(names) or "UNKNOWN",
                    "mapping_evidence": "ENGINE_INSPECTOR_METADATA" if operator != "UNKNOWN" else "UNKNOWN",
                }
                gemm_rows.append(gemm_row)
                engine_gemm_rows.append(gemm_row)

        engine_summaries[label] = {
            "engine_path": str(engine_path),
            "engine_bytes": engine_path.stat().st_size,
            "engine_sha256": hashlib.sha256(blob).hexdigest(),
            "tensorrt": trt.__version__,
            "inspector_layer_count": len(payload["Layers"]),
            "category_counts": dict(counts),
        }
        del engine
        del runtime

    shape_groups: dict[tuple[str, str, str, str, str, str, str], dict[str, object]] = defaultdict(
        lambda: {"frequency": 0, "layer_indices": []}
    )
    for row in gemm_rows:
        key = (
            str(row["engine"]), str(row["operator"]), str(row["operator_kind"]),
            str(row["matrix_m"]), str(row["matrix_n"]), str(row["matrix_k"]),
            str(row["precision"]),
        )
        group = shape_groups[key]
        group["frequency"] += 1
        group["layer_indices"].append(row["layer_index"])
        group["tactic"] = row["tactic"]
        group["mapping_evidence"] = row["mapping_evidence"]
    shape_rows = []
    for key, value in shape_groups.items():
        shape_rows.append({
            "engine": key[0], "operator": key[1], "operator_kind": key[2],
            "matrix_m": key[3], "matrix_n": key[4], "matrix_k": key[5],
            "precision": key[6], "tactic": value["tactic"], "frequency": value["frequency"],
            "layer_indices": ";".join(str(x) for x in value["layer_indices"]),
            "mapping_evidence": value["mapping_evidence"],
        })

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            path.write_text("")
            return
        fields = list(rows[0])
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.out / "layer_inventory.csv", layer_rows)
    write_csv(args.out / "gemm_inventory.csv", gemm_rows)
    write_csv(args.out / "gemm_shape_summary.csv", shape_rows)
    (args.out / "inspector_summary.json").write_text(json.dumps({
        "phase": "Phase 3-E1/E2",
        "method": "TensorRT EngineInspector, DETAILED engine data, no build or execution",
        "host": platform.node(),
        "tensorrt": trt.__version__,
        "engines": engine_summaries,
        "limitations": [
            "Runtime kernel time is not attributable to individual inspector layers from this artifact alone.",
            "Operator identity uses Inspector ONNX metadata; absent metadata remains UNKNOWN.",
            "Dynamic dimensions are labeled UNKNOWN unless a benchmark M is explicitly supplied.",
        ],
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS", "engines": len(args.engine),
        "layer_rows": len(layer_rows), "gemm_rows": len(gemm_rows),
        "shape_rows": len(shape_rows),
    }))


if __name__ == "__main__":
    main()
