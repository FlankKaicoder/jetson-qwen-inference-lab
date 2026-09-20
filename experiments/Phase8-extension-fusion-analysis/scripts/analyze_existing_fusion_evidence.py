#!/usr/bin/env python3
"""Build a compact Phase 8.7 audit from existing repository evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


INSPECTOR_JSON = Path(
    "results/phase3e_kernel_attribution/20260904T121007Z/inspector_v2/"
    "inspector_summary.json"
)
INSPECTOR_CSV = Path(
    "results/phase3e_kernel_attribution/20260904T121007Z/inspector_v2/"
    "layer_inventory.csv"
)
NSYS_RANGE_CSV = Path(
    "results/phase3a_runtime_attribution/20260904T063649Z_nsys/analysis/"
    "a2_range_summary.csv"
)
NSYS_CATEGORY_CSV = Path(
    "results/phase3a_runtime_attribution/20260904T063649Z_nsys/analysis/"
    "a3_kernel_attribution.csv"
)
NSYS_FP16_KERNELS = Path(
    "results/phase3c_residual_runtime/20260904T093500Z_nsys/stats/"
    "fp16_persistent_cuda_gpu_kern_sum.csv"
)
NSYS_MIXED_KERNELS = Path(
    "results/phase3c_residual_runtime/20260904T093500Z_nsys/stats/"
    "mixed_persistent_cuda_gpu_kern_sum.csv"
)
PLUGIN_ROOT = Path("experiments/Phase8-rmsnorm-optimization/artifacts/phase8_4A_20260909T/artifacts")

PROJECTION_RE = re.compile(
    r"/(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"(?:_\d+)?/(MatMul|MatMul__a_quant|MatMul__a_dequant|MatMul__w_dequant)"
)
NORM_RE = re.compile(
    r"/(input_layernorm|post_attention_layernorm|q_norm|k_norm)(?:_\d+)?/"
)
ATTENTION_MATMUL_RE = re.compile(r"^/MatMul_\d+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def projection_family(reference: str) -> str | None:
    match = PROJECTION_RE.search(reference)
    return match.group(1) if match else None


def norm_families(references: list[str]) -> set[str]:
    return {
        match.group(1)
        for reference in references
        if (match := NORM_RE.search(reference))
    }


def classify_fusion(references: list[str]) -> tuple[str, str]:
    projections = {
        family
        for reference in references
        if (family := projection_family(reference))
    }
    norms = norm_families(references)
    if "input_layernorm" in norms and {"q_proj", "k_proj", "v_proj"} <= projections:
        return "INPUT_RMSNORM_FUSED_WITH_QKV_QUANT", "INSPECTOR_ONNX_METADATA"
    if "post_attention_layernorm" in norms and "gate_proj" in projections:
        return "POST_RMSNORM_FUSED_WITH_GATE_QUANT", "INSPECTOR_ONNX_METADATA"
    if {"q_proj", "k_proj", "v_proj"} <= projections and all(
        PROJECTION_RE.search(ref).group(2) == "MatMul"
        for ref in references
        if PROJECTION_RE.search(ref)
    ):
        return "QKV_PROJECTION_FUSED", "INSPECTOR_ONNX_METADATA"
    if "o_proj" in projections and any(ref.startswith("/Add_") for ref in references):
        return "O_PROJ_WITH_RESIDUAL_FUSED", "INSPECTOR_ONNX_METADATA"
    if "down_proj" in projections and any(ref.startswith("/Add_") for ref in references):
        return "DOWN_PROJ_WITH_RESIDUAL_FUSED", "INSPECTOR_ONNX_METADATA"
    if "gate_proj" in projections and any("Sigmoid" in ref for ref in references):
        return "GATE_PROJ_WITH_SWIGLU_FUSED", "INSPECTOR_ONNX_METADATA"
    if any(ATTENTION_MATMUL_RE.fullmatch(ref) for ref in references):
        return "ATTENTION_MATMUL", "INSPECTOR_ONNX_METADATA"
    if norms and not projections:
        if any("Transpose" in ref or "Slice" in ref for ref in references):
            return "QK_NORM_WITH_ROPE_DATA_MOVEMENT", "INSPECTOR_ONNX_METADATA"
        return "NORM_ONLY_INTERNAL_KERNEL", "INSPECTOR_ONNX_METADATA"
    if norms and projections:
        return "NORM_WITH_PROJECTION", "INSPECTOR_ONNX_METADATA"
    if not references or references == ["UNKNOWN"]:
        return "METADATA_ABSENT", "NO_METADATA"
    return "OTHER_METADATA_MAPPED", "INSPECTOR_ONNX_METADATA"


def inspector_analysis(out_dir: Path) -> dict:
    raw_summary = read_json(INSPECTOR_CSV.parent / "inspector_summary.json")
    rows = list(csv.DictReader(INSPECTOR_CSV.open(encoding="utf-8-sig")))

    pattern_rows: list[dict] = []
    norm_rows: list[dict] = []
    examples: dict[tuple[str, str], dict] = {}
    engines: dict[str, dict] = {}

    for engine in sorted({row["engine"] for row in rows}):
        engine_rows = [row for row in rows if row["engine"] == engine]
        layer_types = Counter(row["layer_type"] or "COMPACT_NO_TYPE" for row in engine_rows)
        categories = Counter(row["category"] for row in engine_rows)
        patterns = Counter()
        pattern_evidence: dict[str, str] = {}

        norm_counts: dict[tuple[str, str, str], int] = Counter()
        for row in engine_rows:
            references = row["onnx_layers"].split(";") if row["onnx_layers"] else []
            pattern, evidence = classify_fusion(references)
            patterns[pattern] += 1
            pattern_evidence[pattern] = evidence
            key = (engine, pattern)
            if key not in examples:
                examples[key] = {
                    "engine": engine,
                    "pattern": pattern,
                    "layer_index": int(row["layer_index"]),
                    "layer_name": row["layer_name"],
                    "onnx_layers": row["onnx_layers"][:1000],
                    "layer_type": row["layer_type"] or "COMPACT_NO_TYPE",
                    "category": row["category"],
                    "evidence": evidence,
                }

            for family in sorted(norm_families(references)):
                has_projection = any(projection_family(ref) for ref in references)
                has_quant = any("MatMul__a_quant" in ref for ref in references)
                norm_counts[(family, "QUANT_GEMM" if has_quant and has_projection else ("GEMM" if has_projection else "NO_PROJECTION"), row["layer_type"] or "COMPACT_NO_TYPE")] += 1

        engines[engine] = {
            "inspector_layer_count": raw_summary["engines"][engine]["inspector_layer_count"],
            "engine_bytes": raw_summary["engines"][engine]["engine_bytes"],
            "engine_sha256": raw_summary["engines"][engine]["engine_sha256"],
            "layer_type_counts": dict(sorted(layer_types.items())),
            "category_counts": dict(sorted(categories.items())),
        }
        for pattern, count in patterns.most_common():
            pattern_rows.append({
                "engine": engine,
                "fusion_pattern": pattern,
                "layer_count": count,
                "evidence": pattern_evidence[pattern],
            })
        for (family, projection_state, layer_type), count in sorted(norm_counts.items()):
            norm_rows.append({
                "engine": engine,
                "norm_family": family,
                "projection_state": projection_state,
                "trt_layer_type": layer_type,
                "layer_count": count,
            })

    write_csv(out_dir / "fusion_pattern_summary.csv", pattern_rows)
    write_csv(out_dir / "norm_fusion_summary.csv", norm_rows)
    write_csv(
        out_dir / "fusion_pattern_examples.csv",
        [value for _, value in sorted(examples.items())],
    )
    return {
        "method": "read-only extraction of existing TensorRT EngineInspector layer inventory",
        "tensorrt": raw_summary["tensorrt"],
        "host": raw_summary["host"],
        "engines": engines,
        "limitations": raw_summary["limitations"],
        "evidence_boundary": {
            "fp16_inspector": "COMPACT layer strings; ONNX metadata and tactic/shape unavailable",
            "mixed_inspector": "DETAILED metadata; used to identify exact ONNX-node fusion patterns",
            "mixed_is_int8": True,
            "mixed_used_as_boundary_example_only": True,
        },
    }


def nsys_analysis(out_dir: Path) -> dict:
    range_rows = list(csv.DictReader(NSYS_RANGE_CSV.open(encoding="utf-8-sig")))
    category_rows = list(csv.DictReader(NSYS_CATEGORY_CSV.open(encoding="utf-8-sig")))
    kernel_summaries = {}
    for label, path in (("fp16_persistent", NSYS_FP16_KERNELS), ("mixed_persistent", NSYS_MIXED_KERNELS)):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        kernel_summaries[label] = {
            "unique_kernel_names": len(rows),
            "total_kernel_instances": sum(int(row["Instances"]) for row in rows),
            "total_kernel_time_ms": round(sum(float(row["Total Time (ns)"]) for row in rows) / 1_000_000, 6),
        }

    payload = {
        "method": "read-only extraction of committed Nsight Systems derived summaries",
        "range_summary": range_rows,
        "kernel_category_summary": category_rows,
        "persistent_kernel_summaries": kernel_summaries,
        "limitations": [
            "These are derived CSV summaries, not raw .nsys-rep timelines.",
            "cuda_api_time_ms includes all CUDA APIs and is not isolated kernel-launch overhead.",
            "No new Phase 8.7 Nsight profile was collected in this repository-only audit.",
        ],
    }
    write_json(out_dir / "nsys_summary.json", payload)
    return payload


def plugin_analysis(out_dir: Path) -> dict:
    paths = {
        "rewrite_prefill": PLUGIN_ROOT / "rewrite_prefill.json",
        "rewrite_decode": PLUGIN_ROOT / "rewrite_decode.json",
        "build_baseline_prefill": PLUGIN_ROOT / "build_baseline_prefill.json",
        "build_plugin_prefill": PLUGIN_ROOT / "build_plugin_prefill.json",
        "build_baseline_decode": PLUGIN_ROOT / "build_baseline_decode.json",
        "build_plugin_decode": PLUGIN_ROOT / "build_plugin_decode.json",
        "measure_baseline_prefill": PLUGIN_ROOT / "measure_baseline_prefill.json",
        "measure_plugin_prefill": PLUGIN_ROOT / "measure_plugin_prefill.json",
        "measure_baseline_decode": PLUGIN_ROOT / "measure_baseline_decode.json",
        "measure_plugin_decode": PLUGIN_ROOT / "measure_plugin_decode.json",
        "compare_prefill": PLUGIN_ROOT / "compare_prefill.json",
        "compare_decode": PLUGIN_ROOT / "compare_decode.json",
    }
    data = {name: read_json(path) for name, path in paths.items()}
    baseline_prefill = data["measure_baseline_prefill"]["mean_ms"]
    plugin_prefill = data["measure_plugin_prefill"]["mean_ms"]
    baseline_decode = data["measure_baseline_decode"]["mean_ms"]
    plugin_decode = data["measure_plugin_decode"]["mean_ms"]
    payload = {
        "phase8_4a_evidence": data,
        "plugin_boundary": {
            "replacement_scope": "Layer 0 input_layernorm only",
            "baseline_vs_plugin_onnx_bytes_prefill": data["rewrite_prefill"]["baseline"]["bytes"] - data["rewrite_prefill"]["plugin"]["bytes"],
            "baseline_vs_plugin_engine_bytes_prefill": data["build_baseline_prefill"]["engine_bytes"] - data["build_plugin_prefill"]["engine_bytes"],
            "baseline_vs_plugin_engine_bytes_decode": data["build_baseline_decode"]["engine_bytes"] - data["build_plugin_decode"]["engine_bytes"],
            "prefill_latency_delta_percent": 100.0 * (plugin_prefill - baseline_prefill) / baseline_prefill,
            "decode_latency_delta_percent": 100.0 * (plugin_decode - baseline_decode) / baseline_decode,
        },
        "missing_plugin_boundary_evidence": {
            "plugin_engine_inspector_layer_count": "UNKNOWN",
            "plugin_engine_fusion_pattern_inventory": "UNKNOWN",
            "baseline_vs_plugin_layer_type_delta": "UNKNOWN",
            "baseline_vs_plugin_kernel_count": "UNKNOWN",
            "baseline_vs_plugin_kernel_launch_count": "UNKNOWN",
            "plugin_specific_nsight_timeline": "UNKNOWN",
        },
        "interpretation": {
            "runtime_delta": "PARTIALLY SUPPORTED / one-node fixed-shape slowdown",
            "which_fusion_disappeared": "UNKNOWN",
            "whether_plugin_bypasses_or_breaks_fusion": "INCONCLUSIVE",
        },
    }
    write_json(out_dir / "plugin_boundary_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inputs = [INSPECTOR_JSON, INSPECTOR_CSV, NSYS_RANGE_CSV, NSYS_CATEGORY_CSV, NSYS_FP16_KERNELS, NSYS_MIXED_KERNELS, *sorted(PLUGIN_ROOT.glob("*.json"))]
    manifest = {
        "phase": "Phase 8.7 TensorRT Fusion Boundary Analysis",
        "method": "repository-only existing-artifact analysis; no engine build, execution, or profiling",
        "inputs": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        },
    }

    inspector = inspector_analysis(args.out)
    nsys = nsys_analysis(args.out)
    plugin = plugin_analysis(args.out)
    manifest["outputs"] = [
        "fusion_pattern_summary.csv",
        "norm_fusion_summary.csv",
        "fusion_pattern_examples.csv",
        "inspector_analysis.json",
        "nsys_summary.json",
        "plugin_boundary_summary.json",
    ]
    write_json(args.out / "inspector_analysis.json", inspector)
    write_json(args.out / "input_manifest.json", manifest)

    print(json.dumps({
        "status": "PASS",
        "artifact_dir": str(args.out),
        "inspector_engines": sorted(inspector["engines"]),
        "nsys_ranges": len(nsys["range_summary"]),
        "plugin_engine_inspector": plugin["missing_plugin_boundary_evidence"]["plugin_engine_inspector_layer_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
