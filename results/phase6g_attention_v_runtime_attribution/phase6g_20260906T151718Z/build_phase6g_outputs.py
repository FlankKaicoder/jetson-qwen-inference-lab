from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEMANTIC_PATH = (
    REPO
    / "results/phase6a_unknown_attention_matmul_attribution/20260906T040500Z/semantic_attribution.csv"
)
RUNTIME_CONTEXT_PATH = (
    REPO
    / "results/phase6a_unknown_attention_matmul_attribution/20260906T040500Z/runtime_context_attribution.csv"
)
RAW_EVENTS_PATH = HERE / "raw_attention_v_events.jsonl"

ATTRIBUTION_FIELDS = [
    "candidate_layer",
    "decoder_layer",
    "semantic_identity",
    "TensorRT_layer",
    "ONNX_identity",
    "NVTX_range",
    "kernel_name",
    "kernel_family",
    "correlation_id",
    "duration_ns",
    "warmup_or_steady",
    "runtime_boundary",
    "confidence",
    "source_artifact",
]
MAPPING_FIELDS = [
    "semantic_operator",
    "runtime_layer",
    "kernel_name",
    "kernel_family",
    "launch_count",
    "total_duration_ns",
    "median_duration_ns",
    "steady_duration_ns",
    "confidence",
]
BOUNDARY_FIELDS = [
    "runtime_boundary",
    "warmup_or_steady",
    "av_instance_count",
    "av_kernel_count",
    "total_duration_ns",
    "steady_denominator_ns",
    "steady_share",
    "evidence",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kernel_name(kernel: dict) -> str:
    return kernel.get("demangled_name") or kernel.get("short_name") or "UNKNOWN"


def kernel_family(name: str) -> str:
    if name.startswith("sm80_xmma_gemm_f16f16_f16f32_f32"):
        return "sm80_xmma_gemm_f16f16_f16f32_f32"
    if name.startswith("sm80_xmma_gemm"):
        return "sm80_xmma_gemm"
    if "h16816gemm" in name:
        return "h16816"
    return "UNKNOWN"


def trt_layer(decode_evidence: str) -> str:
    match = re.search(r"decode EngineInspector layer \d+ (\S+)$", decode_evidence)
    if not match:
        return "UNKNOWN"
    return match.group(1)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    semantic_rows = read_csv(SEMANTIC_PATH)
    by_onnx = {row["onnx_node"]: row for row in semantic_rows}
    runtime_context_rows = read_csv(RUNTIME_CONTEXT_PATH)
    context_by_candidate = {
        row["candidate_id"]: row for row in runtime_context_rows
    }
    events = [json.loads(line) for line in RAW_EVENTS_PATH.read_text(
        encoding="utf-8"
    ).splitlines() if line]

    if len(events) != 112:
        raise RuntimeError(f"EXPECTED_112_AV_EVENTS_GOT_{len(events)}")
    if any(event["source_sqlite_sha256"] != (
        "ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8"
    ) for event in events):
        raise RuntimeError("SOURCE_SQLITE_HASH_MISMATCH")
    if any(len(event["correlated_kernels"]) != 1 for event in events):
        raise RuntimeError("EXPECTED_ONE_KERNEL_PER_AV_RANGE")
    if any(event["nvtx_boundary"] != event["kernel_boundaries"][0]
           for event in events):
        raise RuntimeError("NVTX_AND_KERNEL_BOUNDARY_MISMATCH")

    attribution_rows = []
    for event in events:
        onnx_identity = event["nvtx_range"]
        semantic = by_onnx[onnx_identity]
        context = context_by_candidate[semantic["candidate_id"]]
        kernel = event["correlated_kernels"][0]
        name = kernel_name(kernel)
        boundary = event["kernel_boundaries"][0]
        attribution_rows.append({
            "candidate_layer": semantic["candidate_id"],
            "decoder_layer": semantic["decoder_layer"],
            "semantic_identity": semantic["semantic_identity"],
            "TensorRT_layer": trt_layer(context["decode_evidence"]),
            "ONNX_identity": onnx_identity,
            "NVTX_range": event["nvtx_range"],
            "kernel_name": name,
            "kernel_family": kernel_family(name),
            "correlation_id": kernel["correlationId"],
            "duration_ns": kernel["end"] - kernel["start"],
            "warmup_or_steady": "STEADY" if boundary.startswith(
                "PHASE3B_STEADY_"
            ) else "UNKNOWN",
            "runtime_boundary": boundary,
            "confidence": "HIGH",
            "source_artifact": (
                "PHASE3C_MIXED_PERSISTENT_SQLITE;"
                "PHASE6A_SEMANTIC_ATTRIBUTION;"
                "PHASE6A_RUNTIME_CONTEXT_ATTRIBUTION"
            ),
        })

    write_csv(
        HERE / "attention_v_runtime_attribution.csv",
        ATTRIBUTION_FIELDS,
        attribution_rows,
    )

    grouped = defaultdict(list)
    for row in attribution_rows:
        grouped[row["candidate_layer"]].append(row)
    mapping_rows = []
    for candidate_id in sorted(grouped):
        rows = grouped[candidate_id]
        durations = [int(row["duration_ns"]) for row in rows]
        steady_durations = [
            int(row["duration_ns"]) for row in rows
            if row["warmup_or_steady"] == "STEADY"
        ]
        mapping_rows.append({
            "semantic_operator": rows[0]["semantic_identity"],
            "runtime_layer": rows[0]["TensorRT_layer"],
            "kernel_name": rows[0]["kernel_name"],
            "kernel_family": rows[0]["kernel_family"],
            "launch_count": len(rows),
            "total_duration_ns": sum(durations),
            "median_duration_ns": statistics.median(durations),
            "steady_duration_ns": sum(steady_durations),
            "confidence": "HIGH",
        })
    write_csv(
        HERE / "attention_v_kernel_mapping.csv",
        MAPPING_FIELDS,
        mapping_rows,
    )

    steady_denominator_ns = 147_830_560
    boundary_groups = defaultdict(list)
    for row in attribution_rows:
        boundary_groups[row["runtime_boundary"]].append(row)
    boundary_rows = []
    for boundary in (
        "PHASE3B_INIT",
        "PHASE3B_WARMUP",
        "PHASE3B_STEADY_PREFILL_S8",
        "PHASE3B_STEADY_DECODE_STEP_0",
        "PHASE3B_STEADY_DECODE_STEP_1",
        "PHASE3B_STEADY_DECODE_STEP_2",
        "PHASE3B_STEADY_DECODE_STEP_3",
    ):
        rows = boundary_groups.get(boundary, [])
        duration_ns = sum(int(row["duration_ns"]) for row in rows)
        is_steady = boundary.startswith("PHASE3B_STEADY_")
        boundary_rows.append({
            "runtime_boundary": boundary,
            "warmup_or_steady": (
                "WARMUP" if boundary == "PHASE3B_WARMUP"
                else "STEADY" if is_steady
                else "NON_REPRESENTATIVE"
            ),
            "av_instance_count": len(rows),
            "av_kernel_count": len(rows),
            "total_duration_ns": duration_ns,
            "steady_denominator_ns": steady_denominator_ns,
            "steady_share": (
                f"{100.0 * duration_ns / steady_denominator_ns:.6f}%"
                if is_steady else "NOT_APPLICABLE"
            ),
            "evidence": "DIRECT_NVTX_CONTAINMENT_AND_CORRELATION_ID",
        })
    boundary_rows.append({
        "runtime_boundary": "REPRESENTATIVE_STEADY_TOTAL",
        "warmup_or_steady": "STEADY",
        "av_instance_count": sum(
            row["av_instance_count"] for row in boundary_rows
            if row["warmup_or_steady"] == "STEADY"
        ),
        "av_kernel_count": sum(
            row["av_kernel_count"] for row in boundary_rows
            if row["warmup_or_steady"] == "STEADY"
        ),
        "total_duration_ns": sum(
            row["total_duration_ns"] for row in boundary_rows
            if row["warmup_or_steady"] == "STEADY"
        ),
        "steady_denominator_ns": steady_denominator_ns,
        "steady_share": (
            f"{100.0 * sum(row['total_duration_ns'] for row in boundary_rows if row['warmup_or_steady'] == 'STEADY') / steady_denominator_ns:.6f}%"
        ),
        "evidence": "SUM_OF_FIVE_REPRESENTATIVE_STEADY_RANGES",
    })
    boundary_rows.append({
        "runtime_boundary": "ALL_TRACE",
        "warmup_or_steady": "NOT_APPLICABLE",
        "av_instance_count": len(attribution_rows),
        "av_kernel_count": len(attribution_rows),
        "total_duration_ns": sum(
            int(row["duration_ns"]) for row in attribution_rows
        ),
        "steady_denominator_ns": 230_950_048,
        "steady_share": (
            f"{100.0 * sum(int(row['duration_ns']) for row in attribution_rows) / 230_950_048:.6f}%"
        ),
        "evidence": "HISTORICAL_MIXED_PERSISTENT_ALL_TRACE_DENOMINATOR",
    })
    write_csv(
        HERE / "attention_v_boundary_analysis.csv",
        BOUNDARY_FIELDS,
        boundary_rows,
    )

    manifest_rows = []
    manifest_paths = [
        SEMANTIC_PATH,
        RUNTIME_CONTEXT_PATH,
        REPO / "results/phase4a_operator_attribution/20260904T134556Z/runtime_nvtx_kernel_mapping.csv",
        REPO / "results/phase3c_residual_runtime/20260904T093500Z_nsys/analysis/c2_c3_detail.json",
        REPO / "results/phase3c_residual_runtime/20260904T093500Z_nsys/analysis/c2_fp16_vs_mixed.csv",
        RAW_EVENTS_PATH,
        HERE / "remote_query.py",
        HERE / "build_phase6g_outputs.py",
    ]
    for path in manifest_paths:
        manifest_rows.append({
            "artifact": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": (
                "DERIVED_RAW_EVENT_EXPORT" if path == RAW_EVENTS_PATH
                else "INPUT_OR_METHOD_EVIDENCE"
            ),
        })
    manifest_rows.insert(0, {
        "artifact": "/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite",
        "bytes": 3477504,
        "sha256": "ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8",
        "role": "REMOTE_RAW_SQLITE",
    })
    write_csv(
        HERE / "evidence_manifest.csv",
        ["artifact", "bytes", "sha256", "role"],
        manifest_rows,
    )

    total_ns = sum(int(row["duration_ns"]) for row in attribution_rows)
    steady_ns = sum(
        int(row["duration_ns"]) for row in attribution_rows
        if row["warmup_or_steady"] == "STEADY"
    )
    if total_ns != 4_237_568 or steady_ns != 4_237_568:
        raise RuntimeError("DURATION_RECONCILIATION_FAILED")


if __name__ == "__main__":
    main()
