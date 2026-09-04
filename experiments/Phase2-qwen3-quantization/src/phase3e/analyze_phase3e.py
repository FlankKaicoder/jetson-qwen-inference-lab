from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_tactic(value: str) -> str:
    value = value.strip()
    if value.endswith("_by_fusion_tactic"):
        value = value[: -len("_by_fusion_tactic")]
    if value.startswith("trt_"):
        value = value[len("trt_"):]
    return value


def operator_base(operator: str) -> str:
    if ":" not in operator:
        return operator
    return operator.split(":", 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm-inventory", type=Path, required=True)
    parser.add_argument("--top-kernels", type=Path, required=True)
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gemm_rows = read_csv(args.gemm_inventory)
    top_rows = [row for row in read_csv(args.top_kernels)
                if row.get("category") == "GEMM"]
    totals = {row["label"]: float(row["gpu_kernel_time_ms"])
              for row in read_csv(args.aggregate_summary)}

    shape_groups: dict[tuple[str, str, str, str, str, str], dict[str, object]] = defaultdict(
        lambda: {"frequency": 0, "layer_indices": set()}
    )
    for row in gemm_rows:
        if row["operator"] == "UNKNOWN":
            continue
        key = (
            row["engine"], operator_base(row["operator"]), row["matrix_m"],
            row["matrix_n"], row["matrix_k"], row["precision"],
        )
        group = shape_groups[key]
        group["frequency"] += 1
        group["layer_indices"].add(row["layer_index"])
        group.setdefault("tactics", set()).add(row["tactic"])
        group.setdefault("mapping_evidence", set()).add(row["mapping_evidence"])
    shape_output = []
    for key, value in sorted(shape_groups.items()):
        shape_output.append({
            "engine": key[0], "operator": key[1], "matrix_m": key[2],
            "matrix_n": key[3], "matrix_k": key[4], "precision": key[5],
            "frequency": value["frequency"],
            "tactics": ";".join(sorted(str(x) for x in value["tactics"])),
            "layer_indices": ";".join(str(x) for x in sorted(value["layer_indices"])),
            "mapping_evidence": ";".join(sorted(str(x) for x in value["mapping_evidence"])),
        })

    by_tactic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in gemm_rows:
        if row["tactic"] not in ("NONE", "UNKNOWN_NO_DETAILED_INSPECTOR"):
            by_tactic[normalized_tactic(row["tactic"])].append(row)

    kernel_output = []
    unique: dict[str, dict[str, object]] = {}
    for row in top_rows:
        label = row["label"]
        kernel = row["kernel_name"]
        time_ms = float(row["time_ms"])
        total = totals[label]
        tactic_key = normalized_tactic(kernel)
        matches = by_tactic.get(tactic_key, [])
        operators = sorted({operator_base(x["operator"]) for x in matches if x["operator"] != "UNKNOWN"})
        item = {
            "kernel_name": kernel,
            "runtime": label,
            "time_ms": time_ms,
            "phase3c_total_kernel_ms": total,
            "phase3c_pct": 100.0 * time_ms / total,
            "operator_match": ";".join(operators) if operators else "UNKNOWN",
            "mapping_evidence": "INSPECTOR_TACTIC_MATCH" if operators else "TACTIC_NOT_IN_DETAILED_INSPECTOR",
        }
        kernel_output.append(item)
        aggregate = unique.setdefault(kernel, {
            "kernel_name": kernel,
            "aggregate_time_ms": 0.0,
            "runtimes": [],
            "max_runtime_pct": 0.0,
            "operator_match": set(),
            "mapping_evidence": set(),
        })
        aggregate["aggregate_time_ms"] = float(aggregate["aggregate_time_ms"]) + time_ms
        aggregate["runtimes"].append(label)
        aggregate["max_runtime_pct"] = max(float(aggregate["max_runtime_pct"]), item["phase3c_pct"])
        aggregate["operator_match"].update(operators)
        aggregate["mapping_evidence"].add(item["mapping_evidence"])

    selected = sorted(unique.values(), key=lambda row: -float(row["aggregate_time_ms"]))[:3]
    selected_output = []
    for rank, row in enumerate(selected, 1):
        selected_output.append({
            "ncu_rank": rank,
            "kernel_name": row["kernel_name"],
            "aggregate_time_ms": row["aggregate_time_ms"],
            "observed_runtimes": ";".join(sorted(row["runtimes"])),
            "max_single_runtime_pct": row["max_runtime_pct"],
            "operator_match": ";".join(sorted(row["operator_match"])) or "UNKNOWN",
            "mapping_evidence": ";".join(sorted(row["mapping_evidence"])),
            "ncu_kernel_filter": f"regex:{re.escape(str(row['kernel_name']))}",
        })

    write_csv(args.out / "e2_operator_shape_groups.csv", shape_output)
    write_csv(args.out / "e3_nsys_gemm_candidates.csv", kernel_output)
    write_csv(args.out / "e3_selected_nsys_candidates.csv", selected_output)
    (args.out / "analysis_summary.json").write_text(json.dumps({
        "phase": "Phase 3-E2/E3 selection",
        "method": "Join EngineInspector tactic inventory with Phase 3-C aggregate NSYS kernel totals",
        "phase3c_total_kernel_ms": totals,
        "trigger": "Any top single GEMM kernel >20% of that runtime's total kernel time",
        "trigger_evidence": [
            row for row in kernel_output if row["phase3c_pct"] > 20.0
        ],
        "selected_unique_kernel_count": len(selected_output),
        "interpretation_limit": "Tactic-level runtime time is aggregate across executions and cannot be split into individual operators.",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS", "shape_groups": len(shape_output),
        "gemm_kernel_rows": len(kernel_output), "selected": len(selected_output),
    }))


if __name__ == "__main__":
    main()
