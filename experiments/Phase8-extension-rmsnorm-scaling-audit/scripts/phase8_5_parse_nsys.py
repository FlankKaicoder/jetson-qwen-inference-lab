#!/usr/bin/env python3
"""Reduce Nsight Systems CSV reports into compact kernel/launch summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def number(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def parse_report(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"file": str(path), "rows": 0}
    columns = {column.lower(): column for column in rows[0]}
    name_column = next((columns.get(key) for key in ("name", "kernel name", "api")
                        if key in columns), None)
    count_column = next((columns.get(key) for key in ("instances", "num calls")
                         if key in columns), None)
    time_columns = [column for original in columns.values()
                    for column in [original]
                    if "total time" in column.lower() or "total (ms)" in column.lower()]
    if not count_column or not name_column:
        return {"file": str(path), "rows": len(rows), "columns": list(rows[0])}
    total_ns = 0.0
    total_count = 0.0
    by_name = {}
    for row in rows:
        count = number(row.get(count_column, "0"))
        duration = number(row.get(time_columns[0], "0")) if time_columns else 0.0
        name = row.get(name_column, "UNKNOWN")
        total_count += count
        total_ns += duration
        item = by_name.setdefault(name, {"instances": 0.0, "total_time_ns": 0.0})
        item["instances"] += count
        item["total_time_ns"] += duration
    unit = "ns" if any("ns" in column.lower() for column in time_columns) else "ms"
    return {
        "file": str(path), "rows": len(rows),
        "instance_count": total_count,
        "total_time": total_ns,
        "time_unit": unit,
        "distinct_names": len(by_name),
        "by_name": by_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gpu_files = sorted(args.report_dir.glob("*cuda_gpu_kern_sum*.csv"))
    api_files = sorted(args.report_dir.glob("*cuda_api_sum*.csv"))
    gpu_reports = [parse_report(path) for path in gpu_files]
    api_reports = [parse_report(path) for path in api_files]
    launch_names = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel"}
    launches = []
    for report in api_reports:
        for name, item in report.get("by_name", {}).items():
            if name in launch_names:
                launches.append({"name": name, "calls": item["instances"],
                                 "total_time": item["total_time_ns"],
                                 "time_unit": report.get("time_unit", "ns")})
    gpu = gpu_reports[0] if gpu_reports else {"instance_count": 0.0, "total_time": 0.0,
                                               "time_unit": "UNKNOWN", "distinct_names": 0}
    result = {
        "report_directory": str(args.report_dir),
        "gpu_kernel_count": gpu.get("instance_count", 0.0),
        "gpu_kernel_total_time": gpu.get("total_time", 0.0),
        "gpu_kernel_time_unit": gpu.get("time_unit", "UNKNOWN"),
        "gpu_kernel_distinct_names": gpu.get("distinct_names", 0),
        "cuda_launch_count": sum(item["calls"] for item in launches),
        "cuda_launch_rows": launches,
        "gpu_reports": gpu_reports,
        "api_reports": api_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
