from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


RANGES = [
    "PHASE3B_STEADY_PREFILL_S8",
    "PHASE3B_STEADY_DECODE_STEP_0",
    "PHASE3B_STEADY_DECODE_STEP_1",
    "PHASE3B_STEADY_DECODE_STEP_2",
    "PHASE3B_STEADY_DECODE_STEP_3",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def range_wall(stats_dir: Path, label: str, range_name: str) -> float:
    rows = read_csv(stats_dir / f"{label}_nvtx_pushpop_sum.csv")
    for row in rows:
        if row["Range"].lstrip(":") == range_name:
            return float(row["Total Time (ns)"])
    raise KeyError(range_name)


def kernel_summary(stats_dir: Path, label: str, range_name: str) -> dict:
    rows = read_csv(stats_dir / f"{label}_nvtx_kern_sum.csv")
    selected = [row for row in rows if row["NVTX Range"] == f":{range_name}"]
    total_ns = sum(float(row["Total Time (ns)"]) for row in selected)
    count = sum(int(float(row["Kern Inst"])) for row in selected)
    by_name = defaultdict(float)
    for row in selected:
        by_name[row["Kernel Name"]] += float(row["Total Time (ns)"])
    return {
        "kernel_count": count,
        "kernel_time_ns": total_ns,
        "kernel_names": dict(sorted(by_name.items(), key=lambda item: -item[1])),
    }


def api_summary(stats_dir: Path, label: str, range_name: str) -> dict:
    matches = list(stats_dir.glob(
        f"{label}_{range_name}_cuda_api_sum_nvtx={range_name}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"EXPECTED_ONE_API_CSV:{label}:{range_name}:{matches}")
    rows = read_csv(matches[0])
    total_ns = sum(float(row["Total Time (ns)"]) for row in rows)
    calls = sum(int(float(row["Num Calls"])) for row in rows)
    by_name = defaultdict(lambda: {"calls": 0, "time_ns": 0.0})
    for row in rows:
        entry = by_name[row["Name"]]
        entry["calls"] += int(float(row["Num Calls"]))
        entry["time_ns"] += float(row["Total Time (ns)"])
    ordered = dict(sorted(by_name.items(), key=lambda item: -item[1]["time_ns"]))
    memcpy = {
        name: value for name, value in ordered.items() if name.startswith("cuMemcpy")
    }
    return {
        "api_calls": calls,
        "api_time_ns": total_ns,
        "by_name": ordered,
        "memcpy_calls": sum(value["calls"] for value in memcpy.values()),
        "memcpy_time_ns": sum(value["time_ns"] for value in memcpy.values()),
    }


def summary_row(stats_dir: Path, label: str, range_name: str) -> dict:
    wall_ns = range_wall(stats_dir, label, range_name)
    kernels = kernel_summary(stats_dir, label, range_name)
    api = api_summary(stats_dir.parent / "stats3", label, range_name)
    module_load = api["by_name"].get("cuModuleLoadData", {"calls": 0, "time_ns": 0.0})
    module_unload = api["by_name"].get("cuModuleUnload", {"calls": 0, "time_ns": 0.0})
    sync = api["by_name"].get("cudaStreamSynchronize", {"calls": 0, "time_ns": 0.0})
    dtoh = api["by_name"].get("cuMemcpyDtoH_v2", {"calls": 0, "time_ns": 0.0})
    return {
        "label": label,
        "range": range_name,
        "wall_ms": wall_ns / 1e6,
        "gpu_kernel_count": kernels["kernel_count"],
        "gpu_kernel_time_ms": kernels["kernel_time_ns"] / 1e6,
        "gpu_busy_pct": 100.0 * kernels["kernel_time_ns"] / wall_ns,
        "cuda_api_calls": api["api_calls"],
        "cuda_api_time_ms": api["api_time_ns"] / 1e6,
        "cuda_api_pct": 100.0 * api["api_time_ns"] / wall_ns,
        "cuModuleLoadData_calls": module_load["calls"],
        "cuModuleLoadData_ms": module_load["time_ns"] / 1e6,
        "cuModuleUnload_calls": module_unload["calls"],
        "cuModuleUnload_ms": module_unload["time_ns"] / 1e6,
        "cudaStreamSynchronize_calls": sync["calls"],
        "cudaStreamSynchronize_ms": sync["time_ns"] / 1e6,
        "cuMemcpyDtoH_v2_calls": dtoh["calls"],
        "cuMemcpyDtoH_v2_ms": dtoh["time_ns"] / 1e6,
        "memcpy_calls": api["memcpy_calls"],
        "memcpy_ms": api["memcpy_time_ns"] / 1e6,
    }


def reduction(old: float, new: float) -> float:
    return (old - new) / old if old else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = [
        "fp16_legacy", "fp16_persistent", "mixed_legacy", "mixed_persistent"
    ]
    summary = [summary_row(args.stats_dir, label, range_name)
               for label in labels for range_name in RANGES]
    comparisons = []
    detail = {}
    for range_name in RANGES:
        for runtime in ("fp16", "mixed"):
            legacy = next(row for row in summary
                          if row["label"] == f"{runtime}_legacy"
                          and row["range"] == range_name)
            persistent = next(row for row in summary
                              if row["label"] == f"{runtime}_persistent"
                              and row["range"] == range_name)
            comparisons.append({
                "runtime": runtime,
                "range": range_name,
                "wall_reduction": reduction(legacy["wall_ms"], persistent["wall_ms"]),
                "cuda_api_time_reduction": reduction(
                    legacy["cuda_api_time_ms"], persistent["cuda_api_time_ms"]),
                "module_load_unload_count_old": (legacy["cuModuleLoadData_calls"]
                                                 + legacy["cuModuleUnload_calls"]),
                "module_load_unload_count_new": (persistent["cuModuleLoadData_calls"]
                                                 + persistent["cuModuleUnload_calls"]),
                "module_load_unload_count_reduction": reduction(
                    legacy["cuModuleLoadData_calls"] + legacy["cuModuleUnload_calls"],
                    persistent["cuModuleLoadData_calls"] + persistent["cuModuleUnload_calls"]),
                "module_load_unload_time_reduction": reduction(
                    legacy["cuModuleLoadData_ms"] + legacy["cuModuleUnload_ms"],
                    persistent["cuModuleLoadData_ms"] + persistent["cuModuleUnload_ms"]),
                "gpu_kernel_count_old": legacy["gpu_kernel_count"],
                "gpu_kernel_count_new": persistent["gpu_kernel_count"],
                "gpu_kernel_time_old_ms": legacy["gpu_kernel_time_ms"],
                "gpu_kernel_time_new_ms": persistent["gpu_kernel_time_ms"],
            })
    for row in summary:
        kernels = kernel_summary(args.stats_dir, row["label"], row["range"])
        api = api_summary(args.stats_dir.parent / "stats3", row["label"], row["range"])
        detail[f"{row['label']}/{row['range']}"] = {
            "kernel_names": {
                name: {"time_ms": value / 1e6}
                for name, value in list(kernels["kernel_names"].items())[:20]
            },
            "api_top": {
                name: {"calls": value["calls"], "time_ms": value["time_ns"] / 1e6}
                for name, value in list(api["by_name"].items())[:20]
            },
        }

    args.out.mkdir(parents=True, exist_ok=True)
    rounded_summary = []
    for row in summary:
        rounded_summary.append({
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in row.items()
        })
    rounded_comparisons = []
    for row in comparisons:
        rounded_comparisons.append({
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in row.items()
        })
    write_csv(args.out / "b4_range_summary.csv", rounded_summary)
    write_csv(args.out / "b4_causal_ab.csv", rounded_comparisons)
    (args.out / "b4_detail.json").write_text(json.dumps({
        "summary": rounded_summary,
        "comparisons": rounded_comparisons,
        "detail": detail,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rounded_comparisons, indent=2))


if __name__ == "__main__":
    main()
