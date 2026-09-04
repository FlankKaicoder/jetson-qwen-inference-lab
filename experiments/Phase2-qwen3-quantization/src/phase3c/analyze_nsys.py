from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


LABELS = ["fp16_persistent", "mixed_persistent"]
RANGES = [
    "PHASE3B_STEADY_PREFILL_S8",
    "PHASE3B_STEADY_DECODE_STEP_0",
    "PHASE3B_STEADY_DECODE_STEP_1",
    "PHASE3B_STEADY_DECODE_STEP_2",
    "PHASE3B_STEADY_DECODE_STEP_3",
]
CHECK_RANGES = ["PHASE3B_INIT", "PHASE3B_WARMUP"]
CATEGORIES = [
    "GEMM", "Attention", "RMSNorm", "RoPE", "Elementwise",
    "Memory", "TensorRT internal", "Unknown",
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


def categorize(name: str) -> str:
    if name.startswith("__myl_"):
        return "TensorRT internal"
    if re.search(r"(xmma_gemm|h16816gemm|cutlass.*gemm)", name, re.I):
        return "GEMM"
    if re.search(r"gemm_mha", name, re.I):
        return "Attention"
    if re.search(r"rms|layernorm|norm", name, re.I):
        return "RMSNorm"
    if re.search(r"rope|rotary", name, re.I):
        return "RoPE"
    if re.search(r"elementwise|vectorized|broadcast", name, re.I):
        return "Elementwise"
    return "Unknown"


def range_wall(stats_dir: Path, label: str, range_name: str) -> float:
    rows = read_csv(stats_dir / f"{label}_nvtx_pushpop_sum.csv")
    for row in rows:
        if row["Range"].lstrip(":") == range_name:
            return float(row["Total Time (ns)"])
    raise KeyError(f"{label}:{range_name}")


def kernel_rows(stats_dir: Path, label: str, range_name: str) -> list[dict]:
    rows = read_csv(stats_dir / f"{label}_nvtx_kern_sum.csv")
    return [row for row in rows if row["NVTX Range"] == f":{range_name}"]


def category_evidence(category: str) -> str:
    return {
        "GEMM": "NAME_BASED_GEMM_TACTIC",
        "Attention": "NAME_BASED_MHA_GEMM_TACTIC_ONLY",
        "RMSNorm": "NAME_BASED_IF_PRESENT_ELSE_NOT_IDENTIFIED",
        "RoPE": "NAME_BASED_IF_PRESENT_ELSE_NOT_IDENTIFIED",
        "Elementwise": "NAME_BASED_TORCH_STYLE_KERNEL",
        "Memory": "NSYS_CUDA_GPU_MEM_TIME_SUM",
        "TensorRT internal": "TRT_NAMESPACE_ONLY_OPERATOR_UNKNOWN",
        "Unknown": "NO_SAFE_NAME_MAPPING",
    }[category]


def summarize_range(nsys_dir: Path, label: str, range_name: str) -> dict:
    stats = nsys_dir / "stats"
    stats3 = nsys_dir / "stats3"
    stats3_mem = nsys_dir / "stats3_mem"
    wall_ns = range_wall(stats, label, range_name)
    kernels = kernel_rows(stats, label, range_name)
    kernel_count = sum(int(float(row["Kern Inst"])) for row in kernels)
    kernel_ns = sum(float(row["Total Time (ns)"]) for row in kernels)
    by_name = defaultdict(float)
    by_category = defaultdict(lambda: {"count": 0, "time_ns": 0.0})
    for row in kernels:
        by_name[row["Kernel Name"]] += float(row["Total Time (ns)"])
        category = categorize(row["Kernel Name"])
        by_category[category]["count"] += int(float(row["Kern Inst"]))
        by_category[category]["time_ns"] += float(row["Total Time (ns)"])

    matches = list(stats3.glob(
        f"{label}_{range_name}_cuda_api_sum_nvtx={range_name}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"EXPECTED_ONE_API_CSV:{label}:{range_name}:{matches}")
    api_rows = read_csv(matches[0])
    api_ns = sum(float(row["Total Time (ns)"]) for row in api_rows)
    api_calls = sum(int(float(row["Num Calls"])) for row in api_rows)
    api_by_name = defaultdict(lambda: {"calls": 0, "time_ns": 0.0})
    for row in api_rows:
        entry = api_by_name[row["Name"]]
        entry["calls"] += int(float(row["Num Calls"]))
        entry["time_ns"] += float(row["Total Time (ns)"])

    mem_matches = list(stats3_mem.glob(
        f"{label}_{range_name}_cuda_gpu_mem_time_sum_nvtx={range_name}*.csv"))
    if len(mem_matches) > 1:
        raise FileNotFoundError(f"EXPECTED_AT_MOST_ONE_MEM_CSV:{label}:{range_name}")
    mem_rows = read_csv(mem_matches[0]) if mem_matches else []
    mem_ns = sum(float(row["Total Time (ns)"]) for row in mem_rows)
    mem_count = sum(int(float(row["Count"])) for row in mem_rows)

    sync = {name: value for name, value in api_by_name.items()
            if "synchronize" in name.lower() or "sync" in name.lower()}
    memcpy = {name: value for name, value in api_by_name.items()
              if name.startswith(("cuMemcpy", "cudaMemcpy"))}
    module_load = api_by_name.get("cuModuleLoadData", {"calls": 0, "time_ns": 0.0})
    module_unload = api_by_name.get("cuModuleUnload", {"calls": 0, "time_ns": 0.0})
    host_gap_ns = max(0.0, wall_ns - max(api_ns, kernel_ns))
    return {
        "label": label,
        "range": range_name,
        "wall_ms": wall_ns / 1e6,
        "gpu_kernel_count": kernel_count,
        "gpu_kernel_time_ms": kernel_ns / 1e6,
        "gpu_busy_pct": 100.0 * kernel_ns / wall_ns,
        "gpu_idle_ms": (wall_ns - kernel_ns) / 1e6,
        "cuda_api_calls": api_calls,
        "cuda_api_time_ms": api_ns / 1e6,
        "cuda_api_pct": 100.0 * api_ns / wall_ns,
        "host_gap_proxy_ms": host_gap_ns / 1e6,
        "sync_calls": sum(value["calls"] for value in sync.values()),
        "sync_time_ms": sum(value["time_ns"] for value in sync.values()) / 1e6,
        "memcpy_api_calls": sum(value["calls"] for value in memcpy.values()),
        "memcpy_api_time_ms": sum(value["time_ns"] for value in memcpy.values()) / 1e6,
        "gpu_memop_calls": mem_count,
        "gpu_memop_time_ms": mem_ns / 1e6,
        "cuModuleLoadData_calls": module_load["calls"],
        "cuModuleLoadData_ms": module_load["time_ns"] / 1e6,
        "cuModuleUnload_calls": module_unload["calls"],
        "cuModuleUnload_ms": module_unload["time_ns"] / 1e6,
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    by_label = defaultdict(list)
    for row in rows:
        if row["range"] in RANGES:
            by_label[row["label"]].append(row)
    output = []
    additive = [
        "wall_ms", "gpu_kernel_count", "gpu_kernel_time_ms", "gpu_idle_ms",
        "cuda_api_calls", "cuda_api_time_ms", "sync_calls", "sync_time_ms",
        "memcpy_api_calls", "memcpy_api_time_ms", "gpu_memop_calls",
        "gpu_memop_time_ms", "cuModuleLoadData_calls", "cuModuleLoadData_ms",
        "cuModuleUnload_calls", "cuModuleUnload_ms",
    ]
    for label, selected in by_label.items():
        prefill = next(row for row in selected if row["range"] == RANGES[0])
        decode = [row for row in selected if row["range"].startswith("PHASE3B_STEADY_DECODE")]
        aggregate = {
            "label": label,
            "workload": "representative_steady_prefill_plus_4_decode",
        }
        for key in additive:
            aggregate[key] = prefill[key] + sum(row[key] for row in decode)
        aggregate["gpu_busy_pct"] = 100.0 * aggregate["gpu_kernel_time_ms"] / aggregate["wall_ms"]
        aggregate["cuda_api_pct"] = 100.0 * aggregate["cuda_api_time_ms"] / aggregate["wall_ms"]
        aggregate["host_gap_proxy_ms"] = max(
            0.0, aggregate["wall_ms"] - max(
                aggregate["cuda_api_time_ms"], aggregate["gpu_kernel_time_ms"]))
        output.append(aggregate)
    return output


def category_rows(nsys_dir: Path, rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_workload = defaultdict(lambda: defaultdict(
        lambda: {"count": 0, "time_ns": 0.0, "names": defaultdict(float)}))
    for row in rows:
        for kernel in kernel_rows(nsys_dir / "stats", row["label"], row["range"]):
            category = categorize(kernel["Kernel Name"])
            target = by_workload[row["label"]][category]
            target["count"] += int(float(kernel["Kern Inst"]))
            target["time_ns"] += float(kernel["Total Time (ns)"])
            target["names"][kernel["Kernel Name"]] += float(kernel["Total Time (ns)"])

    range_output = []
    for row in rows:
        for category in CATEGORIES:
            values = by_workload[row["label"]][category]
            range_output.append({
                "label": row["label"], "range": row["range"], "category": category,
                "count": values["count"], "time_ms": values["time_ns"] / 1e6,
                "evidence": category_evidence(category),
            })

    aggregate_output = []
    name_output = []
    for label in LABELS:
        memory_time_ms = sum(row["gpu_memop_time_ms"] for row in rows
                             if row["label"] == label)
        memory_count = sum(row["gpu_memop_calls"] for row in rows
                           if row["label"] == label)
        for category in CATEGORIES:
            count = by_workload[label][category]["count"]
            time_ns = by_workload[label][category]["time_ns"]
            names = defaultdict(float)
            for name, value in by_workload[label][category]["names"].items():
                names[name] = value
            if category == "Memory":
                count = memory_count
                time_ns = memory_time_ms * 1e6
            aggregate_output.append({
                "label": label, "category": category, "count": count,
                "time_ms": time_ns / 1e6,
                "top_names": [name for name, _ in sorted(
                    names.items(), key=lambda item: -item[1])[:10]],
                "evidence": category_evidence(category),
            })
        all_names = defaultdict(float)
        for category in CATEGORIES:
            for name, value in by_workload[label][category]["names"].items():
                all_names[name] += value
        for name, time_ns in sorted(all_names.items(), key=lambda item: -item[1])[:25]:
            name_output.append({
                "label": label, "kernel_name": name, "time_ms": time_ns / 1e6,
                "category": categorize(name),
                "evidence": "NSYS_NVTX_KERNEL_NAME_AGGREGATE",
            })
    return range_output, aggregate_output, name_output


def comparison_rows(summary: list[dict]) -> list[dict]:
    output = []
    for range_name in RANGES:
        fp = next(row for row in summary
                  if row["label"] == "fp16_persistent" and row["range"] == range_name)
        mx = next(row for row in summary
                  if row["label"] == "mixed_persistent" and row["range"] == range_name)
        output.append({
            "range": range_name,
            "fp16_wall_ms": fp["wall_ms"], "mixed_wall_ms": mx["wall_ms"],
            "mixed_minus_fp16_wall_ms": mx["wall_ms"] - fp["wall_ms"],
            "fp16_gpu_kernel_time_ms": fp["gpu_kernel_time_ms"],
            "mixed_gpu_kernel_time_ms": mx["gpu_kernel_time_ms"],
            "mixed_minus_fp16_gpu_kernel_ms": mx["gpu_kernel_time_ms"] - fp["gpu_kernel_time_ms"],
            "fp16_cuda_api_time_ms": fp["cuda_api_time_ms"],
            "mixed_cuda_api_time_ms": mx["cuda_api_time_ms"],
            "mixed_minus_fp16_cuda_api_ms": mx["cuda_api_time_ms"] - fp["cuda_api_time_ms"],
            "fp16_gpu_busy_pct": fp["gpu_busy_pct"],
            "mixed_gpu_busy_pct": mx["gpu_busy_pct"],
            "fp16_sync_time_ms": fp["sync_time_ms"],
            "mixed_sync_time_ms": mx["sync_time_ms"],
            "fp16_module_load_unload_ms": fp["cuModuleLoadData_ms"] + fp["cuModuleUnload_ms"],
            "mixed_module_load_unload_ms": mx["cuModuleLoadData_ms"] + mx["cuModuleUnload_ms"],
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = [summarize_range(args.nsys_dir, label, range_name)
               for label in LABELS for range_name in RANGES]
    checks = [summarize_range(args.nsys_dir, label, range_name)
              for label in LABELS for range_name in CHECK_RANGES]
    aggregates = aggregate_rows(summary)
    range_categories, aggregate_categories, top_kernels = category_rows(
        args.nsys_dir, summary)
    comparisons = comparison_rows(summary)

    args.out.mkdir(parents=True, exist_ok=True)
    def rounded(rows: list[dict]) -> list[dict]:
        return [{key: round(value, 6) if isinstance(value, float) else value
                 for key, value in row.items()} for row in rows]
    write_csv(args.out / "c2_range_summary.csv", rounded(summary))
    write_csv(args.out / "c2_module_check.csv", rounded(checks))
    write_csv(args.out / "c2_aggregate_summary.csv", rounded(aggregates))
    write_csv(args.out / "c2_fp16_vs_mixed.csv", rounded(comparisons))
    write_csv(args.out / "c3_range_categories.csv", rounded(range_categories))
    write_csv(args.out / "c3_top_contributors.csv", rounded(aggregate_categories))
    write_csv(args.out / "c3_top_kernels_by_name.csv", rounded(top_kernels))
    (args.out / "c2_c3_detail.json").write_text(json.dumps({
        "phase": "Phase 3-C2/C3",
        "labels": LABELS,
        "steady_ranges": RANGES,
        "summary": rounded(summary),
        "module_checks": rounded(checks),
        "aggregates": rounded(aggregates),
        "comparisons": rounded(comparisons),
        "aggregate_categories": rounded(aggregate_categories),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rounded(aggregates), indent=2))


if __name__ == "__main__":
    main()
