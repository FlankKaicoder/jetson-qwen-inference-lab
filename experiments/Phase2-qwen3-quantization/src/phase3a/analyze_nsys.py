from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def categorize(name: str) -> str:
    if "xmma_gemm_i8" in name:
        return "GEMM_INT8_TRT"
    if "xmma_gemm_f16" in name or "h16816gemm" in name:
        return "GEMM_FP16_TRT"
    if "gemm_mha" in name:
        return "GEMM_FP16_ATTENTION"
    if name.startswith("__myl_"):
        return "TRT_MYELIN_FUSED_UNKNOWN"
    if "elementwise_kernel" in name or "vectorized" in name:
        return "TORCH_ELEMENTWISE"
    return "UNKNOWN"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def kernel_rows_for_range(stats_dir: Path, runtime: str, range_name: str) -> list[dict]:
    path = stats_dir / f"{runtime}_nvtx_kern_sum.csv"
    rows = read_csv(path)
    return [row for row in rows if row["NVTX Range"] == f":{range_name}"]


def kernel_summary(rows: list[dict]) -> dict:
    categories = defaultdict(lambda: {"count": 0, "time_ns": 0.0, "names": defaultdict(float)})
    for row in rows:
        category = categorize(row["Kernel Name"])
        categories[category]["count"] += int(float(row["Kern Inst"]))
        categories[category]["time_ns"] += float(row["Total Time (ns)"])
        categories[category]["names"][row["Kernel Name"]] += float(row["Total Time (ns)"])
    return {
        "kernel_count": sum(cat["count"] for cat in categories.values()),
        "kernel_time_ns": sum(cat["time_ns"] for cat in categories.values()),
        "categories": dict(categories),
    }


def api_summary(stats3_dir: Path, runtime: str, range_name: str) -> dict:
    path = stats3_dir / f"{runtime}_{range_name}_cuda_api_sum_nvtx={range_name}.csv"
    rows = read_csv(path)
    total_ns = sum(float(row["Total Time (ns)"]) for row in rows)
    calls = sum(int(float(row["Num Calls"])) for row in rows)
    by_name = {}
    for row in rows:
        entry = by_name.setdefault(row["Name"], {"calls": 0, "time_ns": 0.0})
        entry["calls"] += int(float(row["Num Calls"]))
        entry["time_ns"] += float(row["Total Time (ns)"])
    return {
        "api_calls": calls,
        "api_time_ns": total_ns,
        "by_name": dict(sorted(by_name.items(), key=lambda kv: -kv[1]["time_ns"])),
    }


def range_wall(pushpop_path: Path, range_name: str) -> float:
    rows = read_csv(pushpop_path)
    for row in rows:
        if row["Range"].lstrip(":") == range_name:
            return float(row["Total Time (ns)"])
    raise KeyError(range_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-dir", type=Path, required=True,
                        help="Directory containing stats/, stats2/ and stats3/ exports")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stats = args.nsys_dir / "stats"
    stats2 = args.nsys_dir / "stats2"
    stats3 = args.nsys_dir / "stats3"
    runtimes = ["fp16", "mixed"]
    ranges = ["A1_prefill_S8", "A1_decode_step_0"]

    summary_rows = []
    detail = {}
    for runtime in runtimes:
        pushpop = stats2 / f"{runtime}_nvtx_pushpop_sum.csv"
        for range_name in ranges:
            kernels = kernel_summary(kernel_rows_for_range(stats, runtime, range_name))
            api = api_summary(stats3, runtime, range_name)
            wall_ns = range_wall(pushpop, range_name)
            module_load = api["by_name"].get("cuModuleLoadData", {})
            module_unload = api["by_name"].get("cuModuleUnload", {})
            sync = api["by_name"].get("cudaStreamSynchronize", {})
            dtoh = api["by_name"].get("cuMemcpyDtoH_v2", {})
            host_remainder_ns = wall_ns - kernels["kernel_time_ns"] - api["api_time_ns"]
            row = {
                "runtime": runtime,
                "range": range_name,
                "wall_ms": round(wall_ns / 1e6, 1),
                "gpu_kernel_time_ms": round(kernels["kernel_time_ns"] / 1e6, 1),
                "gpu_kernel_count": kernels["kernel_count"],
                "gpu_busy_pct_of_wall": round(100.0 * kernels["kernel_time_ns"] / wall_ns, 2),
                "cuda_api_time_ms": round(api["api_time_ns"] / 1e6, 1),
                "cuda_api_calls": api["api_calls"],
                "cuda_api_pct_of_wall": round(100.0 * api["api_time_ns"] / wall_ns, 2),
                "host_nonsystem_time_ms": round(host_remainder_ns / 1e6, 1),
                "host_nonsystem_pct_of_wall": round(100.0 * host_remainder_ns / wall_ns, 2),
                "cuModuleLoadData_calls": module_load.get("calls", 0),
                "cuModuleLoadData_ms": round(module_load.get("time_ns", 0.0) / 1e6, 1),
                "cuModuleUnload_calls": module_unload.get("calls", 0),
                "cuModuleUnload_ms": round(module_unload.get("time_ns", 0.0) / 1e6, 1),
                "cudaStreamSynchronize_calls": sync.get("calls", 0),
                "cudaStreamSynchronize_ms": round(sync.get("time_ns", 0.0) / 1e6, 1),
                "cuMemcpyDtoH_calls": dtoh.get("calls", 0),
                "cuMemcpyDtoH_ms": round(dtoh.get("time_ns", 0.0) / 1e6, 1),
            }
            summary_rows.append(row)
            detail[f"{runtime}/{range_name}"] = {
                "kernel_categories": {
                    name: {"count": value["count"], "time_ms": round(value["time_ns"] / 1e6, 2)}
                    for name, value in kernels["categories"].items()
                },
                "api_top": {
                    name: {"calls": value["calls"], "time_ms": round(value["time_ns"] / 1e6, 2)}
                    for name, value in list(api["by_name"].items())[:12]
                },
            }

    attribution_rows = []
    for range_name in ranges:
        fp = next(row for row in summary_rows if row["runtime"] == "fp16" and row["range"] == range_name)
        mx = next(row for row in summary_rows if row["runtime"] == "mixed" and row["range"] == range_name)
        categories = sorted(set(detail[f"fp16/{range_name}"]["kernel_categories"])
                            | set(detail[f"mixed/{range_name}"]["kernel_categories"]))
        for category in categories:
            fp_cat = detail[f"fp16/{range_name}"]["kernel_categories"].get(category, {"count": 0, "time_ms": 0.0})
            mx_cat = detail[f"mixed/{range_name}"]["kernel_categories"].get(category, {"count": 0, "time_ms": 0.0})
            attribution_rows.append({
                "workload": range_name,
                "category": category,
                "fp16_count": fp_cat["count"],
                "mixed_count": mx_cat["count"],
                "count_delta": mx_cat["count"] - fp_cat["count"],
                "fp16_total_time_ms": fp_cat["time_ms"],
                "mixed_total_time_ms": mx_cat["time_ms"],
                "time_delta_ms": round(mx_cat["time_ms"] - fp_cat["time_ms"], 2),
                "evidence": "nsys nvtx_kern_sum kernel-name aggregation",
                "confidence": ("MAPPING_UNKNOWN" if category == "TRT_MYELIN_FUSED_UNKNOWN"
                               else "NAME_BASED"),
            })
        attribution_rows.append({
            "workload": range_name,
            "category": "CUDA_API_ALL",
            "fp16_count": fp["cuda_api_calls"],
            "mixed_count": mx["cuda_api_calls"],
            "count_delta": mx["cuda_api_calls"] - fp["cuda_api_calls"],
            "fp16_total_time_ms": fp["cuda_api_time_ms"],
            "mixed_total_time_ms": mx["cuda_api_time_ms"],
            "time_delta_ms": round(mx["cuda_api_time_ms"] - fp["cuda_api_time_ms"], 1),
            "evidence": "nsys cuda_api_sum filtered by NVTX range",
            "confidence": "NAME_BASED",
        })
        attribution_rows.append({
            "workload": range_name,
            "category": "cuModuleLoadData+Unload",
            "fp16_count": fp["cuModuleLoadData_calls"] + fp["cuModuleUnload_calls"],
            "mixed_count": mx["cuModuleLoadData_calls"] + mx["cuModuleUnload_calls"],
            "count_delta": (mx["cuModuleLoadData_calls"] + mx["cuModuleUnload_calls"])
                           - (fp["cuModuleLoadData_calls"] + fp["cuModuleUnload_calls"]),
            "fp16_total_time_ms": round(fp["cuModuleLoadData_ms"] + fp["cuModuleUnload_ms"], 1),
            "mixed_total_time_ms": round(mx["cuModuleLoadData_ms"] + mx["cuModuleUnload_ms"], 1),
            "time_delta_ms": round((mx["cuModuleLoadData_ms"] + mx["cuModuleUnload_ms"])
                                   - (fp["cuModuleLoadData_ms"] + fp["cuModuleUnload_ms"]), 1),
            "evidence": "nsys cuda_api_sum filtered by NVTX range",
            "confidence": "NAME_BASED",
        })

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "a2_range_summary.csv", summary_rows)
    write_csv(args.out / "a3_kernel_attribution.csv", attribution_rows)
    (args.out / "a2_a3_detail.json").write_text(json.dumps({
        "summary": summary_rows,
        "detail": detail,
    }, indent=2) + "\n")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
