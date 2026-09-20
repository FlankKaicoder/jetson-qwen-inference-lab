#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path


def number(value):
    if value is None or value == "":
        return None
    return float(value.replace(",", ""))


def read_ncu(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header_index = next(i for i, row in enumerate(rows) if "Kernel Name" in row)
    header = rows[header_index]
    data = []
    for row in rows[header_index + 1 :]:
        if not row or row[0].startswith("==") or row[0] == "ID":
            continue
        if len(row) != len(header) or not row[0].strip().isdigit():
            continue
        data.append(dict(zip(header, row)))
    if not data:
        raise RuntimeError(f"no NCU kernel rows in {path}")
    return data


def metric(row, name):
    summed = row.get(name + ".sum")
    if summed not in (None, ""):
        return number(summed)
    direct = row.get(name)
    if direct not in (None, ""):
        return number(direct)
    for key, value in row.items():
        if key == name or key.startswith(name + "."):
            parsed = number(value)
            if parsed is not None:
                return parsed
    return None


def summarize_ncu(path, mode):
    rows = read_ncu(path)
    kernels = []
    for row in rows:
        duration_ns = metric(row, "gpu__time_duration")
        loads = metric(row, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld")
        stores = metric(row, "l1tex__t_sectors_pipe_lsu_mem_global_op_st")
        occupancy = metric(row, "sm__warps_active.avg.pct_of_peak_sustained_active")
        kernels.append(
            {
                "mode": mode,
                "kernel": row["Kernel Name"],
                "duration_ns": duration_ns,
                "global_load_sectors": loads,
                "global_store_sectors": stores,
                "global_load_bytes_l1_sector_basis": None if loads is None else loads * 32,
                "global_store_bytes_l1_sector_basis": None if stores is None else stores * 32,
                "achieved_active_warps_pct": occupancy,
                "block_size": row.get("Block Size"),
                "grid_size": row.get("Grid Size"),
            }
        )
    return kernels


def read_dict_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_nsys(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Time (%)"))
    return list(csv.DictReader(lines[header_index:]))


def reduction_pct(baseline, fusion):
    return (baseline - fusion) / baseline * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact_dir
    kernels = []
    for mode in ("baseline", "fusion"):
        kernels.extend(summarize_ncu(artifact / f"ncu_{mode}.csv", mode))

    fields = list(kernels[0])
    with (artifact / "ncu_kernel_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kernels)

    aggregate = []
    for mode in ("baseline", "fusion"):
        selected = [row for row in kernels if row["mode"] == mode]
        aggregate.append(
            {
                "mode": mode,
                "kernel_count": len(selected),
                "gpu_time_ns": sum(row["duration_ns"] or 0 for row in selected),
                "global_load_sectors": sum(row["global_load_sectors"] or 0 for row in selected),
                "global_store_sectors": sum(row["global_store_sectors"] or 0 for row in selected),
                "global_load_bytes_l1_sector_basis": sum(
                    row["global_load_bytes_l1_sector_basis"] or 0 for row in selected
                ),
                "global_store_bytes_l1_sector_basis": sum(
                    row["global_store_bytes_l1_sector_basis"] or 0 for row in selected
                ),
            }
        )
    with (artifact / "ncu_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (artifact / "ncu_summary.json").write_text(
        json.dumps({"kernels": kernels, "aggregate": aggregate}, indent=2) + "\n",
        encoding="utf-8",
    )

    benchmark = {row["path"]: row for row in read_dict_rows(artifact / "benchmark_summary.csv")}
    correctness = read_dict_rows(artifact / "correctness.csv")
    ncu = {row["mode"]: row for row in aggregate}
    nsys = {}
    for mode in ("baseline", "fusion"):
        rows = read_nsys(artifact / f"nsys_{mode}_cuda_gpu_kern_sum.csv")
        nsys[mode] = {
            "kernel_count": sum(int(row["Instances"]) for row in rows),
            "gpu_time_ns": sum(float(row["Total Time (ns)"]) for row in rows),
        }

    metrics = [
        ("kernel_count", nsys["baseline"]["kernel_count"], nsys["fusion"]["kernel_count"], "NSYS"),
        ("cuda_event_mean_ms", float(benchmark["baseline"]["mean_ms"]), float(benchmark["fusion"]["mean_ms"]), "CUDA Event"),
        ("cuda_event_median_ms", float(benchmark["baseline"]["median_ms"]), float(benchmark["fusion"]["median_ms"]), "CUDA Event"),
        ("global_load_sectors", ncu["baseline"]["global_load_sectors"], ncu["fusion"]["global_load_sectors"], "NCU L1 global sectors"),
        ("global_store_sectors", ncu["baseline"]["global_store_sectors"], ncu["fusion"]["global_store_sectors"], "NCU L1 global sectors"),
        ("nsys_gpu_time_ms", nsys["baseline"]["gpu_time_ns"] / 1e6, nsys["fusion"]["gpu_time_ns"] / 1e6, "NSYS"),
        ("ncu_gpu_duration_ms", ncu["baseline"]["gpu_time_ns"] / 1e6, ncu["fusion"]["gpu_time_ns"] / 1e6, "NCU diagnostic"),
    ]
    metric_rows = [
        {
            "metric": name,
            "baseline": baseline,
            "fusion": fusion,
            "reduction_pct": reduction_pct(float(baseline), float(fusion)),
            "evidence": evidence,
        }
        for name, baseline, fusion, evidence in metrics
    ]
    with (artifact / "performance_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    correctness_pass = all(row["gate"] == "PASS" for row in correctness)
    checks = [
        ("correctness", correctness_pass, "all three correctness rows PASS"),
        ("kernel_count_reduced", nsys["fusion"]["kernel_count"] < nsys["baseline"]["kernel_count"], "fusion count < baseline count"),
        ("event_mean_at_least_5pct_lower", reduction_pct(float(benchmark["baseline"]["mean_ms"]), float(benchmark["fusion"]["mean_ms"])) >= 5.0, "CUDA Event mean reduction >= 5%"),
        ("event_median_at_least_5pct_lower", reduction_pct(float(benchmark["baseline"]["median_ms"]), float(benchmark["fusion"]["median_ms"])) >= 5.0, "CUDA Event median reduction >= 5%"),
        ("global_load_sectors_reduced", float(ncu["fusion"]["global_load_sectors"]) < float(ncu["baseline"]["global_load_sectors"]), "NCU L1 global-load sectors reduced"),
        ("global_store_sectors_reduced", float(ncu["fusion"]["global_store_sectors"]) < float(ncu["baseline"]["global_store_sectors"]), "NCU L1 global-store sectors reduced"),
        ("nsys_gpu_time_at_least_5pct_lower", reduction_pct(nsys["baseline"]["gpu_time_ns"], nsys["fusion"]["gpu_time_ns"]) >= 5.0, "NSYS GPU kernel time reduction >= 5%"),
    ]
    overall = all(passed for _, passed, _ in checks)
    gate_rows = [
        {"check": name, "result": "PASS" if passed else "FAIL", "criterion": criterion}
        for name, passed, criterion in checks
    ]
    gate_rows.append(
        {
            "check": "overall",
            "result": "FUSION_BENEFIT_PROVEN" if overall else "NO_PROVEN_FUSION_BENEFIT",
            "criterion": "all predeclared checks PASS",
        }
    )
    with (artifact / "gate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0]))
        writer.writeheader()
        writer.writerows(gate_rows)
    (artifact / "gate_summary.json").write_text(
        json.dumps({"checks": gate_rows, "overall": gate_rows[-1]["result"]}, indent=2) + "\n",
        encoding="utf-8",
    )

    experiment_root = Path(__file__).resolve().parent.parent
    manifest_path = artifact / "sha256s.txt"
    manifest_lines = []
    for path in sorted(experiment_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.relative_to(experiment_root).as_posix()}")
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
