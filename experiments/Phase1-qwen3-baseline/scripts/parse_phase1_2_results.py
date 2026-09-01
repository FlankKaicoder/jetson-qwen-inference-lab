#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path


POWER_RE = re.compile(r"\bVDD_IN (\d+)mW/(\d+)mW")
RAM_RE = re.compile(r"\bRAM (\d+)/(\d+)MB")
SWAP_RE = re.compile(r"\bSWAP (\d+)/(\d+)MB")
TEMP_RE = re.compile(r"\b(cpu|soc2|soc0|gpu|tj|soc1)@([0-9.]+)C")


def stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def read_telemetry(path):
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        timestamp, _, payload = line.partition("\t")
        if not payload:
            continue
        power = POWER_RE.search(payload)
        ram = RAM_RE.search(payload)
        swap = SWAP_RE.search(payload)
        temperatures = {name: float(value) for name, value in TEMP_RE.findall(payload)}
        samples.append(
            {
                "timestamp_ns": int(timestamp),
                "vdd_in_w": int(power.group(1)) / 1000.0 if power else None,
                "ram_used_mb": int(ram.group(1)) if ram else None,
                "swap_used_mb": int(swap.group(1)) if swap else None,
                "temperatures": temperatures,
            }
        )
    return samples


def window(samples, start_ns, end_ns):
    return [sample for sample in samples if start_ns <= sample["timestamp_ns"] <= end_ns]


def power_thermal_summary(samples, start_ns, end_ns):
    selected = window(samples, start_ns, end_ns)
    powers = [sample["vdd_in_w"] for sample in selected if sample["vdd_in_w"] is not None]
    peak_temperatures = {}
    for sensor in ("gpu", "cpu", "soc0", "soc1", "soc2", "tj"):
        values = [sample["temperatures"][sensor] for sample in selected if sensor in sample["temperatures"]]
        peak_temperatures[sensor] = max(values) if values else None
    return {
        "samples": len(selected),
        "vdd_in_w": stats(powers),
        "peak_temperatures_c": peak_temperatures,
        "peak_ram_used_mb": max(
            (sample["ram_used_mb"] for sample in selected if sample["ram_used_mb"] is not None),
            default=None,
        ),
        "peak_swap_used_mb": max(
            (sample["swap_used_mb"] for sample in selected if sample["swap_used_mb"] is not None),
            default=None,
        ),
    }


def flatten_trials(result):
    rows = []
    for trial in result["trials"]:
        before = trial["memory_before"]
        after = trial["memory_after"]
        rows.append(
            {
                "isl": trial["isl"],
                "osl": trial["osl"],
                "trial_id": trial["trial_id"],
                "prefill_gpu_ms": trial["prefill_gpu_ms"],
                "prefill_tokens_per_second": trial["prefill_tokens_per_second"],
                "inference_ttft_ms": trial["inference_ttft_ms"],
                "tpot_ms": trial["tpot_ms"],
                "decode_tokens_per_second": trial["decode_tokens_per_second"],
                "e2e_ms": trial["e2e_ms"],
                "e2e_output_tokens_per_second": trial["e2e_output_tokens_per_second"],
                "timing_accounting_error_ms": trial["timing_accounting_error_ms"],
                "cache_growth_valid": trial["cache_growth_valid"],
                "output_token_count": len(trial["fixed_output_token_ids"]),
                "memavailable_before_bytes": before["system"]["MemAvailable"],
                "memavailable_after_bytes": after["system"]["MemAvailable"],
                "swap_used_before_bytes": before["system"]["SwapUsed"],
                "swap_used_after_bytes": after["system"]["SwapUsed"],
                "torch_max_allocated_after_bytes": after["torch_cuda_allocator"]["max_allocated"],
                "torch_max_reserved_after_bytes": after["torch_cuda_allocator"]["max_reserved"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--all-trials", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--power-thermal", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    results = []
    for path in sorted(input_dir.glob("phase1_2_benchmark_isl*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") == "PASS" and result.get("mode") == "benchmark":
            result["_path"] = path
            results.append(result)
    results.sort(key=lambda result: result["shape"]["isl"])
    if [result["shape"]["isl"] for result in results] != [32, 128, 512, 1024]:
        raise RuntimeError("expected one PASS benchmark JSON for ISL 32, 128, 512, and 1024")

    trial_rows = [row for result in results for row in flatten_trials(result)]
    with Path(args.all_trials).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)

    summary_rows = []
    power_rows = []
    for result in results:
        isl = result["shape"]["isl"]
        shape_trials = [row for row in trial_rows if row["isl"] == isl]
        telemetry_path = next(input_dir.glob(f"phase1_2_tegrastats_benchmark_isl{isl}_*.log"))
        telemetry = read_telemetry(telemetry_path)
        idle = power_thermal_summary(
            telemetry, result["loaded_idle_start_ns"], result["loaded_idle_end_ns"]
        )
        active = power_thermal_summary(telemetry, result["formal_start_ns"], result["formal_end_ns"])
        minimum_memavailable = min(
            min(row["memavailable_before_bytes"], row["memavailable_after_bytes"])
            for row in shape_trials
        )
        maximum_swap = max(
            max(row["swap_used_before_bytes"], row["swap_used_after_bytes"])
            for row in shape_trials
        )
        metric = result["summary"]
        summary_rows.append(
            {
                "isl": isl,
                "osl": result["shape"]["osl"],
                "trial_count": len(result["trials"]),
                "prefill_gpu_median_ms": metric["prefill_gpu_ms"]["median"],
                "prefill_gpu_mean_ms": metric["prefill_gpu_ms"]["mean"],
                "prefill_gpu_std_ms": metric["prefill_gpu_ms"]["std"],
                "prefill_gpu_cv_percent": metric["prefill_gpu_ms"]["cv_percent"],
                "prefill_gpu_min_ms": metric["prefill_gpu_ms"]["min"],
                "prefill_gpu_max_ms": metric["prefill_gpu_ms"]["max"],
                "prefill_gpu_p90_ms": metric["prefill_gpu_ms"]["p90"],
                "prefill_tps_median": metric["prefill_tokens_per_second"]["median"],
                "ttft_median_ms": metric["inference_ttft_ms"]["median"],
                "ttft_mean_ms": metric["inference_ttft_ms"]["mean"],
                "ttft_std_ms": metric["inference_ttft_ms"]["std"],
                "ttft_cv_percent": metric["inference_ttft_ms"]["cv_percent"],
                "ttft_min_ms": metric["inference_ttft_ms"]["min"],
                "ttft_max_ms": metric["inference_ttft_ms"]["max"],
                "ttft_p90_ms": metric["inference_ttft_ms"]["p90"],
                "tpot_median_ms": metric["tpot_ms"]["median"],
                "tpot_mean_ms": metric["tpot_ms"]["mean"],
                "tpot_std_ms": metric["tpot_ms"]["std"],
                "tpot_cv_percent": metric["tpot_ms"]["cv_percent"],
                "tpot_min_ms": metric["tpot_ms"]["min"],
                "tpot_max_ms": metric["tpot_ms"]["max"],
                "tpot_p90_ms": metric["tpot_ms"]["p90"],
                "decode_tps_median": metric["decode_tokens_per_second"]["median"],
                "decode_tps_mean": metric["decode_tokens_per_second"]["mean"],
                "decode_tps_std": metric["decode_tokens_per_second"]["std"],
                "decode_tps_cv_percent": metric["decode_tokens_per_second"]["cv_percent"],
                "decode_tps_min": metric["decode_tokens_per_second"]["min"],
                "decode_tps_max": metric["decode_tokens_per_second"]["max"],
                "decode_tps_p90": metric["decode_tokens_per_second"]["p90"],
                "e2e_median_ms": metric["e2e_ms"]["median"],
                "e2e_mean_ms": metric["e2e_ms"]["mean"],
                "e2e_std_ms": metric["e2e_ms"]["std"],
                "e2e_cv_percent": metric["e2e_ms"]["cv_percent"],
                "e2e_min_ms": metric["e2e_ms"]["min"],
                "e2e_max_ms": metric["e2e_ms"]["max"],
                "e2e_p90_ms": metric["e2e_ms"]["p90"],
                "minimum_memavailable_bytes": minimum_memavailable,
                "maximum_swap_used_bytes": maximum_swap,
                "cache_all_valid": all(row["cache_growth_valid"] for row in shape_trials),
                "osl_all_exact": all(row["output_token_count"] == 32 for row in shape_trials),
                "stable": result["stability"]["stable"],
                "power_mode_consistent": result["power_mode_start"] == result["power_mode_end"],
            }
        )
        power_rows.append(
            {
                "isl": isl,
                "rail": "VDD_IN",
                "idle_samples": idle["samples"],
                "idle_mean_w": idle["vdd_in_w"]["mean"],
                "idle_median_w": idle["vdd_in_w"]["median"],
                "active_samples": active["samples"],
                "active_mean_w": active["vdd_in_w"]["mean"],
                "active_median_w": active["vdd_in_w"]["median"],
                "active_peak_w": active["vdd_in_w"]["max"],
                "active_minus_idle_mean_w": (
                    active["vdd_in_w"]["mean"] - idle["vdd_in_w"]["mean"]
                    if active["vdd_in_w"]["mean"] is not None and idle["vdd_in_w"]["mean"] is not None
                    else None
                ),
                "peak_gpu_temp_c": active["peak_temperatures_c"]["gpu"],
                "peak_cpu_temp_c": active["peak_temperatures_c"]["cpu"],
                "peak_soc_temp_c": max(
                    value
                    for name, value in active["peak_temperatures_c"].items()
                    if name.startswith("soc") and value is not None
                ),
                "peak_tj_temp_c": active["peak_temperatures_c"]["tj"],
                "peak_ram_used_mb": active["peak_ram_used_mb"],
                "peak_swap_used_mb": active["peak_swap_used_mb"],
                "scope": "board-level telemetry; not process-exclusive",
            }
        )

    for path, rows in ((Path(args.summary), summary_rows), (Path(args.power_thermal), power_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
