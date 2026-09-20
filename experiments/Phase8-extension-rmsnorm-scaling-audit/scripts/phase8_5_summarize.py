#!/usr/bin/env python3
"""Reduce Phase 8.5 evidence JSON files into compact CSV summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path


KINDS = ("prefill", "decode")
SCALES = (0, 1, 4, 8, 28, 56)


def evidence_root() -> Path:
    return (Path(__file__).resolve().parents[1] /
            "artifacts/phase8_5_20260920T075327Z/remote/artifacts_local/evidence")


def output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts/phase8_5_20260920T075327Z"


def selected_name(prefix: str, kind: str, scale: int) -> str:
    if kind == "decode" and scale == 56 and prefix != "build":
        if prefix == "compare":
            return "compare_decode_scale56_opt1_attempt2.json"
        return f"{prefix}_decode_scale56_opt1.json"
    return f"{prefix}_{kind}_scale{scale}.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    evidence = evidence_root()
    out = output_root()

    hidden_rows = []
    target_rows = []
    numerical_rows = []
    performance_rows = []
    generation_rows = []
    nsys_rows = []

    for kind in KINDS:
        for scale in SCALES:
            opt1 = kind == "decode" and scale == 56
            compare = read_json(evidence / selected_name("compare", kind, scale))
            measure = read_json(evidence / selected_name("measure", kind, scale))
            generation = read_json(evidence / selected_name("generation", kind, scale))
            nsys = read_json(evidence / selected_name("nsys_profile", kind, scale))

            for name, metric in compare["hidden_metrics"].items():
                layer = int(name.removeprefix("hidden_l"))
                hidden_rows.append({
                    "kind": kind, "scale": scale, "layer": layer,
                    "relative_l2": metric["relative_l2"],
                    "max_abs": metric["max_abs"],
                    "finite_ratio": metric["finite_ratio"],
                    "engine_build": "opt1" if opt1 else "default_or_opt0",
                })
            for name, metric in compare["replaced_norm_metrics"].items():
                target_rows.append({
                    "kind": kind, "scale": scale, "output": name,
                    "relative_l2": metric["relative_l2"],
                    "max_abs": metric["max_abs"],
                    "finite_ratio": metric["finite_ratio"],
                    "engine_build": "opt1" if opt1 else "default_or_opt0",
                })
            final = compare["hidden_metrics"].get("hidden_l27", {})
            numerical_rows.append({
                "kind": kind, "scale": scale,
                "engine_build": "opt1" if opt1 else "default",
                "common_output_count": compare["common_output_count"],
                "all_outputs_finite": compare["all_outputs_finite"],
                "replaced_norm_count": len(compare["replaced_norm_metrics"]),
                "replaced_norm_max_relative_l2": compare["replaced_norm_max_relative_l2"],
                "hidden_l27_relative_l2": final.get("relative_l2"),
                "hidden_l27_max_abs": final.get("max_abs"),
                "gate_replaced_norm_pass": compare["gate_replaced_norm_pass"],
                "gate_hidden_l27_pass": compare["gate_hidden_l27_pass"],
            })
            performance_rows.append({
                "kind": kind, "scale": scale,
                "engine_build": "opt1" if opt1 else "default",
                "warmup": measure["warmup"], "repetitions": measure["repetitions"],
                "mean_ms": measure["mean_ms"], "min_ms": measure["min_ms"],
                "max_ms": measure["max_ms"], "timing_boundary": measure["timing_boundary"],
                "gpu_kernel_count": nsys["gpu_kernel_count"],
                "gpu_kernel_total_time_ns": nsys["gpu_kernel_total_time"],
                "cuda_launch_count": nsys["cuda_launch_count"],
            })
            generation_rows.append({
                "kind": kind, "scale": scale,
                "engine_build": "opt1" if opt1 else "default",
                "match": generation["match"],
                "baseline_tokens": ";".join(map(str, generation["baseline_tokens"])),
                "candidate_tokens": ";".join(map(str, generation["candidate_tokens"])),
            })
            nsys_rows.append({
                "kind": kind, "scale": scale,
                "engine_build": "opt1" if opt1 else "default",
                "gpu_kernel_count": nsys["gpu_kernel_count"],
                "gpu_kernel_total_time_ns": nsys["gpu_kernel_total_time"],
                "gpu_kernel_distinct_names": nsys["gpu_kernel_distinct_names"],
                "cuda_launch_count": nsys["cuda_launch_count"],
            })

    write_csv(out / "summary_hidden_relative_l2.csv", [
        "kind", "scale", "layer", "relative_l2", "max_abs", "finite_ratio", "engine_build"], hidden_rows)
    write_csv(out / "summary_replaced_norm_relative_l2.csv", [
        "kind", "scale", "output", "relative_l2", "max_abs", "finite_ratio", "engine_build"], target_rows)
    write_csv(out / "summary_numerical.csv", list(numerical_rows[0]), numerical_rows)
    write_csv(out / "summary_performance.csv", list(performance_rows[0]), performance_rows)
    write_csv(out / "summary_generation.csv", list(generation_rows[0]), generation_rows)
    write_csv(out / "summary_nsys.csv", list(nsys_rows[0]), nsys_rows)


if __name__ == "__main__":
    main()
