#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict


LAUNCH_API_NAMES = {
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cudaLaunchKernel_v7000",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def layer_index(name):
    match = re.fullmatch(r"/up_proj(?:_(\d+))?/MatMul", name)
    if not match:
        return None
    return int(match.group(1) or 0)


def mean(values):
    return sum(values) / len(values) if values else None


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile95(values):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def aggregate_rows(rows):
    nvtx_durations = [row["nvtx_duration_ns"] for row in rows]
    kernel_durations = [row["kernel_total_duration_ns"] for row in rows]
    launch_api_durations = [row["kernel_launch_api_duration_ns"] for row in rows]
    all_runtime_durations = [row["all_runtime_duration_ns"] for row in rows]
    nvtx_minus_kernel = [
        row["nvtx_duration_ns"] - row["kernel_total_duration_ns"]
        for row in rows
    ]
    return {
        "range_count": len(rows),
        "kernel_count": sum(row["kernel_count"] for row in rows),
        "nvtx_duration_ns": {
            "total": sum(nvtx_durations),
            "mean": mean(nvtx_durations),
            "median": median(nvtx_durations),
            "min": min(nvtx_durations),
            "max": max(nvtx_durations),
            "p95": percentile95(nvtx_durations),
        },
        "kernel_duration_ns": {
            "total": sum(kernel_durations),
            "mean_per_range": mean(kernel_durations),
            "median_per_range": median(kernel_durations),
            "min_per_range": min(kernel_durations),
            "max_per_range": max(kernel_durations),
            "p95_per_range": percentile95(kernel_durations),
        },
        "kernel_launch_api_duration_ns": {
            "total": sum(launch_api_durations),
            "mean_per_range": mean(launch_api_durations),
            "median_per_range": median(launch_api_durations),
        },
        "all_runtime_api_duration_ns": {
            "total": sum(all_runtime_durations),
            "mean_per_range": mean(all_runtime_durations),
            "median_per_range": median(all_runtime_durations),
        },
        "nvtx_minus_kernel_ns": {
            "total": sum(nvtx_minus_kernel),
            "mean_per_range": mean(nvtx_minus_kernel),
            "median_per_range": median(nvtx_minus_kernel),
            "min_per_range": min(nvtx_minus_kernel),
            "max_per_range": max(nvtx_minus_kernel),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    args = parser.parse_args()

    connection = sqlite3.connect(
        f"file:{args.sqlite}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row

    nvtx_rows = connection.execute(
        """
        SELECT ne.start, ne.end, ne.globalTid, s.value AS layer_name
        FROM NVTX_EVENTS ne
        JOIN StringIds s ON ne.textId = s.id
        WHERE s.value LIKE '/up_proj%/MatMul'
        ORDER BY ne.start, s.value
        """
    ).fetchall()

    if len(nvtx_rows) != 196:
        raise RuntimeError(f"Expected 196 up_proj NVTX ranges, got {len(nvtx_rows)}")

    # The ordered trace has exactly 28 logical layers per invocation.
    for invocation_index, row in enumerate(nvtx_rows, start=1):
        expected_layer_index = (invocation_index - 1) % 28
        observed_layer_index = layer_index(row["layer_name"])
        if observed_layer_index != expected_layer_index:
            raise RuntimeError(
                "Layer order mismatch at invocation "
                f"{(invocation_index - 1) // 28 + 1}: expected "
                f"{expected_layer_index}, got {observed_layer_index}"
            )

    records = []
    kernel_rows_for_csv = []
    family_totals = defaultdict(lambda: {"count": 0, "duration_ns": 0})
    for invocation_index, nvtx in enumerate(nvtx_rows, start=1):
        runtime_rows = connection.execute(
            """
            SELECT r.start, r.end, r.correlationId, s.value AS api_name
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            WHERE r.globalTid = ?
              AND r.start >= ?
              AND r.end <= ?
            ORDER BY r.start
            """,
            (nvtx["globalTid"], nvtx["start"], nvtx["end"]),
        ).fetchall()

        kernel_by_correlation = {}
        for runtime in runtime_rows:
            if runtime["correlationId"] is None:
                continue
            kernel = connection.execute(
                """
                SELECT k.start, k.end, k.correlationId, k.streamId,
                       demangled.value AS demangled_name,
                       short.value AS short_name
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                LEFT JOIN StringIds demangled ON k.demangledName = demangled.id
                LEFT JOIN StringIds short ON k.shortName = short.id
                WHERE k.correlationId = ?
                """,
                (runtime["correlationId"],),
            ).fetchall()
            if len(kernel) > 1:
                raise RuntimeError(
                    f"Multiple kernels for correlationId {runtime['correlationId']}"
                )
            if len(kernel) == 1:
                kernel_by_correlation[runtime["correlationId"]] = kernel[0]

        kernels = []
        for runtime in runtime_rows:
            kernel = kernel_by_correlation.get(runtime["correlationId"])
            if kernel is None:
                continue
            kernel_name = kernel["demangled_name"] or kernel["short_name"] or "UNKNOWN"
            duration = kernel["end"] - kernel["start"]
            overlap = max(
                0,
                min(kernel["end"], nvtx["end"])
                - max(kernel["start"], nvtx["start"]),
            )
            after_end = max(0, kernel["end"] - nvtx["end"])
            before_start = max(0, nvtx["start"] - kernel["start"])
            kernel_record = {
                "correlation_id": kernel["correlationId"],
                "kernel_name": kernel_name,
                "kernel_family": (
                    "h16816" if "h16816gemm" in kernel_name
                    else "sm80_xmma_gemm" if "sm80_xmma_gemm" in kernel_name
                    else "other"
                ),
                "start_ns": kernel["start"],
                "end_ns": kernel["end"],
                "duration_ns": duration,
                "overlap_with_nvtx_ns": overlap,
                "after_nvtx_end_ns": after_end,
                "before_nvtx_start_ns": before_start,
                "stream_id": kernel["streamId"],
                "launch_api": runtime["api_name"],
            }
            kernels.append(kernel_record)
            family_totals[kernel_record["kernel_family"]]["count"] += 1
            family_totals[kernel_record["kernel_family"]]["duration_ns"] += duration
            kernel_rows_for_csv.append({
                "invocation_index": (invocation_index - 1) // 28 + 1,
                "layer_index": layer_index(nvtx["layer_name"]),
                "layer_name": nvtx["layer_name"],
                "nvtx_start_ns": nvtx["start"],
                "nvtx_end_ns": nvtx["end"],
                "nvtx_duration_ns": nvtx["end"] - nvtx["start"],
                "correlation_id": kernel["correlationId"],
                "kernel_family": kernel_record["kernel_family"],
                "kernel_name": kernel_name,
                "kernel_start_ns": kernel["start"],
                "kernel_end_ns": kernel["end"],
                "kernel_duration_ns": duration,
                "kernel_overlap_with_nvtx_ns": overlap,
                "kernel_after_nvtx_end_ns": after_end,
                "kernel_before_nvtx_start_ns": before_start,
                "kernel_launch_api": runtime["api_name"],
                "kernel_launch_api_duration_ns": runtime["end"] - runtime["start"],
            })

        launch_rows = [
            runtime for runtime in runtime_rows
            if runtime["api_name"] in LAUNCH_API_NAMES
        ]
        records.append({
            "invocation_index": (invocation_index - 1) // 28 + 1,
            "layer_index": layer_index(nvtx["layer_name"]),
            "layer_name": nvtx["layer_name"],
            "nvtx_start_ns": nvtx["start"],
            "nvtx_end_ns": nvtx["end"],
            "nvtx_duration_ns": nvtx["end"] - nvtx["start"],
            "kernel_count": len(kernels),
            "kernel_total_duration_ns": sum(
                kernel["duration_ns"] for kernel in kernels
            ),
            "kernel_launch_api_count": len(launch_rows),
            "kernel_launch_api_duration_ns": sum(
                runtime["end"] - runtime["start"] for runtime in launch_rows
            ),
            "all_runtime_api_count": len(runtime_rows),
            "all_runtime_duration_ns": sum(
                runtime["end"] - runtime["start"] for runtime in runtime_rows
            ),
            "kernels": kernels,
        })

    all_records = records
    steady_state_records = [
        record for record in records if record["invocation_index"] != 1
    ]
    output = {
        "phase": "Phase5A Step3 TensorRT GEMM Boundary Reconciliation",
        "mode": "READ_ONLY_SQLITE_REQUERY_OF_EXISTING_PHASE4F_TRACE",
        "source_sqlite_path": args.sqlite,
        "source_sqlite_sha256": file_sha256(args.sqlite),
        "time_unit": "ns",
        "definitions": {
            "kernel_total_duration_ns": (
                "Sum of CUPTI kernel event end-start durations correlated to "
                "launch APIs contained in the NVTX range."
            ),
            "kernel_launch_api_duration_ns": (
                "Sum of host runtime API durations for cuLaunchKernel, "
                "cuLaunchKernelEx and cudaLaunchKernel_v7000 events contained "
                "in the NVTX range."
            ),
            "nvtx_minus_kernel_ns": (
                "Arithmetic residual only; CPU host API time and asynchronous "
                "GPU kernel time may overlap and must not be treated as a "
                "disjoint component."
            ),
        },
        "coverage": {
            "unique_layer_names": len({row["layer_name"] for row in records}),
            "nvtx_range_instances": len(records),
            "layer_names_per_invocation": 28,
            "invocations": len(records) // 28,
            "kernel_events": sum(record["kernel_count"] for record in records),
        },
        "family_totals": dict(family_totals),
        "aggregate_all_observed_ranges": aggregate_rows(all_records),
        "aggregate_excluding_first_invocation": aggregate_rows(steady_state_records),
        "per_nvtx_range": records,
    }

    with open(args.json_out, "w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")

    csv_fields = list(kernel_rows_for_csv[0].keys())
    with open(args.csv_out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(kernel_rows_for_csv)

    connection.close()


if __name__ == "__main__":
    main()
