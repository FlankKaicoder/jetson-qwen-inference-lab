#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path


ATTENTION_PATTERN = re.compile(
    r"(attn|attention|fmha|flash|mem_efficient|scaled_dot_product)",
    re.IGNORECASE,
)
GEMM_PATTERN = re.compile(r"(gemm|matmul|cutlass|nvjet|cublas)", re.IGNORECASE)
MEMORY_PATTERN = re.compile(
    r"("
    r"elementwise|copy|memcpy|memset|fill|arange|index|gather|scatter|"
    r"embedding|layer_norm|rms|rmsnorm|norm|contiguous|reduce|softmax|"
    r"cast|transform"
    r")",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def classify_kernel(name: str) -> str:
    if ATTENTION_PATTERN.search(name):
        return "attention"
    if GEMM_PATTERN.search(name):
        return "gemm"
    if MEMORY_PATTERN.search(name):
        return "memory_like"
    return "other"


def aggregate_kernels(
    connection: sqlite3.Connection,
    start: int,
    end: int,
    exclude_ranges: list[tuple[int, int]] | None = None,
) -> dict:
    query = """
        SELECT
            k.start,
            k.end,
            COALESCE(short.value, demangled.value, 'UNKNOWN_NAME') AS kernel_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        LEFT JOIN StringIds AS short ON k.shortName = short.id
        LEFT JOIN StringIds AS demangled ON k.demangledName = demangled.id
        WHERE k.start >= ? AND k.end <= ?
    """
    totals: dict[str, dict] = {}
    family_values: dict[str, list[float]] = {
        "gemm": [],
        "attention": [],
        "memory_like": [],
        "other": [],
    }
    total_duration_ns = 0
    count = 0
    query_params = [start, end]
    for exclude_start, exclude_end in exclude_ranges or []:
        query += " AND NOT (k.start < ? AND k.end > ?)"
        query_params.extend([exclude_end, exclude_start])
    query += " ORDER BY k.start"
    for kernel_start, kernel_end, raw_name in connection.execute(query, query_params):
        name = raw_name if isinstance(raw_name, str) else str(raw_name)
        duration_ns = int(kernel_end) - int(kernel_start)
        total_duration_ns += duration_ns
        count += 1
        family = classify_kernel(name)
        family_values[family].append(duration_ns)
        if name not in totals:
            totals[name] = {
                "kernel_name": name,
                "family": family,
                "count": 0,
                "duration_ns": 0,
            }
        totals[name]["count"] += 1
        totals[name]["duration_ns"] += duration_ns

    family_summary = {}
    for family, values in family_values.items():
        family_summary[family] = {
            "kernel_count": len(values),
            "duration_ms": sum(values) / 1_000_000.0,
            "share_of_total_kernel_time": (
                sum(values) / total_duration_ns if total_duration_ns else None
            ),
            "mean_per_launch_ms": (
                statistics.mean(values) / 1_000_000.0 if values else None
            ),
            "median_per_launch_ms": (
                statistics.median(values) / 1_000_000.0 if values else None
            ),
        }

    ranked = sorted(
        totals.values(),
        key=lambda item: item["duration_ns"],
        reverse=True,
    )
    top_kernels = []
    for row in ranked[:30]:
        duration_ms = row["duration_ns"] / 1_000_000.0
        top_kernels.append(
            {
                "kernel_name": row["kernel_name"],
                "family": row["family"],
                "count": row["count"],
                "duration_ms": duration_ms,
                "share_of_total_kernel_time": (
                    row["duration_ns"] / total_duration_ns
                    if total_duration_ns
                    else None
                ),
                "mean_per_launch_ms": duration_ms / row["count"],
            }
        )
    return {
        "kernel_count": count,
        "total_kernel_duration_ms": total_duration_ns / 1_000_000.0,
        "family_attribution": family_summary,
        "top_kernels": top_kernels,
    }


def memory_activity(connection: sqlite3.Connection, start: int, end: int) -> dict:
    result = {}
    tables = sqlite_tables(connection)
    for table in ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        if table not in tables:
            result[table] = {"available": False}
            continue
        rows = list(
            connection.execute(
                f"SELECT start, end, bytes FROM {table} WHERE start >= ? AND end <= ?",
                (start, end),
            )
        )
        durations = [int(row[1]) - int(row[0]) for row in rows]
        result[table] = {
            "available": True,
            "count": len(rows),
            "total_bytes": sum(int(row[2]) for row in rows),
            "total_duration_ms": sum(durations) / 1_000_000.0,
        }
    return result


def nvtx_ranges(connection: sqlite3.Connection) -> dict:
    query = """
        SELECT start, end, text
        FROM NVTX_EVENTS
        WHERE start >= ? AND end <= ?
        ORDER BY start
    """
    ranges = []
    for start, end, text in connection.execute(query, (0, 9_999_999_999_999_999)):
        if isinstance(text, str) and text.startswith("B2_"):
            ranges.append({"start": start, "end": end, "text": text})
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--nsys-report")
    parser.add_argument("--latency-result", required=True)
    parser.add_argument("--profile-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite).resolve()
    nsys_report_path = Path(args.nsys_report).resolve() if args.nsys_report else None
    latency_path = Path(args.latency_result).resolve()
    profile_path = Path(args.profile_result).resolve()
    output_path = Path(args.output).resolve()

    latency_result = json.loads(latency_path.read_text())
    profile_result = json.loads(profile_path.read_text())
    if not latency_result.get("success"):
        raise RuntimeError("latency result is not successful")
    if not profile_result.get("success"):
        raise RuntimeError("profile result is not successful")
    if len(profile_result["trials"]["measured"]) != 1:
        raise RuntimeError("expected exactly one profiled measured generation")

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    generation_ranges = list(
        connection.execute(
            "SELECT start, end, text FROM NVTX_EVENTS WHERE text = ? ORDER BY start",
            ("B2_profile_generation",),
        )
    )
    if len(generation_ranges) != 1:
        raise RuntimeError(
            f"expected one B2_profile_generation range, got {len(generation_ranges)}"
        )
    generation_start, generation_end, _ = generation_ranges[0]
    if generation_end is None:
        raise RuntimeError("B2_profile_generation range has no end timestamp")

    prefill_rows = list(
        connection.execute(
            """
            SELECT start, end FROM NVTX_EVENTS
            WHERE text = 'B2_profile_prefill' AND start >= ? AND end <= ?
            ORDER BY start
            """,
            (generation_start, generation_end),
        )
    )
    decode_rows = list(
        connection.execute(
            """
            SELECT start, end FROM NVTX_EVENTS
            WHERE text = 'B2_profile_decode' AND start >= ? AND end <= ?
            ORDER BY start
            """,
            (generation_start, generation_end),
        )
    )
    if len(prefill_rows) != 1:
        raise RuntimeError(f"expected one prefill range, got {len(prefill_rows)}")
    if len(decode_rows) != 1:
        raise RuntimeError(f"expected one decode range, got {len(decode_rows)}")
    visual_rows = list(
        connection.execute(
            """
            SELECT start, end FROM NVTX_EVENTS
            WHERE text = 'B2_tensorrt_visual' AND start >= ? AND end <= ?
            ORDER BY start
            """,
            (generation_start, generation_end),
        )
    )
    if len(visual_rows) != 1:
        raise RuntimeError(
            f"expected one TensorRT visual range, got {len(visual_rows)}"
        )

    all_ranges = nvtx_ranges(connection)
    generation_analysis = aggregate_kernels(
        connection,
        generation_start,
        generation_end,
    )
    generation_analysis["memory_activity"] = memory_activity(
        connection,
        generation_start,
        generation_end,
    )
    prefill_analysis = aggregate_kernels(
        connection,
        prefill_rows[0][0],
        prefill_rows[0][1],
    )
    visual_analysis = aggregate_kernels(
        connection,
        visual_rows[0][0],
        visual_rows[0][1],
    )
    prefill_excluding_visual_analysis = aggregate_kernels(
        connection,
        prefill_rows[0][0],
        prefill_rows[0][1],
        exclude_ranges=[(visual_rows[0][0], visual_rows[0][1])],
    )
    decode_analysis = aggregate_kernels(
        connection,
        decode_rows[0][0],
        decode_rows[0][1],
    )

    step_analysis = []
    step_rows = list(
        connection.execute(
            """
            SELECT start, end, text FROM NVTX_EVENTS
            WHERE text LIKE 'B2_profile_decode_step_%'
              AND start >= ? AND end <= ?
            ORDER BY start
            """,
            (generation_start, generation_end),
        )
    )
    for start, end, text in step_rows:
        summary = aggregate_kernels(connection, start, end)
        summary["nvtx_text"] = text
        step_analysis.append(summary)
    if len(step_analysis) != 15:
        raise RuntimeError(
            f"expected 15 decode-step NVTX ranges, got {len(step_analysis)}"
        )

    connection.close()
    analysis = {
        "phase": "Phase 9.3-B2",
        "title": "Nsight Systems decoder-side CUDA attribution",
        "analysis_finished_utc": utc_now(),
        "input_artifacts": {
            "sqlite": {
                "path": str(sqlite_path),
                "size_bytes": sqlite_path.stat().st_size,
                "sha256": sha256_file(sqlite_path),
            },
            "nsys_report": {
                "path": str(nsys_report_path) if nsys_report_path else None,
                "size_bytes": (
                    nsys_report_path.stat().st_size if nsys_report_path else None
                ),
                "sha256": sha256_file(nsys_report_path) if nsys_report_path else None,
            },
            "latency_result": {
                "path": str(latency_path),
                "sha256": sha256_file(latency_path),
            },
            "profile_result": {
                "path": str(profile_path),
                "sha256": sha256_file(profile_path),
            },
        },
        "range_validation": {
            "profiled_generation_count": len(generation_ranges),
            "prefill_range_count": len(prefill_rows),
            "visual_range_count": len(visual_rows),
            "decode_range_count": len(decode_rows),
            "decode_step_range_count": len(step_analysis),
            "all_b2_nvtx_range_count": len(all_ranges),
        },
        "profiled_generation": {
            "profile_result_summary": profile_result["summary"],
            "profile_result_stage_attribution": profile_result["stage_attribution"],
            "nvtx_ranges": all_ranges,
            "kernel_attribution": generation_analysis,
            "prefill_kernel_attribution": prefill_analysis,
            "visual_kernel_attribution": visual_analysis,
            "prefill_excluding_visual_kernel_attribution": (
                prefill_excluding_visual_analysis
            ),
            "decode_kernel_attribution": decode_analysis,
            "decode_step_kernel_attribution": step_analysis,
        },
        "latency_run_summary": latency_result["summary"],
        "latency_run_stage_attribution": latency_result["stage_attribution"],
        "attribution_caveats": {
            "kernel_family_method": "kernel-name heuristic",
            "memory_contribution": "memory-like kernel duration and memcpy/memset activity only",
            "dram_counters": "UNKNOWN",
            "achieved_bandwidth": "UNKNOWN",
            "host_launch_gaps": "not attributed to kernels",
        },
    }
    output_path.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis["range_validation"], indent=2))
    print(json.dumps(generation_analysis["family_attribution"], indent=2))


if __name__ == "__main__":
    main()
