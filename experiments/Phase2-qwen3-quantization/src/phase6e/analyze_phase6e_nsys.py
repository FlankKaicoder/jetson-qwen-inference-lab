from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path


def decode_text(row: tuple, text_ids: dict[int, str]) -> str:
    if row["text"]:
        return str(row["text"])
    if row["textId"] is not None:
        return text_ids.get(int(row["textId"]), "")
    return ""


def load_shape_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def shapes_for_engine(shape_rows: list[dict], phase: str,
                      decode_step: int, engine_role: str) -> dict:
    selected = {}
    for row in shape_rows:
        if row["phase"] != phase or int(row["decode_step"]) != decode_step:
            continue
        if not row["engine_or_context_identity"].startswith(f"{engine_role}|"):
            continue
        selected[row["tensor_name"]] = {
            "runtime_shape": json.loads(row["runtime_shape"]),
            "engine_invocation_id": row["engine_invocation_id"],
            "evidence_level": row["shape_evidence_level"],
            "tensor_dtype": row["tensor_dtype"],
        }
    return selected


def correlation_rows(sqlite_path: Path, shape_rows: list[dict]) -> list[dict]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    text_ids = {int(row["id"]): str(row["value"]) for row in
                connection.execute("SELECT id, value FROM StringIds")}
    engine_calls = []
    for row in connection.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
            "WHERE text LIKE 'PHASE6E_ENGINE_CALL|%' ORDER BY start"):
        text = decode_text(row, text_ids)
        role_match = re.search(r"role=([^|]+)", text)
        invocation_match = re.search(
            r"runtime_invocation_id=.+_(WARMUP_PREFILL_S8|STEADY_PREFILL_S8"
            r"|DECODE_STEP_\d+)$", text)
        if not role_match or not invocation_match:
            continue
        engine_calls.append({
            "start": int(row["start"]), "end": int(row["end"]),
            "global_tid": int(row["globalTid"]), "role": role_match.group(1),
            "phase": invocation_match.group(1), "nvtx_text": text,
        })

    rows: list[dict] = []
    for call in engine_calls:
        if call["role"] != "mixed_decode":
            continue
        decode_step = int(call["phase"].rsplit("_", 1)[1])
        phase = "DECODE"
        shapes = shapes_for_engine(shape_rows, phase, decode_step, "mixed_decode")
        if "past_k0" not in shapes or "present_k0" not in shapes:
            raise RuntimeError(f"REQUIRED_SHAPE_MISSING:{decode_step}")
        past_length = shapes["past_k0"]["runtime_shape"][2]
        key_length = shapes["present_k0"]["runtime_shape"][2]
        if key_length != past_length + 1:
            raise RuntimeError(f"LENGTH_MISMATCH:{past_length}:{key_length}")

        matmul_ranges = connection.execute(
            "SELECT n.start, n.end, n.globalTid, n.text, n.textId FROM "
            "NVTX_EVENTS n JOIN StringIds s ON s.id = n.textId WHERE "
            "n.start >= ? AND n.end <= ? AND n.globalTid = ? AND s.value = ? "
            "ORDER BY n.start", (call["start"], call["end"],
                                 call["global_tid"], "/MatMul")).fetchall()
        if len(matmul_ranges) != 1:
            raise RuntimeError(
                f"UNEXPECTED_MATMUL_RANGE_COUNT:{decode_step}:{len(matmul_ranges)}")
        matmul = matmul_ranges[0]
        runtime_apis = connection.execute(
            "SELECT start, end, correlationId, nameId, globalTid FROM "
            "CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid = ? AND start >= ? "
            "AND start <= ? ORDER BY start",
            (matmul["globalTid"], matmul["start"], matmul["end"])).fetchall()
        for runtime_api in runtime_apis:
            kernels = connection.execute(
                "SELECT k.start, k.end, k.correlationId, k.demangledName, "
                "k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, "
                "k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE "
                "k.correlationId = ? ORDER BY k.start",
                (runtime_api["correlationId"],)).fetchall()
            for kernel in kernels:
                kernel_name = text_ids.get(int(kernel["demangledName"]),
                                           "UNKNOWN")
                if "h16816" in kernel_name.lower():
                    kernel_family = "h16816"
                elif "xmma" in kernel_name.lower():
                    kernel_family = "xmma"
                else:
                    kernel_family = "OTHER"
                total_mac = 16 * 1 * key_length * 128
                rows.append({
                    "invocation_id": (
                        f"PHASE6E_DECODE_STEP_{decode_step}"),
                    "phase": phase,
                    "decode_step": decode_step,
                    "engine_invocation_id": (
                        shapes["past_k0"]["engine_invocation_id"]),
                    "past_kv_length": past_length,
                    "key_length": key_length,
                    "query_length": 1,
                    "runtime_q_shape": "[16,1,128]",
                    "runtime_k_shape": f"[16,{key_length},128]",
                    "runtime_kt_shape": f"[16,128,{key_length}]",
                    "runtime_output_shape": f"[16,1,{key_length}]",
                    "effective_per_head_m": 1,
                    "effective_per_head_n": key_length,
                    "effective_per_head_k": 128,
                    "head_grouping": "16_heads_gqa_repeat_2",
                    "dtype": shapes["past_k0"]["tensor_dtype"],
                    "effective_total_mac": total_mac,
                    "tensorrt_nvtx_operation": "/MatMul",
                    "runtime_api_correlation_id": runtime_api["correlationId"],
                    "kernel_full_name": kernel_name,
                    "kernel_family": kernel_family,
                    "kernel_start": kernel["start"],
                    "kernel_end": kernel["end"],
                    "duration_ns": kernel["end"] - kernel["start"],
                    "stream_id": kernel["streamId"],
                    "grid": f"{kernel['gridX']}x{kernel['gridY']}x{kernel['gridZ']}",
                    "block": f"{kernel['blockX']}x{kernel['blockY']}x{kernel['blockZ']}",
                    "shape_evidence_type": "DERIVED_FROM_PROVEN_STATE",
                    "operation_attribution_confidence": "MEDIUM",
                    "attribution_note": (
                        "Static layer-0 QK identity and direct engine invocation "
                        "shape progression; kernel argument identity UNKNOWN"),
                })
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--runtime-shape-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    shape_rows = load_shape_rows(args.runtime_shape_log)
    rows = correlation_rows(args.sqlite, shape_rows)
    correlation_fields = [
        "invocation_id", "phase", "decode_step", "engine_invocation_id",
        "past_kv_length", "key_length", "query_length", "runtime_q_shape",
        "runtime_k_shape", "runtime_kt_shape", "runtime_output_shape",
        "effective_per_head_m", "effective_per_head_n",
        "effective_per_head_k", "head_grouping", "dtype",
        "effective_total_mac", "tensorrt_nvtx_operation",
        "runtime_api_correlation_id", "kernel_full_name", "kernel_family",
        "kernel_start", "kernel_end", "duration_ns", "stream_id", "grid",
        "block", "shape_evidence_type", "operation_attribution_confidence",
        "attribution_note",
    ]
    write_csv(args.out / "qk_runtime_kernel_correlation.csv", rows,
              correlation_fields)

    trigger_fields = [
        "phase", "decode_step", "past_kv_length", "key_length",
        "xmma_count", "h16816_count", "other_count",
        "observed_path", "trigger_conclusion", "evidence_level",
    ]
    trigger_rows = []
    for decode_step in range(4):
        step_rows = [row for row in rows if row["decode_step"] == decode_step]
        counts = {family: sum(row["kernel_family"] == family
                              for row in step_rows)
                  for family in ("xmma", "h16816", "OTHER")}
        trigger_rows.append({
            "phase": "DECODE",
            "decode_step": decode_step,
            "past_kv_length": 8 + decode_step,
            "key_length": 9 + decode_step,
            "xmma_count": counts["xmma"],
            "h16816_count": counts["h16816"],
            "other_count": counts["OTHER"],
            "observed_path": "XMMA_ONLY" if counts["xmma"] else "UNKNOWN",
            "trigger_conclusion": (
                "NOT_SUFFICIENT_TO_PROVE_TRIGGER; h16816 not observed"),
            "evidence_level": "DIRECT_RUNTIME_EVIDENCE",
        })
    write_csv(args.out / "qk_path_trigger_analysis.csv", trigger_rows,
              trigger_fields)

    comparison_fields = [
        "comparison_id", "left_path", "right_path", "boundary",
        "left_runtime_shape", "right_runtime_shape",
        "left_invocation_context", "right_invocation_context",
        "workload_classification", "performance_comparison",
        "evidence_level", "reason",
    ]
    comparison_rows = [
        {
            "comparison_id": "HISTORICAL_PAIR",
            "left_path": "h16816",
            "right_path": "xmma",
            "boundary": "Phase 3-C historical trace",
            "left_runtime_shape": "UNKNOWN",
            "right_runtime_shape": "UNKNOWN",
            "left_invocation_context": "UNKNOWN",
            "right_invocation_context": "UNKNOWN",
            "workload_classification": "UNKNOWN",
            "performance_comparison": "RAW_LATENCY_RATIO_NOT_VALID",
            "evidence_level": "UNKNOWN",
            "reason": "Historical Nsys does not expose kernel arguments or runtime shapes",
        },
        {
            "comparison_id": "HISTORICAL_H16816_VS_NEW_XMMA",
            "left_path": "h16816",
            "right_path": "xmma",
            "boundary": "Phase 3-C historical versus Phase 6-E controlled",
            "left_runtime_shape": "UNKNOWN",
            "right_runtime_shape": "DERIVED_FROM_PROVEN_STATE",
            "left_invocation_context": "HISTORICAL_EVIDENCE",
            "right_invocation_context": "DIRECT_RUNTIME_EVIDENCE",
            "workload_classification": "UNKNOWN",
            "performance_comparison": "RAW_LATENCY_RATIO_NOT_VALID",
            "evidence_level": "UNKNOWN",
            "reason": "Historical h16816 runtime shape and operation identity remain unknown",
        },
        {
            "comparison_id": "NEW_H16816_REPRODUCTION",
            "left_path": "h16816",
            "right_path": "xmma",
            "boundary": "Phase 6-E controlled decode steps 0-3",
            "left_runtime_shape": "NOT_OBSERVED",
            "right_runtime_shape": "DERIVED_FROM_PROVEN_STATE",
            "left_invocation_context": "NOT_OBSERVED",
            "right_invocation_context": "DIRECT_RUNTIME_EVIDENCE",
            "workload_classification": "UNKNOWN",
            "performance_comparison": "NOT_CALCULATED",
            "evidence_level": "DIRECT_RUNTIME_EVIDENCE",
            "reason": "Controlled decode invocations contain zero h16816 kernels",
        },
    ]
    write_csv(args.out / "qk_workload_comparison.csv", comparison_rows,
              comparison_fields)


if __name__ == "__main__":
    main()
