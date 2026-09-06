from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys


SQLITE_PATH = "/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite"
AV_PATTERN = re.compile(r"^/MatMul_(\d+)$")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boundary_for(start: int, end: int, boundaries: list[dict]) -> str:
    matches = [
        row["name"]
        for row in boundaries
        if start >= row["start"] and end <= row["end"]
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return f"MULTIPLE_BOUNDARIES:{'|'.join(matches)}"
    return "OUTSIDE_OBSERVED_BOUNDARIES"


def main() -> None:
    connection = sqlite3.connect(
        f"file:{SQLITE_PATH}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row

    boundaries = [
        dict(row)
        for row in connection.execute(
            """
            SELECT ne.text AS name, ne.start, ne.end
            FROM NVTX_EVENTS ne
            WHERE ne.text LIKE 'PHASE3B_%'
            ORDER BY ne.start
            """
        )
    ]

    av_ranges = []
    for row in connection.execute(
        """
        SELECT ne.start, ne.end, ne.globalTid,
               COALESCE(s.value, ne.text) AS range_text
        FROM NVTX_EVENTS ne
        LEFT JOIN StringIds s ON ne.textId = s.id
        ORDER BY ne.start
        """
    ):
        match = AV_PATTERN.match(row["range_text"] or "")
        if match and int(match.group(1)) % 2 == 1:
            av_ranges.append(dict(row))

    record_count = 0
    for av_range in av_ranges:
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
            (av_range["globalTid"], av_range["start"], av_range["end"]),
        ).fetchall()
        correlated_kernels = []
        for runtime in runtime_rows:
            if runtime["correlationId"] is None:
                continue
            kernels = connection.execute(
                """
                SELECT k.start, k.end, k.correlationId, k.streamId,
                       k.gridX, k.gridY, k.gridZ,
                       k.blockX, k.blockY, k.blockZ,
                       demangled.value AS demangled_name,
                       short.value AS short_name
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                LEFT JOIN StringIds demangled
                  ON k.demangledName = demangled.id
                LEFT JOIN StringIds short ON k.shortName = short.id
                WHERE k.correlationId = ?
                ORDER BY k.start
                """,
                (runtime["correlationId"],),
            ).fetchall()
            if len(kernels) > 1:
                raise RuntimeError(
                    "MULTIPLE_KERNELS_FOR_CORRELATION:"
                    f"{runtime['correlationId']}"
                )
            for kernel in kernels:
                correlated_kernels.append(dict(kernel))

        record = {
            "source_sqlite_path": SQLITE_PATH,
            "source_sqlite_bytes": 3477504,
            "source_sqlite_sha256": sha256(SQLITE_PATH),
            "nvtx_start": av_range["start"],
            "nvtx_end": av_range["end"],
            "nvtx_global_tid": av_range["globalTid"],
            "nvtx_range": av_range["range_text"],
            "nvtx_boundary": boundary_for(
                av_range["start"], av_range["end"], boundaries
            ),
            "runtime_rows": [dict(row) for row in runtime_rows],
            "correlated_kernels": correlated_kernels,
            "kernel_boundaries": [
                boundary_for(kernel["start"], kernel["end"], boundaries)
                for kernel in correlated_kernels
            ],
            "observed_boundaries": [
                {key: row[key] for key in ("name", "start", "end")}
                for row in boundaries
            ],
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        record_count += 1

    connection.close()
    print(
        json.dumps({"av_nvtx_range_count": record_count}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
