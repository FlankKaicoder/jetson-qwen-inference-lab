#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict

measurements, telemetry, output = sys.argv[1:4]
with open(telemetry, newline="") as stream:
    telemetry_rows = list(csv.DictReader(stream))
telemetry_map = defaultdict(dict)
for row in telemetry_rows:
    telemetry_map[(row["dimension"], row["version"], row["trial"])][row["phase"]] = row
with open(measurements, newline="") as stream:
    rows = list(csv.DictReader(stream))
extra = [
    "gpu_freq_before_mhz", "gpu_freq_after_mhz",
    "temperature_before_c", "temperature_after_c",
    "emc_freq_before_mhz", "emc_freq_after_mhz",
    "power_before_mw", "power_after_mw",
    "gpu_util_before_pct", "gpu_util_after_pct",
    "cpu_util_before_raw", "cpu_util_after_raw",
    "telemetry_before_timestamp", "telemetry_after_timestamp",
]
with open(output, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) + extra)
    writer.writeheader()
    for row in rows:
        phases = telemetry_map[(f"{row['width']}x{row['height']}", row["version"], row["trial"])]
        fields = [
            ("gpu_freq_before_mhz", "pre", "gpu_freq_mhz"),
            ("gpu_freq_after_mhz", "post", "gpu_freq_mhz"),
            ("temperature_before_c", "pre", "temperature_c"),
            ("temperature_after_c", "post", "temperature_c"),
            ("emc_freq_before_mhz", "pre", "emc_freq_mhz"),
            ("emc_freq_after_mhz", "post", "emc_freq_mhz"),
            ("power_before_mw", "pre", "power_mw"),
            ("power_after_mw", "post", "power_mw"),
            ("gpu_util_before_pct", "pre", "gpu_util_pct"),
            ("gpu_util_after_pct", "post", "gpu_util_pct"),
            ("cpu_util_before_raw", "pre", "cpu_util_raw"),
            ("cpu_util_after_raw", "post", "cpu_util_raw"),
            ("telemetry_before_timestamp", "pre", "timestamp"),
            ("telemetry_after_timestamp", "post", "timestamp"),
        ]
        for name, phase, field in fields:
            row[name] = phases.get(phase, {}).get(field, "NA")
        writer.writerow(row)
