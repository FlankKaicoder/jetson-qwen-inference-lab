#!/usr/bin/env python3
import csv
import datetime as dt
import re
import sys

line = sys.argv[1] if len(sys.argv) > 1 else ""
output, phase, dimension, version, trial = sys.argv[2:7]

def one(pattern):
    match = re.search(pattern, line)
    return match.group(1) if match else "NA"

gpu = one(r"GR3D_FREQ\s+\d+%@(?:\[)?(\d+)")
temperature = one(r"(?:GPU|gpu)@([0-9.]+)C")
emc = one(r"EMC_FREQ\s+\d+%@(?:\[)?(\d+)")
power = one(r"VDD_IN\s+(\d+)mW")
gpu_util = one(r"GR3D_FREQ\s+(\d+)%@")
ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
ram_used, ram_total = (ram.group(1), ram.group(2)) if ram else ("NA", "NA")
cpu = one(r"CPU\s+\[([^\]]+)\]")
timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
try:
    with open(output, newline="") as stream:
        exists = bool(stream.read(1))
except FileNotFoundError:
    exists = False
with open(output, "a", newline="") as stream:
    writer = csv.writer(stream)
    if not exists:
        writer.writerow(["timestamp", "phase", "dimension", "version", "trial", "gpu_freq_mhz", "temperature_c", "emc_freq_mhz", "power_mw", "gpu_util_pct", "cpu_util_raw", "ram_used_mb", "ram_total_mb", "raw_tegrastats"])
    writer.writerow([timestamp, phase, dimension, version, trial, gpu, temperature, emc, power, gpu_util, cpu, ram_used, ram_total, line])
