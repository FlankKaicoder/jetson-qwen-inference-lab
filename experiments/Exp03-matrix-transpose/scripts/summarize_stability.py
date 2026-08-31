#!/usr/bin/env python3
import csv
import math
import statistics
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
groups = defaultdict(list)
for row in rows:
    groups[(row["version"], int(row["width"]), int(row["height"]))].append(row)
print("version,width,height,trials,mean_ms,median_ms,min_ms,max_ms,std_ms,cv_pct,p95_ms,effective_gb_s,mean_calibrated_kernel_ms,mean_warmup_iterations,mean_estimated_warmup_ms,mean_measured_iterations,mean_timed_window_ms")
for (version, width, height), items in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
    values = sorted(float(row["latency_ms"]) for row in items)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    p95 = values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
    average = lambda key: statistics.mean(float(row[key]) for row in items)
    gb = 2 * width * height * 4 / (mean * 1e6)
    print(f"{version},{width},{height},{len(values)},{mean:.9f},{statistics.median(values):.9f},{min(values):.9f},{max(values):.9f},{std:.9f},{std/mean*100:.6f},{p95:.9f},{gb:.6f},{average('calibrated_kernel_ms'):.9f},{average('warmup_iterations'):.3f},{average('estimated_warmup_ms'):.3f},{average('measured_iterations'):.3f},{average('actual_timed_window_ms'):.3f}")
