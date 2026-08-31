#!/usr/bin/env python3
import csv
import glob
import os
import re
import sys

source_dir, output = sys.argv[1:3]

def number(value):
    if value in (None, "", "N/A"):
        return None
    return float(value.replace(",", ""))

def value(metrics, name):
    result = number(metrics.get(name))
    return "NA" if result is None else result

fields = [
    "version", "kernel", "block_size", "grid_size", "profile_sm_frequency_mhz",
    "profile_duration_ms", "sm_throughput_pct", "memory_throughput_pct",
    "l1tex_throughput_pct", "l2_throughput_pct", "achieved_occupancy_pct",
    "active_warps_per_scheduler", "eligible_warps_per_scheduler", "issue_active_pct",
    "long_scoreboard_cycles_per_issue", "barrier_cycles_per_issue",
    "not_selected_cycles_per_issue", "global_load_requests", "global_load_sectors",
    "load_sectors_per_request", "global_store_requests", "global_store_sectors",
    "store_sectors_per_request", "global_store_excessive_sectors",
    "shared_load_bank_conflicts", "shared_store_bank_conflicts",
    "shared_total_bank_conflicts", "shared_excessive_wavefronts",
    "shared_load_instructions", "shared_store_instructions", "registers_per_thread",
    "static_shared_memory_kb",
]
rows = []
for path in sorted(glob.glob(os.path.join(source_dir, "ncu_v*_raw.csv"))):
    raw = list(csv.reader(open(path, newline="")))
    metrics = dict(zip(raw[0], raw[2]))
    match = re.search(r"ncu_v([1-4])_", os.path.basename(path))
    load_requests = number(metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"))
    load_sectors = number(metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"))
    store_requests = number(metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"))
    store_sectors = number(metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"))
    rows.append({
        "version": f"V{match.group(1)}", "kernel": metrics["Kernel Name"],
        "block_size": metrics["Block Size"], "grid_size": metrics["Grid Size"],
        "profile_sm_frequency_mhz": value(metrics, "gpc__cycles_elapsed.avg.per_second"),
        "profile_duration_ms": value(metrics, "gpu__time_duration.avg"),
        "sm_throughput_pct": value(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "memory_throughput_pct": value(metrics, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
        "l1tex_throughput_pct": value(metrics, "l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
        "l2_throughput_pct": value(metrics, "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        "achieved_occupancy_pct": value(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "active_warps_per_scheduler": value(metrics, "smsp__warps_active.avg.per_cycle_active"),
        "eligible_warps_per_scheduler": value(metrics, "smsp__warps_eligible.avg.per_cycle_active"),
        "issue_active_pct": value(metrics, "smsp__issue_active.avg.pct_of_peak_sustained_active"),
        "long_scoreboard_cycles_per_issue": value(metrics, "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
        "barrier_cycles_per_issue": value(metrics, "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
        "not_selected_cycles_per_issue": value(metrics, "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"),
        "global_load_requests": load_requests, "global_load_sectors": load_sectors,
        "load_sectors_per_request": load_sectors / load_requests,
        "global_store_requests": store_requests, "global_store_sectors": store_sectors,
        "store_sectors_per_request": store_sectors / store_requests,
        "global_store_excessive_sectors": store_sectors - 4 * store_requests,
        "shared_load_bank_conflicts": value(metrics, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
        "shared_store_bank_conflicts": value(metrics, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
        "shared_total_bank_conflicts": value(metrics, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"),
        "shared_excessive_wavefronts": value(metrics, "derived__memory_l1_wavefronts_shared_excessive"),
        "shared_load_instructions": value(metrics, "sass__inst_executed_shared_loads"),
        "shared_store_instructions": value(metrics, "sass__inst_executed_shared_stores"),
        "registers_per_thread": value(metrics, "launch__registers_per_thread"),
        "static_shared_memory_kb": value(metrics, "launch__shared_mem_per_block_static"),
    })
with open(output, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
