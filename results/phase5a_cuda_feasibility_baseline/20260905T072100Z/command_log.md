# Command Log

The analysis script was executed once on Jetson against the existing Phase 4-F
trace. The remote output was then copied unchanged into this result directory.

```bash
python3 phase5a_boundary_reconciliation.py \
  --sqlite /tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite \
  --json-out /tmp/phase5a_boundary_20260905T072100Z/trt_boundary_reconciliation_raw.json \
  --csv-out /tmp/phase5a_boundary_20260905T072100Z/kernel_breakdown.csv
```

The script opened the source SQLite file with read-only URI mode. No new NSYS
capture, TensorRT execution, benchmark or profiler run was performed.

Remote hash verification used:

```bash
sha256sum \
  /tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite \
  /tmp/phase5a_boundary_20260905T072100Z/trt_boundary_reconciliation_raw.json \
  /tmp/phase5a_boundary_20260905T072100Z/kernel_breakdown.csv
```

The observed hashes matched the values recorded in `preflight.json`.
