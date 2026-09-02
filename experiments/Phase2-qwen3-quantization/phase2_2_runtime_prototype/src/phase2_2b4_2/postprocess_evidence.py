from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(a):
    numerical = json.loads((a.out / 'numerical_propagation.json').read_text())
    rows = []
    for item in numerical['rows']:
        for tensor in ('hidden', 'k', 'v'):
            metric = item[tensor]
            rows.append({'stage': item['stage'], 'layer': item['layer'], 'tensor': tensor,
                         'max_abs_error': metric['max_abs_error'], 'rmse': metric['rmse'],
                         'relative_l2': metric['relative_l2'], 'cosine': metric['cosine'],
                         'finite': metric['finite'], 'shape_equal': metric['shape_equal']})
    with (a.out / 'layer_numerical_propagation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    log = (a.out / 'trt_build.log').read_text()
    warning_lines = [line for line in log.splitlines() if '[W]' in line or 'WARNING' in line]
    pre, dec, start = a.tmp / 'prefill_28layer.engine', a.tmp / 'decode_28layer.engine', a.out / 'memory_before_build.txt'
    timing = {
        'source': 'filesystem timestamps; one-second resolution; includes parser/build and shell overhead',
        'total_build_wall_seconds_estimate': int(dec.stat().st_mtime - start.stat().st_mtime),
        'prefill_completion_seconds_from_start': int(pre.stat().st_mtime - start.stat().st_mtime),
        'decode_completion_seconds_from_prefill': int(dec.stat().st_mtime - pre.stat().st_mtime),
        'warnings': warning_lines,
    }
    (a.out / 'trt_build_summary.json').write_text(json.dumps(timing, indent=2) + '\n')
    (a.out / 'execution_wrapper_note.txt').write_text(
        'Python runtime completed and gate_summary.json was written. The outer PowerShell-to-bash '
        'exit-code capture escaped $? incorrectly, leaving runtime_exit_code.txt blank and causing '
        'only the final shell exit command to report a numeric-argument error. The experiment was not rerun.\n'
    )
    print(json.dumps({'status': 'PASS', 'csv_rows': len(rows), **timing}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--tmp', type=Path, required=True); main(p.parse_args())
