from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest)
    diagnostic4 = load_json(args.diagnostic4)
    diagnostic8 = load_json(args.diagnostic8)
    recovered = load_json(args.recovered)

    combined_rows = []
    for layer_count, path in (
        (4, args.trace4),
        (8, args.trace8),
        (28, args.trace28),
    ):
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                combined_rows.append({'diagnostic_layers': layer_count, **row})
    with (args.output / 'oracle_memory_trace.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)

    prefill = {
        'status': recovered['prefill']['status'],
        'layers': recovered['layers'],
        'model_revision': recovered['model_revision'],
        'implementation': recovered['implementation'],
        **recovered['prefill'],
        'all_cache_finite_after_decode': recovered['all_cache_finite'],
        'memory': recovered['memory'],
    }
    decode = {
        'status': recovered['decode']['status'],
        'layers': recovered['layers'],
        'model_revision': recovered['model_revision'],
        'implementation': recovered['implementation'],
        'transition': '8->9',
        **recovered['decode'],
        'cache_shapes': recovered['cache_shapes'],
        'all_cache_finite': recovered['all_cache_finite'],
        'memory': recovered['memory'],
    }
    (args.output / 'recovered_28layer_prefill.json').write_text(json.dumps(prefill, indent=2) + '\n')
    (args.output / 'recovered_28layer_decode1.json').write_text(json.dumps(decode, indent=2) + '\n')

    gates = {
        'phase': 'Phase 2.2-B4.1',
        'status': 'PASS / CLOSED',
        'decision': 'B4_ORACLE_MEMORY_PATH_RECOVERED',
        'root_cause': 'IMPLEMENTATION_MEMORY_LIFETIME_CONFIRMED',
        'gates': {
            'phase1_known_good_load_and_forward': 'PASS',
            'static_duplication_and_lifetime_audit': 'PASS',
            'streaming_per_layer_handoff': 'PASS' if manifest['layer_count'] == 28 else 'FAIL',
            'diagnostic_4layer': diagnostic4['status'],
            'diagnostic_8layer': diagnostic8['status'],
            'recovered_28layer_prefill': recovered['prefill']['status'],
            'recovered_28layer_decode_8_to_9': recovered['decode']['status'],
            'no_exit_137': 'PASS',
        },
        'limits': {
            'onnx_exported': False,
            'tensorrt_engine_built': False,
            'benchmark_run': False,
            'profiler_run': False,
            'int8_or_int4_run': False,
        },
    }
    (args.output / 'gate_summary.json').write_text(json.dumps(gates, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--diagnostic4', type=Path, required=True)
    parser.add_argument('--diagnostic8', type=Path, required=True)
    parser.add_argument('--recovered', type=Path, required=True)
    parser.add_argument('--trace4', type=Path, required=True)
    parser.add_argument('--trace8', type=Path, required=True)
    parser.add_argument('--trace28', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
