from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_phase3b_module():
    source = Path(__file__).resolve().parents[1] / "phase3b" / "phase3b_runtime_context.py"
    spec = importlib.util.spec_from_file_location("phase3b_runtime_context", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P3B = load_phase3b_module()


def benchmark_mode(args) -> dict:
    results = []
    summary = {}
    memory = [P3B.memory_snapshot("phase3c_benchmark_start")]
    for mixed in (False, True):
        result, _info, snapshots = P3B.bench_one(args, mixed=mixed, persistent=True)
        results.append(result)
        summary[result["label"]] = result["rows"]
        memory.extend(snapshots)
    return {
        "phase": "Phase 3-C1",
        "comparison": "FP16 persistent vs Mixed persistent",
        "lifetime": "persistent_context_lifetime",
        "results": results,
        "summary": summary,
        "memory": memory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["bench"], default="bench")
    parser.add_argument("--lifetime", choices=["persistent_context_lifetime"],
                        default="persistent_context_lifetime")
    parser.add_argument("--fp16-dir", type=Path,
                        default=Path("/tmp/phase2_2b4_2_20260902T082326Z"))
    parser.add_argument("--mixed-dir", type=Path,
                        default=Path("/tmp/phase2_3e_20260904T020000Z/work"))
    parser.add_argument("--embedding-engine", type=Path,
                        default=Path("/tmp/phase2_2c1_20260902T090000Z/embedding_fp16.engine"))
    parser.add_argument("--norm-engine", type=Path,
                        default=Path("/tmp/phase2_2c2_20260903T/final_rmsnorm_fp32_reduce.engine"))
    parser.add_argument("--lm-engine", type=Path,
                        default=Path("/tmp/phase2_2c3_20260903T024500Z/lm_head_fp16.engine"))
    parser.add_argument("--eval-manifest", type=Path,
                        default=Path("experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T205610Z/evaluation_manifest.json"))
    parser.add_argument("--force-cont", type=Path,
                        default=Path("experiments/Phase2-qwen3-quantization/src/phase2_3f/force_cont.json"))
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--window-repeats", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads(args.eval_manifest.read_text())
    evaluation = [row for row in manifest["rows"] if row.get("split") == "evaluation"]
    if len(evaluation) != 12:
        raise RuntimeError(f"EVAL_PROMPT_COUNT:{len(evaluation)}")
    args.bench_ids = evaluation[0]["token_ids"]
    args.force_cont = json.loads(args.force_cont.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    P3B.F.dump(args.out / "environment.json", P3B.environment_snapshot())
    P3B.F.dump(args.out / "run_config.json", {
        "mode": args.mode,
        "lifetime": args.lifetime,
        "runtimes": ["fp16_persistent", "mixed_persistent"],
        "prefill_lengths": args.prefill_lengths,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "window_repeats": args.window_repeats,
        "decode_steps": args.decode_steps,
        "workload": "deterministic first evaluation prompt ids, batch 1",
        "sampling": "forced deterministic continuation tokens",
        "timing": "time.perf_counter with torch.cuda.synchronize before/after",
    })
    if args.mode != "bench":
        raise RuntimeError("UNSUPPORTED_MODE")
    result = benchmark_mode(args)
    P3B.F.dump(args.out / "benchmark_result.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
