from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def load_phase2_3f_module():
    f_path = Path(__file__).resolve().parents[1] / "phase2_3f" / "phase2_3f_compare.py"
    spec = importlib.util.spec_from_file_location("phase2_3f_compare", f_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = load_phase2_3f_module()


def read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def command_text(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=20,
                              check=False).stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def environment_snapshot() -> dict:
    import tensorrt as trt

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "tegra_release": read_text("/etc/nv_tegra_release"),
        "nvpmodel": command_text(["nvpmodel", "-q"]),
        "jetson_clocks_show": command_text(["jetson_clocks", "--show"]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "tensorrt": trt.__version__,
        "nsys": command_text(["nsys", "--version"]),
        "git_head": command_text(["git", "rev-parse", "HEAD"]),
        "git_branch": command_text(["git", "branch", "--show-current"]),
    }


def nvtx_range(name: str, enabled: bool):
    class _Ctx:
        def __enter__(self):
            if enabled:
                torch.cuda.nvtx.range_push(name)
            return self

        def __exit__(self, *exc):
            if enabled:
                torch.cuda.nvtx.range_pop()
            return False

    return _Ctx()


def bench_runtime(label: str, pre_path: Path, dec_path: Path, embed, logits,
                  args) -> dict:
    pre = F.TRT(pre_path)
    dec = F.TRT(dec_path)
    rows = {}
    for seq_len in args.prefill_lengths:
        ids = args.bench_ids[:seq_len]

        for _ in range(args.warmup):
            hidden, _, _ = F.run_prefill(pre, embed, ids)
            _ = logits(hidden[:, -1:, :])

        prefill_raw = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx_range(f"prefill_S{seq_len}", args.nvtx):
                hidden, _, _ = F.run_prefill(pre, embed, ids)
                _ = logits(hidden[:, -1:, :])
            torch.cuda.synchronize()
            prefill_raw.append(time.perf_counter() - t0)

        cache_hidden, cache_k, cache_v = F.run_prefill(pre, embed, ids)
        for _ in range(args.warmup):
            dec_hidden, _, _ = F.run_decode(dec, embed, args.force_cont[0],
                                            seq_len, cache_k, cache_v)
            _ = logits(dec_hidden)

        decode_raw = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx_range(f"decode_step_S{seq_len}", args.nvtx):
                dec_hidden, _, _ = F.run_decode(dec, embed, args.force_cont[0],
                                                seq_len, cache_k, cache_v)
                _ = logits(dec_hidden)
            torch.cuda.synchronize()
            decode_raw.append(time.perf_counter() - t0)

        window_raw = []
        # Decode windows are anchored at an 8-token cache so eight decode
        # steps stay within the engine's 16-token optimization profile.
        window_ids = args.bench_ids[:8]
        for _ in range(args.window_repeats):
            running_hidden, running_k, running_v = F.run_prefill(pre, embed, window_ids)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with nvtx_range(f"decode_window_S{seq_len}", args.nvtx):
                for step in range(args.decode_steps):
                    token = int(args.force_cont[step % len(args.force_cont)])
                    running_hidden, running_k, running_v = F.run_decode(
                        dec, embed, token, 8 + step, running_k, running_v)
                    _ = logits(running_hidden)
            torch.cuda.synchronize()
            window_raw.append(time.perf_counter() - t0)

        rows[str(seq_len)] = {
            "prefill_ttft_ms": F.timing_summary(prefill_raw),
            "prefill_raw_s": prefill_raw,
            "decode_tpot_ms": F.timing_summary(decode_raw),
            "decode_raw_s": decode_raw,
            "decode_tokens_per_sec": float(1.0 / np.mean(decode_raw)),
            "decode_window_ms": F.timing_summary(window_raw),
            "decode_window_raw_s": window_raw,
            "decode_window_steps": args.decode_steps,
            "decode_window_cache_len": 8,
        }
    del pre, dec
    return rows


def compare(fp16_rows: dict, mixed_rows: dict, lengths: list[int]) -> dict:
    out = {}
    for seq_len in lengths:
        fp = fp16_rows[str(seq_len)]
        mx = mixed_rows[str(seq_len)]
        out[str(seq_len)] = {
            "prefill_median_ms": {
                "fp16": fp["prefill_ttft_ms"]["median_ms"],
                "mixed": mx["prefill_ttft_ms"]["median_ms"],
                "mixed_slower_pct": 100.0 * (mx["prefill_ttft_ms"]["median_ms"]
                                             / fp["prefill_ttft_ms"]["median_ms"] - 1.0),
            },
            "decode_median_ms": {
                "fp16": fp["decode_tpot_ms"]["median_ms"],
                "mixed": mx["decode_tpot_ms"]["median_ms"],
                "mixed_slower_pct": 100.0 * (mx["decode_tpot_ms"]["median_ms"]
                                             / fp["decode_tpot_ms"]["median_ms"] - 1.0),
            },
            "decode_tokens_per_sec": {
                "fp16": fp["decode_tokens_per_sec"],
                "mixed": mx["decode_tokens_per_sec"],
            },
            "reproduced_mixed_slower": (
                mx["prefill_ttft_ms"]["median_ms"] > fp["prefill_ttft_ms"]["median_ms"]
                and mx["decode_tpot_ms"]["median_ms"] > fp["decode_tpot_ms"]["median_ms"]
            ),
        }
    return out


def profile_workload(args) -> dict:
    engine_dir = args.profile_engine_dir
    prefix = "" if args.profile_runtime == "fp16" else "mixed_"
    pre = F.TRT(engine_dir / f"{prefix}prefill_28layer.engine")
    dec = F.TRT(engine_dir / f"{prefix}decode_28layer.engine")
    embedding = F.TRT(args.embedding_engine)
    norm = F.TRT(args.norm_engine)
    lm = F.TRT(args.lm_engine)
    embed, logits = F.make_pipeline(embedding, norm, lm)
    ids = args.bench_ids[:args.profile_seq_len]

    for _ in range(2):
        hidden, _, _ = F.run_prefill(pre, embed, ids)
        _ = logits(hidden[:, -1:, :])
    torch.cuda.synchronize()

    with nvtx_range("A1_prefill_S8", True):
        hidden, cache_k, cache_v = F.run_prefill(pre, embed, ids)
        _ = logits(hidden[:, -1:, :])
    torch.cuda.synchronize()

    for step in range(args.profile_decode_steps):
        with nvtx_range(f"A1_decode_step_{step}", True):
            hidden, cache_k, cache_v = F.run_decode(
                dec, embed, int(args.force_cont[step % len(args.force_cont)]),
                args.profile_seq_len + step, cache_k, cache_v)
            _ = logits(hidden)
        torch.cuda.synchronize()

    del pre, dec
    return {"profiled": True, "engine_dir": str(engine_dir),
            "seq_len": args.profile_seq_len,
            "decode_steps": args.profile_decode_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["bench", "profile"], default="bench")
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
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--window-repeats", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--profile-runtime", choices=["fp16", "mixed"], default="fp16")
    parser.add_argument("--profile-seq-len", type=int, default=8)
    parser.add_argument("--profile-decode-steps", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.eval_manifest.read_text())
    evaluation = [row for row in manifest["rows"] if row.get("split") == "evaluation"]
    if len(evaluation) != 12:
        raise RuntimeError(f"EVAL_PROMPT_COUNT:{len(evaluation)}")
    args.bench_ids = evaluation[0]["token_ids"]
    args.force_cont = json.loads(args.force_cont.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    F.dump(args.out / "environment.json", environment_snapshot())

    if args.mode == "profile":
        engine_dir = args.fp16_dir if args.profile_runtime == "fp16" else args.mixed_dir
        args.profile_engine_dir = engine_dir
        result = profile_workload(args)
        F.dump(args.out / "profile_manifest.json", result)
        print(json.dumps(result))
        return

    embedding = F.TRT(args.embedding_engine)
    norm = F.TRT(args.norm_engine)
    lm = F.TRT(args.lm_engine)
    embed, logits = F.make_pipeline(embedding, norm, lm)

    fp16_rows = bench_runtime("fp16", args.fp16_dir / "prefill_28layer.engine",
                              args.fp16_dir / "decode_28layer.engine",
                              embed, logits, args)
    mixed_rows = bench_runtime("mixed", args.mixed_dir / "mixed_prefill_28layer.engine",
                               args.mixed_dir / "mixed_decode_28layer.engine",
                               embed, logits, args)

    comparison = compare(fp16_rows, mixed_rows, args.prefill_lengths)
    config = {
        "mode": "bench",
        "workload": "deterministic first evaluation prompt ids, batch 1",
        "prefill_lengths": args.prefill_lengths,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "window_repeats": args.window_repeats,
        "decode_steps": args.decode_steps,
        "sampling": "forced deterministic continuation tokens",
        "timing": "time.perf_counter with torch.cuda.synchronize before/after",
        "engines": {
            "fp16_dir": str(args.fp16_dir),
            "mixed_dir": str(args.mixed_dir),
            "embedding": str(args.embedding_engine),
            "norm": str(args.norm_engine),
            "lm_head": str(args.lm_engine),
        },
    }
    F.dump(args.out / "benchmark_config.json", config)
    F.dump(args.out / "benchmark_fp16.json", fp16_rows)
    F.dump(args.out / "benchmark_mixed.json", mixed_rows)
    F.dump(args.out / "benchmark_comparison.json", comparison)
    summary = {
        "phase": "Phase 3-A1",
        "reproduced_mixed_slower": all(
            comparison[str(s)]["reproduced_mixed_slower"] for s in args.prefill_lengths),
        "comparison": comparison,
    }
    F.dump(args.out / "a1_gate.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
