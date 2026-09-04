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


class LegacyContext(F.TRT):
    def __init__(self, path: Path):
        super().__init__(path)
        self.lifetime = "legacy_context_lifetime"
        self.context_creations = 0
        self.context_reuses = 0
        self.engine_path = str(path)

    def run(self, inputs: dict) -> dict:
        self.context_creations += 1
        return super().run(inputs)


class PersistentContext(F.TRT):
    def __init__(self, path: Path):
        super().__init__(path)
        self.lifetime = "persistent_context_lifetime"
        self.engine_path = str(path)
        self.context_creations = 1
        self.context_reuses = 0
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"CONTEXT_CREATION_FAILED:{path}")

    def run(self, inputs: dict) -> dict:
        self.context_reuses += 1
        ctx = self.context
        for name, value in inputs.items():
            if not ctx.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"INPUT_SHAPE_REJECTED:{name}:{tuple(value.shape)}")
            ctx.set_tensor_address(name, value.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                tdtype = torch.float16 if dtype == self.trt.DataType.HALF else torch.float32
                outputs[name] = torch.empty(shape, device="cuda", dtype=tdtype)
                ctx.set_tensor_address(name, outputs[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"EXECUTE_FAILED:{self.engine_path}")
        self.stream.synchronize()
        return outputs


def make_trt(path: Path, persistent: bool):
    return PersistentContext(path) if persistent else LegacyContext(path)


def lifetime_counters(runtime: dict) -> dict:
    return {
        "context_creations": sum(obj.context_creations for obj in runtime["objects"]),
        "context_reuses": sum(obj.context_reuses for obj in runtime["objects"]),
    }


def read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def command_text(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=20, check=False).stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def current_rss_kb() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        pass
    return None


def memory_snapshot(stage: str) -> dict:
    row = F.snap(stage)
    row["current_rss_kb"] = current_rss_kb()
    return row


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


def load_pipeline(engine_dir: Path, persistent: bool, mixed: bool,
                  args, label: str) -> tuple[dict, list]:
    prefix = "mixed_" if mixed else ""
    paths = [
        args.embedding_engine,
        args.norm_engine,
        args.lm_engine,
        engine_dir / f"{prefix}prefill_28layer.engine",
        engine_dir / f"{prefix}decode_28layer.engine",
    ]
    lifetime = "persistent_context_lifetime" if persistent else "legacy_context_lifetime"
    t0 = time.perf_counter()
    objects = [make_trt(path, persistent) for path in paths]
    init_s = time.perf_counter() - t0
    embed = objects[0]
    norm = objects[1]
    lm = objects[2]
    pre = objects[3]
    dec = objects[4]
    pipeline = F.make_pipeline(embed, norm, lm)
    info = {
        "label": label,
        "mixed": mixed,
        "lifetime": lifetime,
        "engine_paths": [str(path) for path in paths],
        "initialization_s": init_s,
        "engine_context_creations": sum(obj.context_creations for obj in objects),
    }
    return {"embed": pipeline, "embed_fn": pipeline[0], "logits_fn": pipeline[1],
            "objects": objects,
            "embedding": embed, "norm": norm, "lm": lm, "pre": pre, "dec": dec}, info


def cache_invariant_step(prev_ks: list, prev_vs: list, new_ks: list,
                         new_vs: list, old_len: int) -> dict:
    prefix_ok = True
    shape_ok = True
    # Phase 2.2-B4.2 evidence records present_k/present_v as [B, 8, L, 128].
    for new_k in new_ks:
        shape_ok &= bool(tuple(new_k.shape) == (1, 8, old_len, 128))
    for new_v in new_vs:
        shape_ok &= bool(tuple(new_v.shape) == (1, 8, old_len, 128))
    for prev_k, new_k in zip(prev_ks, new_ks):
        prefix_ok &= bool(torch.equal(prev_k, new_k[:, :, :old_len, :]))
    for prev_v, new_v in zip(prev_vs, new_vs):
        prefix_ok &= bool(torch.equal(prev_v, new_v[:, :, :old_len, :]))
    lengths = [int(k.shape[2]) for k in new_ks]
    k_ptrs = [int(k.data_ptr()) for k in new_ks]
    v_ptrs = [int(v.data_ptr()) for v in new_vs]
    return {
        "prefix_preserved": bool(prefix_ok),
        "shape_pass": bool(shape_ok),
        "cache_lengths": lengths,
        "expected_length": old_len,
        "all_lengths_equal_expected": all(length == old_len for length in lengths),
        "k_pointer_isolation": len(set(k_ptrs)) == 28,
        "v_pointer_isolation": len(set(v_ptrs)) == 28,
        "k_v_pointer_isolation": len(set(k_ptrs) | set(v_ptrs)) == 56,
    }


def functional_mode(args) -> dict:
    engine_dir = args.mixed_dir
    rows = []
    memory = [memory_snapshot("functional_start")]

    for seq_len in args.prefill_lengths:
        ids = args.bench_ids[:seq_len]
        decode_steps = 0 if seq_len == 16 else args.functional_decode_steps
        legacy, legacy_info = load_pipeline(engine_dir, False, True, args,
                                            f"mixed_legacy_S{seq_len}")
        legacy_memory = memory_snapshot(f"mixed_legacy_loaded_S{seq_len}")

        hidden, ks, vs = F.run_prefill(legacy["pre"], legacy["embed_fn"], ids)
        legacy_logits = legacy["lm"].run({"hidden_states": legacy["norm"].run(
            {"hidden_states": hidden[:, -1:, :]})["normalized_hidden_states"]})["logits"]
        legacy_rows = [{"hidden": hidden.detach().cpu(), "logits": legacy_logits.detach().cpu(),
                        "ks": [k.detach().cpu() for k in ks], "vs": [v.detach().cpu() for v in vs],
                        "invariants": {}}]
        prev_ks, prev_vs = ks, vs
        old_len = seq_len
        for step in range(decode_steps):
            token = int(args.force_cont[step % len(args.force_cont)])
            hidden, ks, vs = F.run_decode(legacy["dec"], legacy["embed_fn"], token,
                                          old_len, prev_ks, prev_vs)
            logits = legacy["lm"].run({"hidden_states": legacy["norm"].run(
                {"hidden_states": hidden})["normalized_hidden_states"]})["logits"]
            invariant = cache_invariant_step(prev_ks, prev_vs, ks, vs, old_len)
            legacy_rows.append({"hidden": hidden.detach().cpu(),
                                "logits": logits.detach().cpu(),
                                "ks": [k.detach().cpu() for k in ks],
                                "vs": [v.detach().cpu() for v in vs],
                                "invariants": invariant})
            prev_ks, prev_vs = ks, vs
            old_len += 1

        legacy_info["post_workload_context_counters"] = lifetime_counters(legacy)
        del legacy, hidden, ks, vs, legacy_logits, prev_ks, prev_vs
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        persistent, persistent_info = load_pipeline(engine_dir, True, True, args,
                                                     f"mixed_persistent_S{seq_len}")
        persistent_memory = memory_snapshot(f"mixed_persistent_loaded_S{seq_len}")
        hidden, ks, vs = F.run_prefill(persistent["pre"], persistent["embed_fn"], ids)
        persistent_logits = persistent["lm"].run({"hidden_states": persistent["norm"].run(
            {"hidden_states": hidden[:, -1:, :]})["normalized_hidden_states"]})["logits"]
        persistent_rows = [{"hidden": hidden.detach().cpu(), "logits": persistent_logits.detach().cpu(),
                            "ks": [k.detach().cpu() for k in ks], "vs": [v.detach().cpu() for v in vs],
                        "invariants": {}}]
        prev_ks, prev_vs = ks, vs
        old_len = seq_len
        for step in range(decode_steps):
            token = int(args.force_cont[step % len(args.force_cont)])
            hidden, ks, vs = F.run_decode(persistent["dec"], persistent["embed_fn"], token,
                                          old_len, prev_ks, prev_vs)
            logits = persistent["lm"].run({"hidden_states": persistent["norm"].run(
                {"hidden_states": hidden})["normalized_hidden_states"]})["logits"]
            invariant = cache_invariant_step(prev_ks, prev_vs, ks, vs, old_len)
            persistent_rows.append({"hidden": hidden.detach().cpu(),
                                    "logits": logits.detach().cpu(),
                                    "ks": [k.detach().cpu() for k in ks],
                                    "vs": [v.detach().cpu() for v in vs],
                                    "invariants": invariant})
            prev_ks, prev_vs = ks, vs
            old_len += 1

        comparisons = []
        kv_prefix_pass = True
        all_invariants_pass = True
        for step, (a, b) in enumerate(zip(legacy_rows, persistent_rows)):
            if step > 0:
                kv_prefix_pass &= a["invariants"]["prefix_preserved"]
                kv_prefix_pass &= b["invariants"]["prefix_preserved"]
                kv_prefix_pass &= a["invariants"]["all_lengths_equal_expected"]
                kv_prefix_pass &= b["invariants"]["all_lengths_equal_expected"]
                kv_prefix_pass &= a["invariants"]["k_pointer_isolation"]
                kv_prefix_pass &= a["invariants"]["v_pointer_isolation"]
                kv_prefix_pass &= b["invariants"]["k_pointer_isolation"]
                kv_prefix_pass &= b["invariants"]["v_pointer_isolation"]
            all_invariants_pass &= bool(a["invariants"].get("prefix_preserved", True))
            all_invariants_pass &= bool(b["invariants"].get("prefix_preserved", True))
            all_invariants_pass &= bool(a["invariants"].get("all_lengths_equal_expected", True))
            all_invariants_pass &= bool(b["invariants"].get("all_lengths_equal_expected", True))
            all_invariants_pass &= bool(a["invariants"].get("k_pointer_isolation", True))
            all_invariants_pass &= bool(a["invariants"].get("v_pointer_isolation", True))
            all_invariants_pass &= bool(b["invariants"].get("k_pointer_isolation", True))
            all_invariants_pass &= bool(b["invariants"].get("v_pointer_isolation", True))
            comparisons.append({
                "step": step,
                "hidden_exact": bool(torch.equal(a["hidden"], b["hidden"])),
                "hidden_metric": F.metric(a["hidden"], b["hidden"]),
                "logits_exact": bool(torch.equal(a["logits"], b["logits"])),
                "logits_metric": F.metric(a["logits"], b["logits"]),
                "kv_exact": all(
                    bool(torch.equal(x, y)) for x, y in zip(a["ks"], b["ks"])
                ) and all(bool(torch.equal(x, y)) for x, y in zip(a["vs"], b["vs"])),
            })

        exact = all(row["hidden_exact"] and row["logits_exact"] and row["kv_exact"]
                    for row in comparisons)
        finite = all(row["hidden_metric"]["finite"] and row["logits_metric"]["finite"]
                     for row in comparisons)
        rows.append({
            "seq_len": seq_len,
            "decode_steps": decode_steps,
            "legacy_info": legacy_info,
            "persistent_info": persistent_info,
            "legacy_memory": legacy_memory,
            "persistent_memory": persistent_memory,
            "comparisons": comparisons,
            "all_exact": exact,
            "all_finite": finite,
            "shape_pass": all(
                item["invariants"].get("shape_pass", True)
                for row in (legacy_rows, persistent_rows) for item in row
            ),
            "kv_gate_pass": kv_prefix_pass,
            "invariants_pass": all_invariants_pass,
        })
        persistent_info["post_workload_context_counters"] = lifetime_counters(persistent)
        del persistent, hidden, ks, vs, persistent_logits, prev_ks, prev_vs
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        memory.append(memory_snapshot(f"functional_complete_S{seq_len}"))

    gate_pass = all(row["all_exact"] and row["all_finite"] and row["shape_pass"]
                    and row["kv_gate_pass"]
                    and row["invariants_pass"] for row in rows)
    result = {
        "phase": "Phase 3-B2",
        "comparison": "Mixed legacy_context_lifetime vs Mixed persistent_context_lifetime",
        "gate": "PASS" if gate_pass else "BLOCKED",
        "pass": gate_pass,
        "rows": rows,
        "memory": memory,
    }
    return result


def bench_one(args, mixed: bool, persistent: bool) -> tuple[dict, dict, list]:
    engine_dir = args.mixed_dir if mixed else args.fp16_dir
    label = f"{'mixed' if mixed else 'fp16'}_{'persistent' if persistent else 'legacy'}"
    runtime, info = load_pipeline(engine_dir, persistent, mixed, args, label)
    memory = [memory_snapshot(f"{label}_loaded")]
    embed, logits = runtime["embed"]
    rows = {}
    for seq_len in args.prefill_lengths:
        ids = args.bench_ids[:seq_len]
        for _ in range(args.warmup):
            hidden, _, _ = F.run_prefill(runtime["pre"], embed, ids)
            _ = logits(hidden[:, -1:, :])
        memory.append(memory_snapshot(f"{label}_prefill_warmup_S{seq_len}"))
        prefill_raw = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hidden, _, _ = F.run_prefill(runtime["pre"], embed, ids)
            _ = logits(hidden[:, -1:, :])
            torch.cuda.synchronize()
            prefill_raw.append(time.perf_counter() - t0)

        cache_hidden, cache_k, cache_v = F.run_prefill(runtime["pre"], embed, ids)
        for _ in range(args.warmup):
            dec_hidden, _, _ = F.run_decode(runtime["dec"], embed,
                                             args.force_cont[0], seq_len,
                                             cache_k, cache_v)
            _ = logits(dec_hidden)
        memory.append(memory_snapshot(f"{label}_decode_warmup_S{seq_len}"))
        decode_raw = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dec_hidden, _, _ = F.run_decode(runtime["dec"], embed,
                                             args.force_cont[0], seq_len,
                                             cache_k, cache_v)
            _ = logits(dec_hidden)
            torch.cuda.synchronize()
            decode_raw.append(time.perf_counter() - t0)

        window_raw = []
        window_ids = args.bench_ids[:8]
        for _ in range(args.window_repeats):
            running_hidden, running_k, running_v = F.run_prefill(runtime["pre"], embed, window_ids)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for step in range(args.decode_steps):
                token = int(args.force_cont[step % len(args.force_cont)])
                running_hidden, running_k, running_v = F.run_decode(
                    runtime["dec"], embed, token, 8 + step, running_k, running_v)
                _ = logits(running_hidden)
            torch.cuda.synchronize()
            window_raw.append(time.perf_counter() - t0)
        memory.append(memory_snapshot(f"{label}_steady_state_S{seq_len}"))

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
    info["post_workload_context_counters"] = lifetime_counters(runtime)
    result = {"label": label, "mixed": mixed, "lifetime": info["lifetime"],
              "runtime_info": info, "rows": rows}
    del runtime
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    memory.append(memory_snapshot(f"{label}_released"))
    return result, info, memory


def benchmark_mode(args) -> dict:
    configs = []
    for mixed in (False, True):
        for persistent in (False, True):
            configs.append((mixed, persistent))
    results = []
    memory = [memory_snapshot("benchmark_start")]
    for mixed, persistent in configs:
        result, info, snapshots = bench_one(args, mixed, persistent)
        results.append(result)
        memory.extend(snapshots)
    summary = {result["label"]: result["rows"] for result in results}
    gaps = {}
    for seq_len in map(str, args.prefill_lengths):
        old_gap_prefill = (summary["mixed_legacy"][seq_len]["prefill_ttft_ms"]["median_ms"]
                           - summary["fp16_legacy"][seq_len]["prefill_ttft_ms"]["median_ms"])
        new_gap_prefill = (summary["mixed_persistent"][seq_len]["prefill_ttft_ms"]["median_ms"]
                           - summary["fp16_persistent"][seq_len]["prefill_ttft_ms"]["median_ms"])
        old_gap_decode = (summary["mixed_legacy"][seq_len]["decode_tpot_ms"]["median_ms"]
                          - summary["fp16_legacy"][seq_len]["decode_tpot_ms"]["median_ms"])
        new_gap_decode = (summary["mixed_persistent"][seq_len]["decode_tpot_ms"]["median_ms"]
                          - summary["fp16_persistent"][seq_len]["decode_tpot_ms"]["median_ms"])
        gaps[f"prefill_S{seq_len}"] = {
            "old_gap_ms": old_gap_prefill, "new_gap_ms": new_gap_prefill,
            "gap_recovery": (old_gap_prefill - new_gap_prefill) / old_gap_prefill,
        }
        gaps[f"decode_S{seq_len}"] = {
            "old_gap_ms": old_gap_decode, "new_gap_ms": new_gap_decode,
            "gap_recovery": (old_gap_decode - new_gap_decode) / old_gap_decode,
        }
    return {"phase": "Phase 3-B3", "results": results, "summary": summary,
            "gap_recovery": gaps, "memory": memory}


def profile_workload(args) -> dict:
    mixed = args.profile_runtime == "mixed"
    persistent = args.lifetime == "persistent_context_lifetime"
    engine_dir = args.mixed_dir if mixed else args.fp16_dir
    label = f"{args.profile_runtime}_{args.lifetime}"
    with nvtx_range("PHASE3B_INIT", True):
        runtime, info = load_pipeline(engine_dir, persistent, mixed, args, label)
    embed, logits = runtime["embed"]
    ids = args.bench_ids[:args.profile_seq_len]
    with nvtx_range("PHASE3B_WARMUP", True):
        for _ in range(2):
            hidden, _, _ = F.run_prefill(runtime["pre"], embed, ids)
            _ = logits(hidden[:, -1:, :])
        torch.cuda.synchronize()
    with nvtx_range("PHASE3B_STEADY_PREFILL_S8", True):
        hidden, cache_k, cache_v = F.run_prefill(runtime["pre"], embed, ids)
        _ = logits(hidden[:, -1:, :])
    torch.cuda.synchronize()
    for step in range(args.profile_decode_steps):
        with nvtx_range(f"PHASE3B_STEADY_DECODE_STEP_{step}", True):
            hidden, cache_k, cache_v = F.run_decode(
                runtime["dec"], embed, int(args.force_cont[step % len(args.force_cont)]),
                args.profile_seq_len + step, cache_k, cache_v)
            _ = logits(hidden)
        torch.cuda.synchronize()
    info["post_workload_context_counters"] = lifetime_counters(runtime)
    result = {"profiled": True, "label": label, "runtime": args.profile_runtime,
              "lifetime": args.lifetime, "mixed": mixed, "persistent": persistent,
              "engine_dir": str(engine_dir), "seq_len": args.profile_seq_len,
              "decode_steps": args.profile_decode_steps,
              "context_creations": info["engine_context_creations"],
              "initialization_s": info["initialization_s"]}
    del runtime
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["functional", "bench", "profile"],
                        default="functional")
    parser.add_argument("--lifetime", choices=["legacy_context_lifetime",
                                                "persistent_context_lifetime"],
                        default="legacy_context_lifetime")
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
    parser.add_argument("--functional-decode-steps", type=int, default=8)
    parser.add_argument("--profile-runtime", choices=["fp16", "mixed"], default="mixed")
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
    F.dump(args.out / "run_config.json", {
        "mode": args.mode, "lifetime": args.lifetime,
        "prefill_lengths": args.prefill_lengths, "warmup": args.warmup,
        "repeats": args.repeats, "window_repeats": args.window_repeats,
        "decode_steps": args.decode_steps,
        "functional_decode_steps": args.functional_decode_steps,
        "workload": "deterministic first evaluation prompt ids, batch 1",
        "sampling": "forced deterministic continuation tokens",
        "timing": "time.perf_counter with torch.cuda.synchronize before/after",
    })
    if args.mode == "functional":
        result = functional_mode(args)
    elif args.mode == "bench":
        result = benchmark_mode(args)
    else:
        result = profile_workload(args)
    F.dump(args.out / f"{args.mode}_result.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
