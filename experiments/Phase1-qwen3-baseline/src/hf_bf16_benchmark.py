#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REPO = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
OSL = 32
MEM_AVAILABLE_MIN = int(2.5 * 1024**3)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def meminfo():
    values = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = int(value.strip().split()[0]) * 1024
    values["SwapUsed"] = values["SwapTotal"] - values["SwapFree"]
    return values


def memory_snapshot():
    system = meminfo()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "timestamp_ns": time.time_ns(),
        "system": system,
        "torch_cuda_allocator": {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        },
        "cuda_mem_info": {"free": free_bytes, "total": total_bytes},
    }


def power_mode():
    completed = subprocess.run(
        ["sudo", "-n", "/usr/sbin/nvpmodel", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def cache_length(cache):
    if not hasattr(cache, "get_seq_length"):
        raise RuntimeError(f"cache class has no get_seq_length(): {type(cache).__name__}")
    return int(cache.get_seq_length())


def stats(values):
    ordered = sorted(values)
    count = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    p90_index = max(0, math.ceil(0.9 * count) - 1)
    return {
        "count": count,
        "mean": mean,
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "std": std,
        "cv_percent": 100.0 * std / mean if mean else 0.0,
        "p90": ordered[p90_index],
    }


def exact_input(tokenizer, isl):
    seed_text = "Hello world. "
    seed_ids = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    if not seed_ids:
        raise RuntimeError("seed text produced no tokens")
    repeated = (seed_ids * math.ceil(isl / len(seed_ids)))[:isl]
    if len(repeated) != isl or any(token < 0 or token >= len(tokenizer) for token in repeated):
        raise RuntimeError("invalid exact input construction")
    return torch.tensor([repeated], dtype=torch.long, device="cuda:0"), {
        "algorithm": "tokenize 'Hello world. ' without special tokens, repeat, truncate to exact ISL",
        "seed_text": seed_text,
        "seed_token_count": len(seed_ids),
        "actual_isl": len(repeated),
        "padding": False,
    }


def load_runtime(snapshot):
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    torch.cuda.synchronize()
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if devices != ["cuda:0"] or dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"unexpected model placement: devices={devices}, dtypes={dtypes}")
    return tokenizer, model


def run_request(model, input_ids, osl=OSL, capture_steps=False):
    if input_ids.shape[0] != 1 or input_ids.shape[1] < 1 or osl < 2:
        raise ValueError("batch 1, nonempty input, and OSL >= 2 are required")
    isl = int(input_ids.shape[1])
    attention_mask = torch.ones_like(input_ids)
    memory_steps = []
    cache_lengths = []
    decode_gpu_ms = []
    decode_wall_ms = []
    output_ids = []

    torch.cuda.synchronize()
    wall_start_ns = time.perf_counter_ns()
    epoch_start_ns = time.time_ns()
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_start.record()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    prefill_end.record()
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    past = outputs.past_key_values
    torch.cuda.synchronize()
    inference_ttft_ms = (time.perf_counter_ns() - wall_start_ns) / 1e6
    prefill_gpu_ms = prefill_start.elapsed_time(prefill_end)
    cache_lengths.append(cache_length(past))
    output_ids.append(int(next_token.item()))
    if capture_steps:
        memory_steps.append({"step": "prefill", **memory_snapshot()})

    for decode_index in range(1, osl):
        attention_mask = torch.ones((1, isl + decode_index), dtype=torch.long, device="cuda:0")
        torch.cuda.synchronize()
        step_wall_start = time.perf_counter_ns()
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record()
        with torch.inference_mode():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
        step_end.record()
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        past = outputs.past_key_values
        torch.cuda.synchronize()
        decode_wall_ms.append((time.perf_counter_ns() - step_wall_start) / 1e6)
        decode_gpu_ms.append(step_start.elapsed_time(step_end))
        cache_lengths.append(cache_length(past))
        output_ids.append(int(next_token.item()))
        if capture_steps:
            memory_steps.append({"step": decode_index, **memory_snapshot()})

    e2e_ms = (time.perf_counter_ns() - wall_start_ns) / 1e6
    epoch_end_ns = time.time_ns()
    tpot_ms = (e2e_ms - inference_ttft_ms) / (osl - 1)
    direct_tpot_ms = statistics.fmean(decode_wall_ms)
    accounting_error_ms = abs((e2e_ms - inference_ttft_ms) - sum(decode_wall_ms))
    expected_cache = list(range(isl, isl + osl))
    return {
        "epoch_start_ns": epoch_start_ns,
        "epoch_end_ns": epoch_end_ns,
        "isl": isl,
        "osl": osl,
        "prefill_gpu_ms": prefill_gpu_ms,
        "prefill_tokens_per_second": isl * 1000.0 / prefill_gpu_ms,
        "inference_ttft_ms": inference_ttft_ms,
        "decode_step_gpu_ms": decode_gpu_ms,
        "decode_step_wall_ms": decode_wall_ms,
        "tpot_ms": tpot_ms,
        "direct_mean_decode_wall_ms": direct_tpot_ms,
        "decode_tokens_per_second": 1000.0 / tpot_ms,
        "e2e_ms": e2e_ms,
        "e2e_output_tokens_per_second": osl * 1000.0 / e2e_ms,
        "timing_accounting_error_ms": accounting_error_ms,
        "timing_accounting_tolerance_ms": max(5.0, e2e_ms * 0.01),
        "cache_class": type(past).__name__,
        "cache_lengths": cache_lengths,
        "cache_lengths_expected": expected_cache,
        "cache_growth_valid": cache_lengths == expected_cache,
        "fixed_output_token_ids": output_ids,
        "memory_steps": memory_steps,
    }


def validate_method(tokenizer, model):
    prompt = "What is 2 + 3? Reply with only the number."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda:0")
    attention_mask = encoded["attention_mask"].to("cuda:0")
    with torch.inference_mode():
        reference = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    reference_ids = reference[0, input_ids.shape[1] :].tolist()
    manual = run_request(model, input_ids, osl=8, capture_steps=False)
    manual_ids = manual["fixed_output_token_ids"]
    return {
        "prompt": prompt,
        "input_tokens": int(input_ids.shape[1]),
        "reference_token_ids": reference_ids,
        "manual_token_ids": manual_ids,
        "tokens_equal": reference_ids == manual_ids,
        "cache_class": manual["cache_class"],
        "cache_lengths": manual["cache_lengths"],
        "cache_lengths_expected": manual["cache_lengths_expected"],
        "cache_growth_valid": manual["cache_growth_valid"],
        "timing_accounting_error_ms": manual["timing_accounting_error_ms"],
        "timing_accounting_tolerance_ms": manual["timing_accounting_tolerance_ms"],
        "timing_accounting_valid": manual["timing_accounting_error_ms"]
        <= manual["timing_accounting_tolerance_ms"],
        "pass": reference_ids == manual_ids
        and manual["cache_growth_valid"]
        and manual["timing_accounting_error_ms"] <= manual["timing_accounting_tolerance_ms"],
    }


def summarize_trials(trials):
    metrics = {
        "prefill_gpu_ms": [trial["prefill_gpu_ms"] for trial in trials],
        "prefill_tokens_per_second": [trial["prefill_tokens_per_second"] for trial in trials],
        "inference_ttft_ms": [trial["inference_ttft_ms"] for trial in trials],
        "tpot_ms": [trial["tpot_ms"] for trial in trials],
        "decode_tokens_per_second": [trial["decode_tokens_per_second"] for trial in trials],
        "e2e_ms": [trial["e2e_ms"] for trial in trials],
        "e2e_output_tokens_per_second": [trial["e2e_output_tokens_per_second"] for trial in trials],
    }
    return {name: stats(values) for name, values in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["validate", "pilot", "benchmark"], required=True)
    parser.add_argument("--isl", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()

    initial_memory = meminfo()
    if initial_memory["MemAvailable"] < MEM_AVAILABLE_MIN:
        raise RuntimeError(f"pre-load MemAvailable below 2.5 GiB: {initial_memory['MemAvailable']}")
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "started_at": utc_now(),
        "git": {"commit": args.git_commit, "branch": args.branch},
        "platform": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_file": torch.__file__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "weight_sha256": WEIGHT_SHA256,
            "snapshot": str(Path(args.snapshot).resolve()),
            "dtype": "torch.bfloat16",
            "attention_implementation": "eager",
            "batch": 1,
        },
        "shape": {"isl": args.isl, "osl": OSL},
        "protocol": {
            "warmups": args.warmups,
            "initial_trials": args.trials,
            "stability_extension_trials": 5,
            "stability_cv_threshold_percent": 5.0,
            "fresh_process": True,
            "fixed_output_length_despite_eos": True,
            "selection": "greedy argmax",
            "gpu_timing": "CUDA Events around model forward only",
            "wall_timing": "perf_counter_ns with CUDA synchronization at boundaries",
        },
        "power_mode_start": power_mode(),
        "memory_before_model_load": initial_memory,
    }
    tokenizer, model = load_runtime(args.snapshot)
    input_ids, construction = exact_input(tokenizer, args.isl)
    result["input_construction"] = construction
    result["model_loaded_at_ns"] = time.time_ns()
    result["memory_after_model_load"] = memory_snapshot()
    result["loaded_idle_start_ns"] = time.time_ns()
    time.sleep(5)
    result["loaded_idle_end_ns"] = time.time_ns()

    if args.mode == "validate":
        result["method_validation"] = validate_method(tokenizer, model)
        if not result["method_validation"]["pass"]:
            raise RuntimeError("manual-loop method validation failed")
    elif args.mode == "pilot":
        torch.cuda.reset_peak_memory_stats()
        baseline = memory_snapshot()
        pilot = run_request(model, input_ids, capture_steps=True)
        final = memory_snapshot()
        observed = [baseline, final] + [entry for entry in pilot["memory_steps"]]
        minimum_available = min(entry["system"]["MemAvailable"] for entry in observed)
        maximum_swap = max(entry["system"]["SwapUsed"] for entry in observed)
        swap_growth = maximum_swap - initial_memory["SwapUsed"]
        result["pilot"] = pilot
        result["pilot_memory_gate"] = {
            "minimum_memavailable_bytes": minimum_available,
            "maximum_swap_used_bytes": maximum_swap,
            "swap_growth_from_preload_bytes": swap_growth,
            "memavailable_threshold_bytes": int(1.5 * 1024**3),
            "swap_growth_threshold_bytes": 512 * 1024**2,
            "pass": minimum_available >= int(1.5 * 1024**3) and swap_growth <= 512 * 1024**2,
        }
    else:
        result["warmup_start_ns"] = time.time_ns()
        for _ in range(args.warmups):
            warmup = run_request(model, input_ids)
            if not warmup["cache_growth_valid"]:
                raise RuntimeError("warmup cache growth failed")
        result["warmup_end_ns"] = time.time_ns()
        result["formal_start_ns"] = time.time_ns()
        trials = []
        for trial_id in range(1, args.trials + 1):
            torch.cuda.reset_peak_memory_stats()
            before = memory_snapshot()
            trial = run_request(model, input_ids)
            trial["trial_id"] = trial_id
            trial["memory_before"] = before
            trial["memory_after"] = memory_snapshot()
            if not trial["cache_growth_valid"] or len(trial["fixed_output_token_ids"]) != OSL:
                raise RuntimeError("formal trial cache or OSL validation failed")
            trials.append(trial)
        initial_summary = summarize_trials(trials)
        extended = initial_summary["inference_ttft_ms"]["cv_percent"] > 5.0 or initial_summary["tpot_ms"]["cv_percent"] > 5.0
        if extended:
            for trial_id in range(args.trials + 1, args.trials + 6):
                torch.cuda.reset_peak_memory_stats()
                before = memory_snapshot()
                trial = run_request(model, input_ids)
                trial["trial_id"] = trial_id
                trial["memory_before"] = before
                trial["memory_after"] = memory_snapshot()
                if not trial["cache_growth_valid"] or len(trial["fixed_output_token_ids"]) != OSL:
                    raise RuntimeError("extended trial cache or OSL validation failed")
                trials.append(trial)
        result["formal_end_ns"] = time.time_ns()
        result["trials"] = trials
        result["stability_extension_applied"] = extended
        result["summary"] = summarize_trials(trials)
        result["stability"] = {
            "ttft_cv_percent": result["summary"]["inference_ttft_ms"]["cv_percent"],
            "tpot_cv_percent": result["summary"]["tpot_ms"]["cv_percent"],
            "trial_count": len(trials),
            "stable": result["summary"]["inference_ttft_ms"]["cv_percent"] <= 5.0
            and result["summary"]["tpot_ms"]["cv_percent"] <= 5.0,
        }

    result["memory_at_end"] = memory_snapshot()
    result["power_mode_end"] = power_mode()
    result["completed_at"] = utc_now()
    result["status"] = "PASS"
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "mode": args.mode, "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
