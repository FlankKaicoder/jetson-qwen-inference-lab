from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


WEIGHT_KEY = "model.norm.weight"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def proc_meminfo() -> dict[str, int]:
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in wanted:
            result[name] = int(value.strip().split()[0]) * 1024
    return result


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    return x_f32 * torch.rsqrt(variance + eps) * weight.float()


def rmsnorm_baseline(x: torch.Tensor, weight: torch.Tensor, hidden_size: int, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (hidden_size,), weight, eps)


def metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    a = actual.detach().float().cpu()
    b = expected.detach().float().cpu()
    diff = a - b
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "finite": bool(torch.isfinite(a).all()),
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "relative_l2": float(torch.linalg.vector_norm(diff) / torch.clamp(norm_a, min=tiny)),
        "cosine": float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(norm_a * norm_b, min=tiny)),
    }


def cuda_memory_snapshot() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda_free_bytes": int(free),
        "cuda_total_bytes": int(total),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def summarize(values_ms: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values_ms)
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "stddev_ms": statistics.pstdev(values_ms),
        "cv": statistics.pstdev(values_ms) / mean if mean else 0.0,
    }


def benchmark_config(
    case: str,
    x_cpu: torch.Tensor,
    weight_cpu: torch.Tensor,
    hidden_size: int,
    eps: float,
    dtype: torch.dtype,
    warmup: int,
    repetitions: int,
    trials: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x = x_cpu.to(device=device, dtype=dtype)
    weight = weight_cpu.to(device=device, dtype=dtype)
    expected = rmsnorm_reference(x, weight, eps)
    actual = rmsnorm_baseline(x, weight, hidden_size, eps)
    torch.cuda.synchronize()

    memory_before = cuda_memory_snapshot()
    input_bytes = x.numel() * x.element_size()
    weight_bytes = weight.numel() * weight.element_size()
    output_bytes = actual.numel() * actual.element_size()

    for _ in range(warmup):
        rmsnorm_baseline(x, weight, hidden_size, eps)
    torch.cuda.synchronize()

    trial_distributions: list[dict[str, object]] = []
    all_call_times: list[float] = []
    amortized_event_times: list[float] = []
    amortized_submit_times: list[float] = []
    amortized_wall_times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for trial_index in range(trials):
        call_times: list[float] = []
        for _ in range(repetitions):
            start.record()
            rmsnorm_baseline(x, weight, hidden_size, eps)
            end.record()
            torch.cuda.synchronize()
            call_times.append(start.elapsed_time(end))
        all_call_times.extend(call_times)
        trial_distributions.append(
            {
                "case": case,
                "dtype": str(dtype).replace("torch.", ""),
                "trial": trial_index,
                **summarize(call_times),
                "all_calls_ms": call_times,
            }
        )

    for _ in range(trials):
        torch.cuda.synchronize()
        start.record()
        for _ in range(repetitions):
            rmsnorm_baseline(x, weight, hidden_size, eps)
        end.record()
        torch.cuda.synchronize()
        amortized_event_times.append(start.elapsed_time(end) / repetitions)

        torch.cuda.synchronize()
        submit_start_ns = time.perf_counter_ns()
        for _ in range(repetitions):
            rmsnorm_baseline(x, weight, hidden_size, eps)
        submit_end_ns = time.perf_counter_ns()
        torch.cuda.synchronize()
        wall_end_ns = time.perf_counter_ns()
        amortized_submit_times.append((submit_end_ns - submit_start_ns) / repetitions / 1e6)
        amortized_wall_times.append((wall_end_ns - submit_start_ns) / repetitions / 1e6)

    memory_after = cuda_memory_snapshot()
    result = {
        "dtype": str(dtype).replace("torch.", ""),
        "shape": list(x.shape),
        "warmup_calls": warmup,
        "repetitions_per_trial": repetitions,
        "trials": trials,
        "timing_method": "CUDA event pair around each PyTorch F.rms_norm call; host synchronization after each call",
        "amortized_timing_method": "five trials of repetitions calls between one CUDA event pair; host loop also records submit-only and submit-plus-drain wall time",
        "correctness_vs_fp32_reduction": metric(actual, expected),
        "reference_is_not_tensorrt_baseline": True,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "tensor_bytes": {
            "input": input_bytes,
            "weight": weight_bytes,
            "output": output_bytes,
        },
        "trial_summary": trial_distributions,
        "aggregate": summarize(all_call_times),
        "amortized_event_aggregate": summarize(amortized_event_times),
        "amortized_host_submit_aggregate": summarize(amortized_submit_times),
        "amortized_host_wall_aggregate": summarize(amortized_wall_times),
    }
    return result, trial_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise RuntimeError(f"OUTPUT_DIR_ALREADY_EXISTS:{args.output_dir}")
    args.output_dir.mkdir(parents=True)

    config_path = args.model_dir / "config.json"
    checkpoint_path = args.model_dir / "model.safetensors"
    config = json.loads(config_path.read_text())
    hidden_size = int(config["hidden_size"])
    eps = float(config["rms_norm_eps"])

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        weight_cpu = checkpoint.get_tensor(WEIGHT_KEY)

    torch.manual_seed(args.seed)
    generator_device = torch.device("cpu")
    prefill_cpu = torch.randn((1, 8, hidden_size), device=generator_device, dtype=torch.float32)
    decode_cpu = torch.randn((1, 1, hidden_size), device=generator_device, dtype=torch.float32)

    results: list[dict[str, object]] = []
    trials: list[dict[str, object]] = []
    for name, x_cpu in (("prefill_s8", prefill_cpu), ("decode_s1", decode_cpu)):
        for dtype in (torch.bfloat16, torch.float16):
            result, distributions = benchmark_config(
                case=name,
                x_cpu=x_cpu,
                weight_cpu=weight_cpu,
                hidden_size=hidden_size,
                eps=eps,
                dtype=dtype,
                warmup=args.warmup,
                repetitions=args.repetitions,
                trials=args.trials,
            )
            result["case"] = name
            results.append(result)
            trials.extend(distributions)

    payload = {
        "experiment": "phase8_0_rmsnorm_pytorch_baseline",
        "status": "PASS",
        "scope": "PyTorch RMSNorm baseline only; no CUDA kernel, TensorRT engine change, or optimization",
        "model_dir": str(args.model_dir),
        "weight_key": WEIGHT_KEY,
        "config": {
            "hidden_size": hidden_size,
            "rms_norm_eps": eps,
            "num_hidden_layers": int(config["num_hidden_layers"]),
            "torch_dtype": config.get("torch_dtype"),
        },
        "implementation": "torch.nn.functional.rms_norm",
        "reference": "explicit FP32 reduction and multiplication",
        "seed": args.seed,
        "results": results,
    }
    write_json(args.output_dir / "rmsnorm_baseline_results.json", payload)

    with (args.output_dir / "rmsnorm_baseline_trials.csv").open("w", encoding="ascii") as output:
        output.write("case,dtype,trial,mean_ms,median_ms,min_ms,max_ms,stddev_ms,cv\n")
        for row in trials:
            output.write(
                f"{row.get('case', '')},{row.get('dtype', '')},{row['trial']},"
                f"{row['mean_ms']:.9f},{row['median_ms']:.9f},{row['min_ms']:.9f},"
                f"{row['max_ms']:.9f},{row['stddev_ms']:.9f},{row['cv']:.9f}\n"
            )

    with (args.output_dir / "rmsnorm_baseline_summary.csv").open("w", encoding="ascii") as output:
        output.write(
            "case,dtype,shape,event_percall_mean_ms,event_amortized_mean_ms,host_submit_mean_ms,host_wall_mean_ms,max_abs,relative_l2,cosine,finite\n"
        )
        for row in results:
            correctness = row["correctness_vs_fp32_reduction"]
            assert isinstance(correctness, dict)
            event_amortized = row["amortized_event_aggregate"]
            host_submit = row["amortized_host_submit_aggregate"]
            host_wall = row["amortized_host_wall_aggregate"]
            assert all(isinstance(value, dict) for value in (event_amortized, host_submit, host_wall))
            output.write(
                f"{row['case']},{row['dtype']},{row['shape']},{row['aggregate']['mean_ms']:.9f},"
                f"{event_amortized['mean_ms']:.9f},{host_submit['mean_ms']:.9f},{host_wall['mean_ms']:.9f},"
                f"{correctness['max_abs']:.12g},"
                f"{correctness['relative_l2']:.12g},{correctness['cosine']:.12g},{correctness['finite']}\n"
            )

    free, total = torch.cuda.mem_get_info()
    environment = {
        "collected_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": torch.cuda.get_device_capability(0),
        "cuda_free_bytes": int(free),
        "cuda_total_bytes": int(total),
        "host_meminfo": proc_meminfo(),
    }
    write_json(args.output_dir / "environment.json", environment)


if __name__ == "__main__":
    main()
