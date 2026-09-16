#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt
import torch


EXPECTED_CONFIG_SHA256 = (
    "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"
)
EXPECTED_WEIGHTS_SHA256 = (
    "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
)
EXPECTED_ENGINE_SHA256 = (
    "aa5c200eb5dcbc5abaef55bc014b5210236fb9ca34394217a024d3796e44823c"
)
OUTPUT_NAMES = [
    "final_hidden",
    "deepstack_0",
    "deepstack_1",
    "deepstack_2",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latency_summary(values_ms: list[float]) -> dict:
    return {
        "sample_count": len(values_ms),
        "mean_ms": statistics.mean(values_ms),
        "median_ms": statistics.median(values_ms),
        "stddev_ms": statistics.stdev(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def rate_summary(values: list[float], unit: str) -> dict:
    return {
        "sample_count": len(values),
        "unit": unit,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def finite_summary(tensor) -> dict:
    values = tensor.detach().float()
    finite_count = int(torch.isfinite(values).sum().item())
    element_count = values.numel()
    return {
        "shape": list(values.shape),
        "element_count": element_count,
        "finite_count": finite_count,
        "nonfinite_count": element_count - finite_count,
        "finite_ratio": finite_count / element_count if element_count else 0.0,
        "nan_count": int(torch.isnan(values).sum().item()),
        "positive_inf_count": int(torch.isposinf(values).sum().item()),
        "negative_inf_count": int(torch.isneginf(values).sum().item()),
        "all_finite": finite_count == element_count,
    }


def cuda_memory_snapshot() -> dict:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def start_tegrastats(log_path: Path, interval_ms: int) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "tegrastats",
            "--interval",
            str(interval_ms),
            "--logfile",
            str(log_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_tegrastats(log_path: Path) -> dict:
    if not log_path.exists():
        return {
            "status": "TEGRASTATS_LOG_MISSING",
            "sample_count": 0,
        }
    samples = {
        "VDD_IN": [],
        "VDD_CPU_GPU_CV": [],
        "VDD_SOC": [],
    }
    for line in log_path.read_text(errors="replace").splitlines():
        for key in samples:
            match = re.search(rf"{key}\s+(\d+)mW/", line)
            if match:
                samples[key].append(int(match.group(1)))

    output = {
        "status": "PASS" if samples["VDD_IN"] else "NO_SAMPLES",
        "sample_count": len(samples["VDD_IN"]),
    }
    for key, values in samples.items():
        if not values:
            output[key] = {
                "sample_count": 0,
                "status": "NO_SAMPLES",
            }
            continue
        output[key] = {
            "sample_count": len(values),
            "min_mw": min(values),
            "mean_mw": statistics.mean(values),
            "median_mw": statistics.median(values),
            "max_mw": max(values),
            "stddev_mw": statistics.stdev(values),
        }
    return output


class RuntimeLogger(trt.ILogger):
    def __init__(self):
        super().__init__()
        self.records = []

    def log(self, severity, message):
        try:
            severity_name = severity.name
        except AttributeError:
            severity_name = str(severity)
        self.records.append(
            {
                "severity": severity_name,
                "message": message,
            }
        )
        print(f"[TRT:{severity_name}] {message}", flush=True)


def timed_trt_execute(context, pixel_values, output_tensors) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    execute_returned = context.execute_async_v3(
        torch.cuda.current_stream().cuda_stream
    )
    end.record()
    torch.cuda.synchronize()
    if not execute_returned:
        raise RuntimeError("TensorRT execute_async_v3 returned False")
    return start.elapsed_time(end)


def run_pytorch_backend(pixel_values, grid_thw, config, model_path: Path) -> dict:
    import safetensors
    import transformers
    from safetensors import safe_open
    from transformers.models.qwen3_vl import Qwen3VLVisionModel

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_before = cuda_memory_snapshot()
    visual_config = config.vision_config
    visual_config._attn_implementation = "eager"
    visual = Qwen3VLVisionModel(config=visual_config)

    state = {}
    with safe_open(
        str(model_path / "model.safetensors"), framework="pt", device="cpu"
    ) as checkpoint:
        for key in checkpoint.keys():
            if key.startswith("model.visual."):
                state[key[len("model.visual.") :]] = checkpoint.get_tensor(key)
    load_result = visual.load_state_dict(state, strict=True)
    del state
    gc.collect()
    visual.to(device=pixel_values.device, dtype=torch.float16)
    visual.eval()

    samples_ms = []
    images_per_second = []
    vision_tokens_per_second = []
    with torch.no_grad():
        for warmup_index in range(3):
            visual(pixel_values, grid_thw)
            torch.cuda.synchronize()
        for trial_index in range(30):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            final_hidden, deepstack_outputs = visual(pixel_values, grid_thw)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            samples_ms.append(elapsed_ms)
            images_per_second.append(1000.0 / elapsed_ms)
            vision_tokens_per_second.append(196000.0 / elapsed_ms)

    output_finite = {
        "final_hidden": finite_summary(final_hidden),
    }
    for name, tensor in zip(
        OUTPUT_NAMES[1:], deepstack_outputs, strict=True
    ):
        output_finite[name] = finite_summary(tensor)
    memory_after = cuda_memory_snapshot()

    result = {
        "backend": "pytorch_fp16",
        "class": visual.__class__.__name__,
        "parameter_count": sum(parameter.numel() for parameter in visual.parameters()),
        "attention_implementation": visual_config._attn_implementation,
        "state_dict_loaded": True,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "blocks": len(visual.blocks),
        "warmup_trials": 3,
        "measured_trials": 30,
        "latency_ms_samples": samples_ms,
        "latency_ms": latency_summary(samples_ms),
        "images_per_second": rate_summary(images_per_second, "images/s"),
        "vision_tokens_per_second": rate_summary(
            vision_tokens_per_second, "vision_tokens/s"
        ),
        "last_output_finite": output_finite,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    del visual, final_hidden, deepstack_outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["memory_after_cleanup"] = cuda_memory_snapshot()
    result["runtime_versions"] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    return result


def run_tensorrt_backend(pixel_values, engine_path: Path) -> dict:
    import tensorrt as trt

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_before = cuda_memory_snapshot()
    runtime_logger = RuntimeLogger()
    runtime = trt.Runtime(runtime_logger)
    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    del engine_bytes
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization returned None")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT execution context creation returned None")

    input_name = "pixel_values"
    if not context.set_input_shape(input_name, tuple(pixel_values.shape)):
        raise RuntimeError("TensorRT set_input_shape returned False")
    context.set_tensor_address(input_name, int(pixel_values.data_ptr()))
    output_tensors = {}
    for name in OUTPUT_NAMES:
        shape = tuple(context.get_tensor_shape(name))
        output_tensors[name] = torch.empty(
            shape, dtype=torch.float16, device=pixel_values.device
        )
        context.set_tensor_address(name, int(output_tensors[name].data_ptr()))
    if not context.all_binding_shapes_specified:
        raise RuntimeError("TensorRT binding shapes were not fully specified")

    for _ in range(3):
        timed_trt_execute(context, pixel_values, output_tensors)

    samples_ms = []
    images_per_second = []
    vision_tokens_per_second = []
    for _ in range(30):
        elapsed_ms = timed_trt_execute(context, pixel_values, output_tensors)
        samples_ms.append(elapsed_ms)
        images_per_second.append(1000.0 / elapsed_ms)
        vision_tokens_per_second.append(196000.0 / elapsed_ms)

    output_finite = {
        name: finite_summary(output_tensors[name]) for name in OUTPUT_NAMES
    }
    memory_after = cuda_memory_snapshot()
    engine_metadata = {
        "name": engine.name,
        "num_layers": engine.num_layers,
        "num_io_tensors": engine.num_io_tensors,
        "num_optimization_profiles": engine.num_optimization_profiles,
        "device_memory_size_v2": engine.device_memory_size_v2,
    }
    log_counts = {}
    for record in runtime_logger.records:
        severity = record["severity"]
        log_counts[severity] = log_counts.get(severity, 0) + 1
    error_count = sum(
        1 for record in runtime_logger.records if record["severity"] == "ERROR"
    )

    result = {
        "backend": "tensorrt_fp16",
        "engine_metadata": engine_metadata,
        "warmup_trials": 3,
        "measured_trials": 30,
        "latency_ms_samples": samples_ms,
        "latency_ms": latency_summary(samples_ms),
        "images_per_second": rate_summary(images_per_second, "images/s"),
        "vision_tokens_per_second": rate_summary(
            vision_tokens_per_second, "vision_tokens/s"
        ),
        "last_output_finite": output_finite,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "tensorrt_runtime_log": {
            "record_count": len(runtime_logger.records),
            "counts_by_severity": log_counts,
            "error_count": error_count,
        },
    }
    del context, engine, runtime
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["memory_after_cleanup"] = cuda_memory_snapshot()
    result["runtime_versions"] = {
        "tensorrt": trt.__version__,
    }
    return result


def backend_with_power(power_log_path: Path, interval_ms: int, runner, *args) -> tuple:
    process = start_tegrastats(power_log_path, interval_ms)
    power_started_utc = utc_now()
    try:
        result = runner(*args)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    power_finished_utc = utc_now()
    result["power"] = {
        "started_utc": power_started_utc,
        "finished_utc": power_finished_utc,
        "scope": "backend_setup_warmup_and_measurement",
        "summary": parse_tegrastats(power_log_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--engine-path", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--power-log-pytorch", required=True)
    parser.add_argument("--power-log-tensorrt", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    engine_path = Path(args.engine_path).resolve()
    protocol_path = Path(args.protocol).resolve()
    result_path = Path(args.result_path).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("protocol_state") != "FROZEN_BEFORE_MEASUREMENT":
        raise RuntimeError("benchmark protocol is not frozen")

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(device)
    torch.cuda.synchronize()

    result = {
        "phase": "Phase 9.2-D1",
        "title": "Isolated Qwen3-VL Vision Encoder latency benchmark",
        "started_utc": utc_now(),
        "hostname": platform.node(),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "frozen_at_utc": protocol["frozen_at_utc"],
            "timing": protocol["timing"],
            "power": protocol["power"],
        },
        "model_identity": {
            "identity": protocol["model"]["identity"],
            "revision": protocol["model"]["revision"],
            "local_path": str(model_path),
            "config_sha256": sha256_file(model_path / "config.json"),
            "weights_sha256": sha256_file(model_path / "model.safetensors"),
            "weights_size_bytes": (model_path / "model.safetensors").stat().st_size,
        },
        "engine_file": {
            "path": str(engine_path),
            "size_bytes": engine_path.stat().st_size,
            "sha256": sha256_file(engine_path),
        },
        "boundary": protocol["workload"],
        "prohibited_operations_performed": {
            "optimization": False,
            "tensorrt_engine_rebuild": False,
            "quantization": False,
            "model_change": False,
            "cuda_modification": False,
            "environment_modification": False,
        },
        "backends": {},
    }
    try:
        if result["model_identity"]["config_sha256"] != EXPECTED_CONFIG_SHA256:
            raise RuntimeError("pinned config SHA-256 mismatch")
        if result["model_identity"]["weights_sha256"] != EXPECTED_WEIGHTS_SHA256:
            raise RuntimeError("pinned weights SHA-256 mismatch")
        if result["engine_file"]["sha256"] != EXPECTED_ENGINE_SHA256:
            raise RuntimeError("unchanged TensorRT engine SHA-256 mismatch")

        import transformers
        from transformers.models.qwen3_vl import Qwen3VLConfig

        result["runtime_dependency_status"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "total_memory_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
            "allocated_before_bytes": torch.cuda.memory_allocated(),
            "reserved_before_bytes": torch.cuda.memory_reserved(),
        }

        fp32_source = torch.linspace(
            -1.0,
            1.0,
            784 * 1536,
            device=device,
            dtype=torch.float32,
        ).reshape(784, 1536)
        pixel_values = fp32_source.to(dtype=torch.float16)
        result["input"] = {
            "fp32_source": finite_summary(fp32_source),
            "fp16_model_input": finite_summary(pixel_values),
        }
        del fp32_source
        result["input"]["fp32_source"]["dtype"] = "torch.float32"
        result["input"]["fp32_source"]["shape"] = [784, 1536]
        result["input"]["fp16_model_input"]["dtype"] = str(pixel_values.dtype)
        if not result["input"]["fp16_model_input"]["all_finite"]:
            raise RuntimeError("benchmark input is non-finite")

        config = Qwen3VLConfig.from_pretrained(str(model_path))
        grid_thw = torch.tensor(
            [[1, 28, 28]], device=device, dtype=torch.long
        )

        pytorch_result = backend_with_power(
            Path(args.power_log_pytorch).resolve(),
            protocol["power"]["interval_ms"],
            run_pytorch_backend,
            pixel_values,
            grid_thw,
            config,
            model_path,
        )
        result["backends"]["pytorch_fp16"] = pytorch_result

        tensorrt_result = backend_with_power(
            Path(args.power_log_tensorrt).resolve(),
            protocol["power"]["interval_ms"],
            run_tensorrt_backend,
            pixel_values,
            engine_path,
        )
        result["backends"]["tensorrt_fp16"] = tensorrt_result

        pytorch_latency = pytorch_result["latency_ms"]
        tensorrt_latency = tensorrt_result["latency_ms"]
        result["comparison_summary"] = {
            "mean_latency_delta_ms_tensorrt_minus_pytorch": (
                tensorrt_latency["mean_ms"] - pytorch_latency["mean_ms"]
            ),
            "median_latency_delta_ms_tensorrt_minus_pytorch": (
                tensorrt_latency["median_ms"] - pytorch_latency["median_ms"]
            ),
            "mean_latency_ratio_tensorrt_over_pytorch": (
                tensorrt_latency["mean_ms"] / pytorch_latency["mean_ms"]
            ),
            "median_latency_ratio_tensorrt_over_pytorch": (
                tensorrt_latency["median_ms"] / pytorch_latency["median_ms"]
            ),
            "interpretation": "descriptive fixed-boundary comparison only; no optimization claim",
        }
        result["success"] = True
    except Exception as exc:
        result["success"] = False
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    result["finished_utc"] = utc_now()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
