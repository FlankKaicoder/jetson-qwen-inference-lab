#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt
import torch
import torch.cuda.nvtx as nvtx
from torch import nn


EXPECTED_CONFIG_SHA256 = (
    "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"
)
EXPECTED_WEIGHTS_SHA256 = (
    "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
)
EXPECTED_ENGINE_SHA256 = (
    "aa5c200eb5dcbc5abaef55bc014b5210236fb9ca34394217a024d3796e44823c"
)
OUTPUT_NAMES = ["final_hidden", "deepstack_0", "deepstack_1", "deepstack_2"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return -1


def cuda_memory_snapshot() -> dict:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "host_mem_available_bytes": mem_available_bytes(),
    }


def latency_summary(values_ms: list[float]) -> dict:
    return {
        "sample_count": len(values_ms),
        "mean_ms": statistics.mean(values_ms),
        "median_ms": statistics.median(values_ms),
        "stddev_ms": statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def optional_latency_summary(values: list[float | None]) -> dict | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return latency_summary(usable)


def tensor_facts(value) -> dict:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "is_finite": bool(torch.isfinite(value.float()).all().item()),
    }


class EventTimer:
    def __init__(self, name: str):
        self.name = name
        self.samples: list[tuple] = []
        self._active_start = None
        self.handles: list = []

    def register(self, module) -> None:
        self.handles.append(module.register_forward_pre_hook(self._pre_hook))
        self.handles.append(module.register_forward_hook(self._post_hook))

    def _pre_hook(self, module, args):
        self._active_start = torch.cuda.Event(enable_timing=True)
        self._active_start.record()

    def _post_hook(self, module, args, output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.samples.append((self._active_start, end))
        self._active_start = None

    def drain_ms(self) -> list[float]:
        values = [start.elapsed_time(end) for start, end in self.samples]
        self.samples.clear()
        return values

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class RuntimeLogger(trt.ILogger):
    def __init__(self):
        super().__init__()
        self.records = []

    def log(self, severity, message):
        try:
            severity_name = severity.name
        except AttributeError:
            severity_name = str(severity)
        self.records.append({"severity": severity_name, "message": message})
        print(f"[TRT:{severity_name}] {message}", flush=True)


class TensorRTVisionAdapter(nn.Module):
    def __init__(
        self,
        engine_path: Path,
        spatial_merge_size: int,
        enable_nvtx: bool,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.enable_nvtx = enable_nvtx
        self.engine_path = str(engine_path)
        self.runtime_logger = RuntimeLogger()
        self.runtime = trt.Runtime(self.runtime_logger)
        engine_bytes = engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        del engine_bytes
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization returned None")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT execution context creation returned None")
        self.input_name = "pixel_values"
        self.output_names = OUTPUT_NAMES
        self.output_tensors = {}

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16

    def prepare(self, pixel_shape: tuple[int, int]):
        if not self.context.set_input_shape(self.input_name, pixel_shape):
            raise RuntimeError("TensorRT set_input_shape returned False")
        self.output_tensors = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            self.output_tensors[name] = torch.empty(
                shape,
                dtype=torch.float16,
                device="cuda:0",
            )

    def forward(self, pixel_values, grid_thw=None, **kwargs):
        if tuple(pixel_values.shape) != (784, 1536):
            raise RuntimeError(
                f"TensorRT adapter expected [784,1536], got {tuple(pixel_values.shape)}"
            )
        if grid_thw is None or tuple(grid_thw.shape) != (1, 3) or grid_thw.tolist() != [
            [1, 28, 28]
        ]:
            raise RuntimeError("TensorRT adapter expected grid_thw=[[1,28,28]]")
        if not self.output_tensors:
            self.prepare(tuple(pixel_values.shape))
        self.context.set_tensor_address(
            self.input_name,
            int(pixel_values.data_ptr()),
        )
        for name, tensor in self.output_tensors.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        if self.enable_nvtx:
            nvtx.range_push("B2_tensorrt_visual")
        try:
            if not self.context.execute_async_v3(
                torch.cuda.current_stream().cuda_stream
            ):
                raise RuntimeError("TensorRT execute_async_v3 returned False")
            if self.enable_nvtx:
                torch.cuda.synchronize()
        finally:
            if self.enable_nvtx:
                nvtx.range_pop()
        return (
            self.output_tensors["final_hidden"],
            [
                self.output_tensors["deepstack_0"],
                self.output_tensors["deepstack_1"],
                self.output_tensors["deepstack_2"],
            ],
        )

    def metadata(self) -> dict:
        counts: dict[str, int] = {}
        for record in self.runtime_logger.records:
            severity = record["severity"]
            counts[severity] = counts.get(severity, 0) + 1
        return {
            "engine_name": self.engine.name,
            "num_layers": self.engine.num_layers,
            "num_io_tensors": self.engine.num_io_tensors,
            "num_optimization_profiles": self.engine.num_optimization_profiles,
            "device_memory_size_v2": self.engine.device_memory_size_v2,
            "log_counts_by_severity": counts,
            "error_count": sum(
                1
                for record in self.runtime_logger.records
                if record["severity"] == "ERROR"
            ),
        }


def make_image():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((112, 112, 335, 335), fill=(192, 0, 0))
    return image


def preprocess(model, processor, image, prompt: str, device: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    torch.cuda.synchronize()
    start = time.monotonic()
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise RuntimeError("processor did not produce the image input contract")
    inputs = inputs.to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    torch.cuda.synchronize()
    elapsed_ms = (time.monotonic() - start) * 1000.0
    expected_vision_tokens = (
        inputs["image_grid_thw"].prod(dim=-1)
        // model.model.visual.spatial_merge_size**2
    ).sum().item()
    image_token_count = int(
        (inputs["input_ids"] == model.config.image_token_id).sum().item()
    )
    if expected_vision_tokens != image_token_count:
        raise RuntimeError("image token count does not match visual output tokens")
    if tuple(inputs["pixel_values"].shape) != (784, 1536):
        raise RuntimeError("processor output did not match fixed vision boundary")
    return text, inputs, elapsed_ms


def inspect_cache(cache, include_finite: bool, include_layers: bool) -> dict:
    if not hasattr(cache, "layers"):
        raise RuntimeError(f"cache does not expose layers: {type(cache).__name__}")
    layers = list(cache.layers)
    layer_rows = []
    total_key_numel = 0
    total_value_numel = 0
    total_bytes = 0
    sequence_lengths = set()
    key_dtypes = set()
    value_dtypes = set()
    finite_elements = 0
    finite_total = 0

    for index, layer in enumerate(layers):
        if not hasattr(layer, "keys") or not hasattr(layer, "values"):
            raise RuntimeError(f"cache layer {index} does not expose keys/values")
        keys = layer.keys
        values = layer.values
        if keys is None or values is None:
            raise RuntimeError(f"cache layer {index} has a None key or value tensor")
        key_numel = int(keys.numel())
        value_numel = int(values.numel())
        bytes_this_layer = key_numel * keys.element_size()
        bytes_this_layer += value_numel * values.element_size()
        total_key_numel += key_numel
        total_value_numel += value_numel
        total_bytes += bytes_this_layer
        sequence_lengths.add(int(keys.shape[-2]))
        key_dtypes.add(str(keys.dtype))
        value_dtypes.add(str(values.dtype))
        if include_finite:
            finite_elements += int(torch.isfinite(keys.float()).sum().item())
            finite_elements += int(torch.isfinite(values.float()).sum().item())
            finite_total += key_numel + value_numel
        if include_layers:
            layer_rows.append(
                {
                    "layer_index": index,
                    "key_shape": list(keys.shape),
                    "value_shape": list(values.shape),
                    "key_dtype": str(keys.dtype),
                    "value_dtype": str(values.dtype),
                    "key_numel": key_numel,
                    "value_numel": value_numel,
                    "logical_bytes": bytes_this_layer,
                }
            )

    result = {
        "cache_class": type(cache).__name__,
        "layer_count": len(layers),
        "sequence_lengths": sorted(sequence_lengths),
        "key_dtypes": sorted(key_dtypes),
        "value_dtypes": sorted(value_dtypes),
        "total_key_numel": total_key_numel,
        "total_value_numel": total_value_numel,
        "total_logical_bytes": total_bytes,
    }
    if include_finite:
        result["finite_ratio"] = (
            finite_elements / finite_total if finite_total else None
        )
    if include_layers:
        result["layers"] = layer_rows
    return result


def run_trial(
    model,
    processor,
    image,
    prompt: str,
    device: str,
    max_new_tokens: int,
    profiled: bool,
):
    text, inputs, preprocess_ms = preprocess(
        model,
        processor,
        image,
        prompt,
        device,
    )
    attention_mask = inputs["attention_mask"]
    base_length = int(inputs["input_ids"].shape[1])
    eos_token_id = processor.tokenizer.eos_token_id
    token_ids: list[int] = []

    torch.cuda.synchronize()
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        if profiled:
            nvtx.range_push("B2_profile_prefill")
        prefill_start.record()
        prefill_output = model(
            **inputs,
            use_cache=True,
            logits_to_keep=1,
        )
        prefill_end.record()
        if profiled:
            torch.cuda.synchronize()
            nvtx.range_pop()
        else:
            torch.cuda.synchronize()
        prefill_ms = prefill_start.elapsed_time(prefill_end)

        prefill_logits = prefill_output.logits[:, -1, :]
        next_token = prefill_logits.argmax(dim=-1, keepdim=True)
        token_ids.append(int(next_token.item()))
        cache = prefill_output.past_key_values
        decode_output = prefill_output
        cache_after_prefill = inspect_cache(
            cache,
            include_finite=True,
            include_layers=True,
        )

        decode_event_pairs = []
        decode_step_ms = []
        if profiled:
            nvtx.range_push("B2_profile_decode")
        decode_start = time.monotonic()
        for step in range(1, max_new_tokens):
            logits = decode_output.logits[:, -1, :]
            if step < max_new_tokens - 1:
                logits[:, eos_token_id] = -float("inf")
            next_token = logits.argmax(dim=-1, keepdim=True)
            token_ids.append(int(next_token.item()))
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=torch.int64,
                        device=next_token.device,
                    ),
                ],
                dim=1,
            )
            cache_position = torch.tensor(
                [base_length + step],
                dtype=torch.int64,
                device=next_token.device,
            )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            step_name = f"B2_profile_decode_step_{step:02d}"
            if profiled:
                nvtx.range_push(step_name)
            start.record()
            decode_output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                logits_to_keep=1,
            )
            end.record()
            if profiled:
                torch.cuda.synchronize()
                nvtx.range_pop()
                decode_step_ms.append(start.elapsed_time(end))
                cache_after_step = inspect_cache(
                    cache,
                    include_finite=False,
                    include_layers=False,
                )
                cache_after_step["decode_step"] = step
                cache_after_step["cache_position"] = int(cache_position.item())
                cache_after_step["next_token_id"] = token_ids[-1]
                decode_step_ms[-1] = {
                    "decode_step": step,
                    "latency_ms": start.elapsed_time(end),
                    "cache": cache_after_step,
                }
            else:
                decode_event_pairs.append((start, end))
        torch.cuda.synchronize()
        decode_wall_ms = (time.monotonic() - decode_start) * 1000.0
        if profiled:
            nvtx.range_pop()
        else:
            for start, end in decode_event_pairs:
                decode_step_ms.append(start.elapsed_time(end))

        cache_after_decode = inspect_cache(
            cache,
            include_finite=True,
            include_layers=True,
        )

    generated_seconds = prefill_ms / 1000.0 + decode_wall_ms / 1000.0
    generated_text = processor.tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )
    decode_event_total_ms = None
    if not profiled:
        decode_event_total_ms = sum(decode_step_ms)
    return {
        "profiled": profiled,
        "generation_success": len(token_ids) == max_new_tokens,
        "generated_tokens": len(token_ids),
        "generated_token_ids": token_ids,
        "generated_text": generated_text,
        "preprocess_ms": preprocess_ms,
        "visual_total_ms": None,
        "prefill_ms": prefill_ms,
        "decode_wall_ms": decode_wall_ms,
        "decode_event_total_ms": decode_event_total_ms,
        "decode_per_token_wall_ms": decode_wall_ms / (max_new_tokens - 1),
        "generation_total_wall_ms": prefill_ms + decode_wall_ms,
        "generation_tokens_per_second": max_new_tokens / generated_seconds,
        "decode_step_timings": decode_step_ms,
        "cache_after_prefill": cache_after_prefill,
        "cache_after_decode": cache_after_decode,
        "input_ids": tensor_facts(inputs["input_ids"]),
        "pixel_values": tensor_facts(inputs["pixel_values"]),
        "image_grid_thw": tensor_facts(inputs["image_grid_thw"]),
        "prompt": text,
    }


def summarize_trials(trials: list[dict]) -> dict:
    stage_keys = [
        "preprocess_ms",
        "visual_total_ms",
        "prefill_ms",
        "decode_wall_ms",
        "decode_event_total_ms",
        "decode_per_token_wall_ms",
        "generation_total_wall_ms",
    ]
    summary = {
        "profiled": trials[0]["profiled"],
        "generation_success_count": sum(
            1 for trial in trials if trial["generation_success"]
        ),
        "sample_count": len(trials),
        "all_token_sequences_equal": all(
            trial["generated_token_ids"] == trials[0]["generated_token_ids"]
            for trial in trials
        ),
        "all_generated_texts_equal": all(
            trial["generated_text"] == trials[0]["generated_text"]
            for trial in trials
        ),
    }
    for key in stage_keys:
        summary[key] = optional_latency_summary(
            [trial.get(key) for trial in trials]
        )
    rates = [trial["generation_tokens_per_second"] for trial in trials]
    summary["generation_tokens_per_second"] = {
        "sample_count": len(rates),
        "mean": statistics.mean(rates),
        "median": statistics.median(rates),
        "stddev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "min": min(rates),
        "max": max(rates),
    }
    return summary


def attribution(summary: dict) -> dict:
    total = summary["generation_total_wall_ms"]["mean_ms"]
    prefill = summary["prefill_ms"]["mean_ms"]
    decode = summary["decode_wall_ms"]["mean_ms"]
    return {
        "prefill_share_of_generation": prefill / total,
        "decode_share_of_generation": decode / total,
        "dominant_generation_stage": "decode" if decode > prefill else "prefill",
    }


def write_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n")


def fail(path: Path, result: dict, exc: Exception):
    result["success"] = False
    result["failure"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    result["finished_utc"] = utc_now()
    write_result(path, result)
    print(json.dumps(result["failure"], indent=2))
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--engine-path", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--run-mode", choices=["latency", "profile"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--intended-nsys-report-path")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    engine_path = Path(args.engine_path).resolve()
    protocol_path = Path(args.protocol).resolve()
    result_path = Path(args.result_path).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("protocol_state") != "FROZEN_BEFORE_MEASUREMENT":
        raise RuntimeError("protocol is not frozen")

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(device)
    torch.cuda.synchronize()

    result = {
        "phase": "Phase 9.3-B2",
        "title": "Qwen3-VL decoder-side runtime bottleneck attribution",
        "run_mode": args.run_mode,
        "started_utc": utc_now(),
        "hostname": platform.node(),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tensorrt_version": trt.__version__,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "frozen_at_utc": protocol["frozen_at_utc"],
            "workload": protocol["workload"],
            "run": protocol["runs"][args.run_mode],
            "attribution_boundary": protocol["attribution_boundary"],
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
            "sha256_before": sha256_file(engine_path),
            "sha256_after": None,
        },
        "authorization_boundary": protocol["authorization"],
        "runtime_dependency_status": {},
        "model_evidence": {},
        "memory": {},
        "trials": {
            "warmup": [],
            "measured": [],
        },
        "summary": {},
        "stage_attribution": {},
        "profile_capture": {
            "intended_nsys_report_path": args.intended_nsys_report_path,
            "report_present_after_process_exit": None,
        },
        "prohibited_operations_performed": {
            "tensorrt_llm_migration": False,
            "optimization": False,
            "quantization": False,
            "decoder_modification": False,
            "cuda_modification": False,
            "tensorrt_engine_rebuild": False,
            "environment_modification": False,
            "benchmark_sweep": False,
        },
    }
    try:
        if result["model_identity"]["config_sha256"] != EXPECTED_CONFIG_SHA256:
            raise RuntimeError("pinned config SHA-256 mismatch")
        if result["model_identity"]["weights_sha256"] != EXPECTED_WEIGHTS_SHA256:
            raise RuntimeError("pinned weights SHA-256 mismatch")
        if result["engine_file"]["sha256_before"] != EXPECTED_ENGINE_SHA256:
            raise RuntimeError("unchanged TensorRT engine SHA-256 mismatch")

        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        result["runtime_dependency_status"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "allocated_before_bytes": torch.cuda.memory_allocated(),
            "reserved_before_bytes": torch.cuda.memory_reserved(),
            "host_mem_available_bytes": mem_available_bytes(),
        }

        torch.manual_seed(20260916)
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=protocol["model"]["attention_implementation"],
        )
        model.eval()
        result["model_evidence"] = {
            "load_success": True,
            "model_class": type(model).__name__,
            "base_model_class": type(model.model).__name__,
            "language_model_class": type(model.model.language_model).__name__,
            "visual_backend_class_before_injection": type(model.model.visual).__name__,
            "model_dtype": str(model.dtype),
            "attention_implementation": protocol["model"][
                "attention_implementation"
            ],
            "total_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "language_model_parameter_count": sum(
                parameter.numel()
                for parameter in model.model.language_model.parameters()
            ),
            "tokenizer_class": type(processor.tokenizer).__name__,
        }
        result["memory"]["after_model_load"] = cuda_memory_snapshot()

        original_visual = model.model.visual
        adapter = TensorRTVisionAdapter(
            engine_path,
            int(original_visual.spatial_merge_size),
            args.run_mode == "profile",
        )
        model.model.visual = adapter
        original_visual = None
        gc.collect()
        torch.cuda.empty_cache()
        result["model_evidence"]["visual_backend_class_after_injection"] = type(
            model.model.visual
        ).__name__
        result["model_evidence"]["language_model_parameter_count_after_injection"] = sum(
            parameter.numel()
            for parameter in model.model.language_model.parameters()
        )
        result["model_evidence"]["tensorrt_adapter_metadata"] = adapter.metadata()
        result["memory"]["after_tensorrt_injection"] = cuda_memory_snapshot()

        visual_timer = EventTimer("tensorrt_visual_total")
        visual_timer.register(model.model.visual)
        image = make_image()
        prompt = protocol["workload"]["prompt"]
        max_new_tokens = int(protocol["workload"]["max_new_tokens"])
        run_config = protocol["runs"][args.run_mode]
        warmups = int(run_config["warmup_generations"])
        measured = int(run_config["measured_generations"])

        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmups):
            warmup_trial = run_trial(
                model,
                processor,
                image,
                prompt,
                args.device,
                max_new_tokens,
                profiled=False,
            )
            visual_samples = visual_timer.drain_ms()
            if len(visual_samples) != 1:
                raise RuntimeError(
                    f"expected one warmup visual event sample, got {len(visual_samples)}"
                )
            warmup_trial["visual_total_ms"] = visual_samples[0]
            result["trials"]["warmup"].append(warmup_trial)
        for _ in range(measured):
            if args.run_mode == "profile":
                nvtx.range_push("B2_profile_generation")
            trial = run_trial(
                model,
                processor,
                image,
                prompt,
                args.device,
                max_new_tokens,
                profiled=args.run_mode == "profile",
            )
            if args.run_mode == "profile":
                torch.cuda.synchronize()
                nvtx.range_pop()
            visual_samples = visual_timer.drain_ms()
            if len(visual_samples) != 1:
                raise RuntimeError(
                    f"expected one visual event sample, got {len(visual_samples)}"
                )
            trial["visual_total_ms"] = visual_samples[0]
            result["trials"]["measured"].append(trial)
        visual_timer.remove()

        result["summary"] = summarize_trials(result["trials"]["measured"])
        result["stage_attribution"] = attribution(result["summary"])
        result["memory"]["after_trials"] = cuda_memory_snapshot()
        result["engine_file"]["sha256_after"] = sha256_file(engine_path)
        if result["engine_file"]["sha256_after"] != EXPECTED_ENGINE_SHA256:
            raise RuntimeError("TensorRT engine hash changed during run")
        result["success"] = True
    except Exception as exc:
        result["success"] = False
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    result["finished_utc"] = utc_now()
    write_result(result_path, result)
    if result["success"]:
        print(
            json.dumps(
                {
                    "success": True,
                    "phase": result["phase"],
                    "run_mode": result["run_mode"],
                    "summary": result["summary"],
                    "stage_attribution": result["stage_attribution"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(result["failure"], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
