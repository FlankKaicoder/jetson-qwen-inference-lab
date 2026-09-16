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
        "stddev_ms": statistics.stdev(values_ms),
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
    def __init__(self, engine_path: Path, spatial_merge_size: int):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
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
        if not self.context.execute_async_v3(
            torch.cuda.current_stream().cuda_stream
        ):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
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


def run_trial(
    model,
    processor,
    image,
    prompt: str,
    device: str,
    max_new_tokens: int,
    visual_timer: EventTimer,
    projector_timers: list[EventTimer],
    backend: str,
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
        prefill_start.record()
        prefill_output = model(
            **inputs,
            use_cache=True,
            logits_to_keep=1,
        )
        prefill_end.record()
        torch.cuda.synchronize()
        prefill_ms = prefill_start.elapsed_time(prefill_end)

        visual_samples = visual_timer.drain_ms()
        if len(visual_samples) != 1:
            raise RuntimeError(
                f"expected one visual event sample, got {len(visual_samples)}"
            )
        visual_total_ms = visual_samples[0]
        if backend == "pytorch_fp16":
            projector_samples = [
                value
                for timer in projector_timers
                for value in timer.drain_ms()
            ]
            if len(projector_samples) != 4:
                raise RuntimeError(
                    f"expected four projector event samples, got {len(projector_samples)}"
                )
            projector_ms = sum(projector_samples)
            vision_encoder_ms = max(0.0, visual_total_ms - projector_ms)
            vision_projector_combined_ms = None
        elif backend == "tensorrt_fp16":
            for timer in projector_timers:
                timer.drain_ms()
            projector_ms = None
            vision_encoder_ms = None
            vision_projector_combined_ms = visual_total_ms
        else:
            raise RuntimeError(f"unsupported backend {backend}")

        prefill_logits = prefill_output.logits[:, -1, :]
        next_token = prefill_logits.argmax(dim=-1, keepdim=True)
        token_ids.append(int(next_token.item()))
        cache = prefill_output.past_key_values
        decode_output = prefill_output

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
            decode_output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                logits_to_keep=1,
            )
        torch.cuda.synchronize()
        decode_ms = (time.monotonic() - decode_start) * 1000.0

    generated_seconds = prefill_ms / 1000.0 + decode_ms / 1000.0
    generated_text = processor.tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )
    return {
        "backend": backend,
        "generation_success": len(token_ids) == max_new_tokens,
        "generated_tokens": len(token_ids),
        "generated_token_ids": token_ids,
        "generated_text": generated_text,
        "preprocess_ms": preprocess_ms,
        "visual_total_ms": visual_total_ms,
        "vision_encoder_ms": vision_encoder_ms,
        "projector_ms": projector_ms,
        "vision_projector_combined_ms": vision_projector_combined_ms,
        "prefill_ms": prefill_ms,
        "derived_language_decoder_prefill_ms": prefill_ms - visual_total_ms,
        "decode_ms": decode_ms,
        "decode_per_token_ms": decode_ms / (max_new_tokens - 1),
        "generation_total_ms": prefill_ms + decode_ms,
        "generation_tokens_per_second": max_new_tokens / generated_seconds,
        "input_ids": tensor_facts(inputs["input_ids"]),
        "pixel_values": tensor_facts(inputs["pixel_values"]),
        "image_grid_thw": tensor_facts(inputs["image_grid_thw"]),
        "prompt": text,
    }


def summarize_trials(trials: list[dict]) -> dict:
    stage_keys = [
        "preprocess_ms",
        "visual_total_ms",
        "vision_encoder_ms",
        "projector_ms",
        "vision_projector_combined_ms",
        "prefill_ms",
        "derived_language_decoder_prefill_ms",
        "decode_ms",
        "decode_per_token_ms",
        "generation_total_ms",
    ]
    summary = {
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
        "stddev": statistics.stdev(rates),
        "min": min(rates),
        "max": max(rates),
    }
    return summary


def backend_attribution(summary: dict) -> dict:
    total = summary["generation_total_ms"]["mean_ms"]
    prefill = summary["prefill_ms"]["mean_ms"]
    decode = summary["decode_ms"]["mean_ms"]
    visual = summary["visual_total_ms"]["mean_ms"]
    encoder = summary["vision_encoder_ms"]
    projector = summary["projector_ms"]
    encoder_mean = encoder["mean_ms"] if encoder is not None else None
    projector_mean = projector["mean_ms"] if projector is not None else None
    return {
        "prefill_share_of_generation": prefill / total,
        "decode_share_of_generation": decode / total,
        "visual_share_of_prefill": visual / prefill,
        "vision_encoder_share_of_prefill": (
            encoder_mean / prefill if encoder_mean is not None else None
        ),
        "projector_share_of_prefill": (
            projector_mean / prefill if projector_mean is not None else None
        ),
        "dominant_generation_stage": "decode" if decode > prefill else "prefill",
        "dominant_prefill_visual_boundary": (
            "vision_encoder"
            if encoder_mean is not None and encoder_mean >= (projector_mean or 0.0)
            else "vision_projector_combined"
        ),
    }


def write_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n")


def fail(path: Path, result: dict, exc: Exception) -> None:
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
    parser.add_argument("--device", default="cuda:0")
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
        "phase": "Phase 9.3-B1",
        "title": "Qwen3-VL end-to-end stage latency breakdown",
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
            "timing": protocol["timing"],
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
        "backends": {},
        "stage_attribution": {},
        "comparison_summary": {},
        "prohibited_operations_performed": {
            "optimization": False,
            "quantization": False,
            "decoder_modification": False,
            "tensorrt_engine_rebuild": False,
            "cuda_modification": False,
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
            "total_memory_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
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
            "visual_backend_class": type(model.model.visual).__name__,
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

        image = make_image()
        prompt = protocol["workload"]["prompt"]
        max_new_tokens = int(protocol["workload"]["max_new_tokens"])
        warmups = int(protocol["workload"]["warmup_generations_per_backend"])
        measured = int(protocol["workload"]["measured_generations_per_backend"])

        visual_timer = EventTimer("visual_total")
        visual_timer.register(model.model.visual)
        projector_timers = [EventTimer("visual_merger")]
        projector_timers[0].register(model.model.visual.merger)
        for child in model.model.visual.deepstack_merger_list:
            timer = EventTimer("deepstack_merger")
            timer.register(child)
            projector_timers.append(timer)

        torch.cuda.reset_peak_memory_stats()
        warmup_trials = []
        measured_trials = []
        for _ in range(warmups):
            warmup_trials.append(
                run_trial(
                    model,
                    processor,
                    image,
                    prompt,
                    args.device,
                    max_new_tokens,
                    visual_timer,
                    projector_timers,
                    "pytorch_fp16",
                )
            )
        for _ in range(measured):
            measured_trials.append(
                run_trial(
                    model,
                    processor,
                    image,
                    prompt,
                    args.device,
                    max_new_tokens,
                    visual_timer,
                    projector_timers,
                    "pytorch_fp16",
                )
            )
        pytorch_summary = summarize_trials(measured_trials)
        result["backends"]["pytorch_fp16"] = {
            "warmup_trials": warmup_trials,
            "measured_trials": measured_trials,
            "summary": pytorch_summary,
            "memory_after_backend": cuda_memory_snapshot(),
        }
        result["memory"]["after_pytorch_fp16"] = cuda_memory_snapshot()

        for timer in [visual_timer, *projector_timers]:
            timer.remove()

        original_visual = model.model.visual
        adapter = TensorRTVisionAdapter(
            engine_path,
            int(original_visual.spatial_merge_size),
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

        tensorrt_timer = EventTimer("tensorrt_visual_total")
        tensorrt_timer.register(model.model.visual)
        torch.cuda.reset_peak_memory_stats()
        warmup_trials = []
        measured_trials = []
        for _ in range(warmups):
            warmup_trials.append(
                run_trial(
                    model,
                    processor,
                    image,
                    prompt,
                    args.device,
                    max_new_tokens,
                    tensorrt_timer,
                    [],
                    "tensorrt_fp16",
                )
            )
        for _ in range(measured):
            measured_trials.append(
                run_trial(
                    model,
                    processor,
                    image,
                    prompt,
                    args.device,
                    max_new_tokens,
                    tensorrt_timer,
                    [],
                    "tensorrt_fp16",
                )
            )
        tensorrt_summary = summarize_trials(measured_trials)
        result["backends"]["tensorrt_fp16"] = {
            "warmup_trials": warmup_trials,
            "measured_trials": measured_trials,
            "summary": tensorrt_summary,
            "memory_after_backend": cuda_memory_snapshot(),
        }
        result["memory"]["after_tensorrt_fp16"] = cuda_memory_snapshot()
        tensorrt_timer.remove()

        result["stage_attribution"] = {
            "pytorch_fp16": backend_attribution(pytorch_summary),
            "tensorrt_fp16": backend_attribution(tensorrt_summary),
        }
        baseline_trial = result["backends"]["pytorch_fp16"]["measured_trials"][0]
        tensorrt_trial = result["backends"]["tensorrt_fp16"]["measured_trials"][0]
        result["comparison_summary"] = {
            "generated_token_sequences_equal": baseline_trial[
                "generated_token_ids"
            ] == tensorrt_trial["generated_token_ids"],
            "token_ids": baseline_trial["generated_token_ids"],
            "decoded_text": baseline_trial["generated_text"],
            "pytorch_fp16": {
                key: value
                for key, value in pytorch_summary.items()
                if key
                in {
                    "preprocess_ms",
                    "vision_encoder_ms",
                    "projector_ms",
                    "prefill_ms",
                    "decode_per_token_ms",
                }
            },
            "tensorrt_fp16": {
                key: value
                for key, value in tensorrt_summary.items()
                if key
                in {
                    "preprocess_ms",
                    "vision_encoder_ms",
                    "projector_ms",
                    "vision_projector_combined_ms",
                    "prefill_ms",
                    "decode_per_token_ms",
                }
            },
            "interpretation": "descriptive fixed-workload stage attribution only; no optimization claim",
        }
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
                    "backends": list(result["backends"]),
                    "pytorch_summary": result["backends"]["pytorch_fp16"][
                        "summary"
                    ],
                    "tensorrt_summary": result["backends"]["tensorrt_fp16"][
                        "summary"
                    ],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(result["failure"], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
