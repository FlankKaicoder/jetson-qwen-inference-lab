#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
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


def rate_summary(values: list[float]) -> dict:
    return {
        "sample_count": len(values),
        "unit": "tokens/s",
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
        "dtype": str(values.dtype),
    }


def tensor_metrics(reference, actual) -> dict:
    reference_f32 = reference.detach().float().cpu()
    actual_f32 = actual.detach().float().cpu()
    reference_summary = finite_summary(reference_f32)
    actual_summary = finite_summary(actual_f32)
    both_finite = reference_summary["all_finite"] and actual_summary["all_finite"]
    if both_finite:
        difference = actual_f32 - reference_f32
        metrics = {
            "max_absolute_error": difference.abs().max().item(),
            "mean_absolute_error": difference.abs().mean().item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                reference_f32.flatten(),
                actual_f32.flatten(),
                dim=0,
            ).item(),
            "metrics_finite": True,
        }
    else:
        metrics = {
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "cosine_similarity": None,
            "metrics_finite": False,
        }
    return {
        "shape_reference": list(reference.shape),
        "shape_actual": list(actual.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_actual": str(actual.dtype),
        "reference": reference_summary,
        "actual": actual_summary,
        **metrics,
    }


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


class TensorRTVisionAdapter(nn.Module):
    def __init__(self, engine_path: Path, spatial_merge_size: int):
        super().__init__()
        self.engine_path = engine_path
        self.spatial_merge_size = spatial_merge_size
        self.output_names = OUTPUT_NAMES
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
            self.input_name, int(pixel_values.data_ptr())
        )
        for name, tensor in self.output_tensors.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        execute_returned = self.context.execute_async_v3(
            torch.cuda.current_stream().cuda_stream
        )
        if not execute_returned:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        self.last_execute_returned = True
        return (
            self.output_tensors["final_hidden"],
            [
                self.output_tensors["deepstack_0"],
                self.output_tensors["deepstack_1"],
                self.output_tensors["deepstack_2"],
            ],
        )

    def metadata(self) -> dict:
        log_counts = {}
        for record in self.runtime_logger.records:
            severity = record["severity"]
            log_counts[severity] = log_counts.get(severity, 0) + 1
        return {
            "engine_name": self.engine.name,
            "num_layers": self.engine.num_layers,
            "num_io_tensors": self.engine.num_io_tensors,
            "num_optimization_profiles": self.engine.num_optimization_profiles,
            "device_memory_size_v2": self.engine.device_memory_size_v2,
            "log_counts_by_severity": log_counts,
            "error_count": sum(
                1
                for record in self.runtime_logger.records
                if record["severity"] == "ERROR"
            ),
        }


def cuda_memory_snapshot() -> dict:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return -1


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
    facts = {
        "input_ids": finite_summary(inputs["input_ids"]),
        "attention_mask": finite_summary(inputs["attention_mask"]),
        "pixel_values": finite_summary(inputs["pixel_values"]),
        "image_grid_thw": finite_summary(inputs["image_grid_thw"]),
        "prompt": text,
    }
    expected_vision_tokens = (
        inputs["image_grid_thw"].prod(dim=-1)
        // model.model.visual.spatial_merge_size**2
    ).sum().item()
    image_token_count = int(
        (inputs["input_ids"] == model.config.image_token_id).sum().item()
    )
    if expected_vision_tokens != image_token_count:
        raise RuntimeError("image token count does not match visual output tokens")
    if inputs["pixel_values"].shape != (784, 1536):
        raise RuntimeError("processor output did not match fixed vision boundary")
    return inputs, facts


def compare_logits(reference, actual, baseline_token, actual_token) -> dict:
    metrics = tensor_metrics(reference, actual)
    metrics["baseline_top1_token"] = int(baseline_token.item())
    metrics["actual_top1_token"] = int(actual_token.item())
    metrics["top1_agreement"] = bool(
        baseline_token.item() == actual_token.item()
    )
    return metrics


def run_generation_trial(
    model,
    inputs,
    max_new_tokens: int,
    eos_token_id: int,
):
    attention_mask = inputs["attention_mask"].clone()
    base_length = int(inputs["input_ids"].shape[1])
    token_ids = []

    torch.cuda.synchronize()
    total_start = torch.cuda.Event(enable_timing=True)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        total_start.record()
        prefill_start.record()
        prefill_output = model(
            **inputs,
            use_cache=True,
            logits_to_keep=1,
        )
        prefill_end.record()
        torch.cuda.synchronize()
        first_token_latency_ms = prefill_start.elapsed_time(prefill_end)

        prefill_logits = prefill_output.logits[:, -1, :]
        next_token = prefill_logits.argmax(dim=-1, keepdim=True)
        token_ids.append(int(next_token.item()))
        cache = prefill_output.past_key_values
        decode_output = prefill_output
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
        total_end.record()
        torch.cuda.synchronize()
        total_latency_ms = total_start.elapsed_time(total_end)

        prefill_logits_finite = finite_summary(prefill_logits)["all_finite"]
        final_decode_logits_finite = finite_summary(
            decode_output.logits[:, -1, :]
        )["all_finite"]
        first_token_logits = prefill_logits.detach().float().cpu()
        top1_token = prefill_logits.argmax(dim=-1, keepdim=True)

    return {
        "generation_success": len(token_ids) == max_new_tokens,
        "generated_tokens": len(token_ids),
        "generated_token_ids": token_ids,
        "first_token_latency_ms": first_token_latency_ms,
        "total_latency_ms": total_latency_ms,
        "tokens_per_second": max_new_tokens / (total_latency_ms / 1000.0),
        "prefill_logits_finite": prefill_logits_finite,
        "final_decode_logits_finite": final_decode_logits_finite,
        "first_token_logits": first_token_logits,
        "first_token_top1": top1_token,
    }


def strip_logits_for_storage(trial: dict) -> dict:
    return {
        key: value
        for key, value in trial.items()
        if key not in {"first_token_logits", "first_token_top1"}
    }


def summarize_trials(trials: list[dict]) -> dict:
    latencies = [trial["first_token_latency_ms"] for trial in trials]
    totals = [trial["total_latency_ms"] for trial in trials]
    rates = [trial["tokens_per_second"] for trial in trials]
    return {
        "generation_success_count": sum(
            1 for trial in trials if trial["generation_success"]
        ),
        "sample_count": len(trials),
        "first_token_latency_ms": latency_summary(latencies),
        "total_latency_ms": latency_summary(totals),
        "tokens_per_second": rate_summary(rates),
        "unique_token_id_sequences": [
            trial["generated_token_ids"] for trial in trials
        ],
        "all_token_sequences_equal": all(
            trial["generated_token_ids"] == trials[0]["generated_token_ids"]
            for trial in trials
        ),
    }


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
        raise RuntimeError("harness protocol is not frozen")

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.cuda.set_device(device)
    torch.cuda.synchronize()

    result = {
        "phase": "Phase 9.3-A",
        "title": "Qwen3-VL end-to-end TensorRT vision integration harness",
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
            "timing": protocol["timing"],
            "workload": protocol["workload"],
            "correctness": protocol["correctness"],
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
        "input": {},
        "integration_check": {},
        "decoder_unchanged_evidence": {},
        "memory": {},
        "backends": {},
        "comparison_summary": {},
        "prohibited_operations_performed": {
            "optimization": False,
            "quantization": False,
            "tensorrt_engine_rebuild": False,
            "decoder_modification": False,
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
            "original_visual_class": type(model.model.visual).__name__,
            "model_dtype": str(model.dtype),
            "attention_implementation": protocol["model"][
                "attention_implementation"
            ],
            "total_parameter_count_before_injection": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "language_model_parameter_count": sum(
                parameter.numel()
                for parameter in model.model.language_model.parameters()
            ),
            "tokenizer_class": type(processor.tokenizer).__name__,
        }
        result["memory"]["after_model_load"] = {
            **cuda_memory_snapshot(),
            "host_mem_available_bytes": mem_available_bytes(),
        }

        image = make_image()
        prompt = protocol["workload"]["prompt"]
        inputs, input_facts = preprocess(
            model,
            processor,
            image,
            prompt,
            "cuda:0",
        )
        result["input"] = input_facts
        if not result["input"]["pixel_values"]["all_finite"]:
            raise RuntimeError("processor pixel_values are non-finite")

        grid_thw = inputs["image_grid_thw"]
        original_visual = model.model.visual
        with torch.no_grad():
            reference_final_hidden, reference_deepstack = original_visual(
                inputs["pixel_values"],
                grid_thw=grid_thw,
            )
            torch.cuda.synchronize()
        result["memory"]["after_pytorch_vision_reference"] = (
            cuda_memory_snapshot()
        )

        adapter = TensorRTVisionAdapter(
            engine_path,
            int(original_visual.spatial_merge_size),
        )
        actual_final_hidden, actual_deepstack = adapter(
            inputs["pixel_values"],
            grid_thw=grid_thw,
        )
        torch.cuda.synchronize()
        integration_output_metrics = {
            "final_hidden": tensor_metrics(
                reference_final_hidden, actual_final_hidden
            ),
        }
        for name, reference, actual in zip(
            OUTPUT_NAMES[1:],
            reference_deepstack,
            actual_deepstack,
            strict=True,
        ):
            integration_output_metrics[name] = tensor_metrics(
                reference, actual
            )
        all_shapes_match = all(
            metrics["shape_reference"] == metrics["shape_actual"]
            for metrics in integration_output_metrics.values()
        )
        all_outputs_finite = all(
            metrics["reference"]["all_finite"]
            and metrics["actual"]["all_finite"]
            for metrics in integration_output_metrics.values()
        )
        result["integration_check"] = {
            "input_generation": "processor_output_fp16_fixed_boundary",
            "all_shapes_match": all_shapes_match,
            "all_outputs_finite": all_outputs_finite,
            "adapter_metadata": adapter.metadata(),
            "output_metrics": integration_output_metrics,
            "numerical_tolerance_applied": False,
        }
        if not all_shapes_match or not all_outputs_finite:
            raise RuntimeError("TensorRT vision integration outputs failed shape/finite checks")

        decoder_parameter_count_before = result["model_evidence"][
            "language_model_parameter_count"
        ]
        model.model.visual = adapter
        original_visual = None
        reference_final_hidden = None
        reference_deepstack = None
        actual_final_hidden = None
        actual_deepstack = None
        gc.collect()
        torch.cuda.empty_cache()
        result["decoder_unchanged_evidence"] = {
            "decoder_path": "model.model.language_model and model.lm_head unchanged",
            "language_model_class": type(model.model.language_model).__name__,
            "language_model_parameter_count_after_injection": sum(
                parameter.numel()
                for parameter in model.model.language_model.parameters()
            ),
            "language_model_parameter_count_unchanged": (
                sum(
                    parameter.numel()
                    for parameter in model.model.language_model.parameters()
                )
                == decoder_parameter_count_before
            ),
            "visual_backend_class_after_injection": type(
                model.model.visual
            ).__name__,
            "original_visual_reference_released": True,
        }
        result["memory"]["after_tensorrt_injection_and_cleanup"] = {
            **cuda_memory_snapshot(),
            "host_mem_available_bytes": mem_available_bytes(),
        }

        eos_token_id = processor.tokenizer.eos_token_id
        warmups = int(protocol["workload"]["warmup_generations_per_backend"])
        measured = int(protocol["workload"]["measured_generations_per_backend"])
        max_new_tokens = int(protocol["workload"]["max_new_tokens"])
        raw_trials_by_backend = {}

        for backend_name in protocol["workload"]["backend_order"]:
            warmup_trials = []
            measured_trials = []
            for _ in range(warmups):
                warmup = run_generation_trial(
                    model,
                    inputs,
                    max_new_tokens,
                    eos_token_id,
                )
                warmup_trials.append(strip_logits_for_storage(warmup))
            for _ in range(measured):
                trial = run_generation_trial(
                    model,
                    inputs,
                    max_new_tokens,
                    eos_token_id,
                )
                measured_trials.append(trial)
            summary = summarize_trials(measured_trials)
            reference_logits = measured_trials[0]["first_token_logits"]
            reference_token = measured_trials[0]["first_token_top1"]
            logit_comparisons = []
            for trial in measured_trials:
                logit_comparisons.append(
                    compare_logits(
                        reference_logits,
                        trial["first_token_logits"],
                        reference_token,
                        trial["first_token_top1"],
                    )
                )
            decoded_texts = [
                processor.tokenizer.decode(
                    trial["generated_token_ids"],
                    skip_special_tokens=True,
                )
                for trial in measured_trials
            ]
            result["backends"][backend_name] = {
                "warmup_trials": warmup_trials,
                "measured_trials": [
                    strip_logits_for_storage(trial)
                    for trial in measured_trials
                ],
                "summary": summary,
                "first_token_logits_self_comparison": logit_comparisons,
                "decoded_texts": decoded_texts,
                "unique_decoded_texts": sorted(set(decoded_texts)),
                "memory_after_backend": cuda_memory_snapshot(),
            }
            raw_trials_by_backend[backend_name] = measured_trials
            result["memory"][f"after_{backend_name}"] = cuda_memory_snapshot()

        baseline = result["backends"]["pytorch_fp16"]
        tensorrt = result["backends"]["tensorrt_fp16"]
        baseline_raw_trials = raw_trials_by_backend["pytorch_fp16"]
        tensorrt_raw_trials = raw_trials_by_backend["tensorrt_fp16"]
        baseline_reference_logits = baseline_raw_trials[0]["first_token_logits"]
        baseline_reference_token = baseline_raw_trials[0]["first_token_top1"]
        cross_backend_logits = [
            compare_logits(
                baseline_reference_logits,
                trial["first_token_logits"],
                baseline_reference_token,
                trial["first_token_top1"],
            )
            for trial in tensorrt_raw_trials
        ]
        result["integration_check"]["cross_backend_first_token_logits"] = {
            "reference": "pytorch_fp16_measured_trial_1",
            "comparison_trials": cross_backend_logits,
            "all_top1_agree": all(
                comparison["top1_agreement"]
                for comparison in cross_backend_logits
            ),
            "numerical_tolerance_applied": False,
        }
        baseline_trial = baseline["measured_trials"][0]
        tensorrt_trial = tensorrt["measured_trials"][0]
        baseline_summary = baseline["summary"]
        tensorrt_summary = tensorrt["summary"]
        result["comparison_summary"] = {
            "generated_token_sequences_equal": (
                baseline_trial["generated_token_ids"]
                == tensorrt_trial["generated_token_ids"]
            ),
            "baseline_token_ids": baseline_trial["generated_token_ids"],
            "tensorrt_token_ids": tensorrt_trial["generated_token_ids"],
            "baseline_decoded_text": baseline["decoded_texts"][0],
            "tensorrt_decoded_text": tensorrt["decoded_texts"][0],
            "baseline_mean_first_token_latency_ms": baseline_summary[
                "first_token_latency_ms"
            ]["mean_ms"],
            "tensorrt_mean_first_token_latency_ms": tensorrt_summary[
                "first_token_latency_ms"
            ]["mean_ms"],
            "first_token_latency_delta_ms_tensorrt_minus_pytorch": (
                tensorrt_summary["first_token_latency_ms"]["mean_ms"]
                - baseline_summary["first_token_latency_ms"]["mean_ms"]
            ),
            "baseline_mean_total_latency_ms": baseline_summary[
                "total_latency_ms"
            ]["mean_ms"],
            "tensorrt_mean_total_latency_ms": tensorrt_summary[
                "total_latency_ms"
            ]["mean_ms"],
            "total_latency_delta_ms_tensorrt_minus_pytorch": (
                tensorrt_summary["total_latency_ms"]["mean_ms"]
                - baseline_summary["total_latency_ms"]["mean_ms"]
            ),
            "total_latency_ratio_tensorrt_over_pytorch": (
                tensorrt_summary["total_latency_ms"]["mean_ms"]
                / baseline_summary["total_latency_ms"]["mean_ms"]
            ),
            "baseline_mean_tokens_per_second": baseline_summary[
                "tokens_per_second"
            ]["mean"],
            "tensorrt_mean_tokens_per_second": tensorrt_summary[
                "tokens_per_second"
            ]["mean"],
            "interpretation": "descriptive fixed-workload comparison only; no optimization claim",
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
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
