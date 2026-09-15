#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import platform
import re
import resource
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_MODEL_SHA256 = (
    "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return -1


def summary_ms(values_ms: list[float]) -> dict:
    return {
        "sample_count": len(values_ms),
        "mean_ms": statistics.mean(values_ms),
        "median_ms": statistics.median(values_ms),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "stddev_ms": statistics.stdev(values_ms),
    }


def summarize_seconds(values_s: list[float]) -> dict:
    return {
        "sample_count": len(values_s),
        "mean_seconds": statistics.mean(values_s),
        "median_seconds": statistics.median(values_s),
        "min_seconds": min(values_s),
        "max_seconds": max(values_s),
        "stddev_seconds": statistics.stdev(values_s),
    }


def summarize_rate(values: list[float]) -> dict:
    return {
        "sample_count": len(values),
        "mean_tokens_per_second": statistics.mean(values),
        "median_tokens_per_second": statistics.median(values),
        "min_tokens_per_second": min(values),
        "max_tokens_per_second": max(values),
        "stddev_tokens_per_second": statistics.stdev(values),
    }


def tensor_facts(value) -> dict:
    import torch

    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "is_finite": bool(torch.isfinite(value).all()),
    }


class EventTimer:
    def __init__(self, name: str):
        self.name = name
        self.samples: list[tuple] = []
        self._active_start = None
        self.handles: list = []

    def register(self, module) -> None:
        self.handles.append(
            module.register_forward_pre_hook(self._pre_hook)
        )
        self.handles.append(module.register_forward_hook(self._post_hook))

    def _pre_hook(self, module, args):
        import torch

        self._active_start = torch.cuda.Event(enable_timing=True)
        self._active_start.record()

    def _post_hook(self, module, args, output):
        import torch

        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.samples.append((self._active_start, end))
        self._active_start = None

    def drain_ms(self) -> list[float]:
        values = []
        for start, end in self.samples:
            values.append(start.elapsed_time(end))
        self.samples.clear()
        return values

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def start_tegrastats(log_path: Path, interval_ms: int):
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
        return {"sample_count": 0, "status": "TEGRASTATS_LOG_MISSING"}
    samples: dict[str, list[int]] = {
        "VDD_IN": [],
        "VDD_CPU_GPU_CV": [],
        "VDD_SOC": [],
    }
    for line in log_path.read_text(errors="replace").splitlines():
        for key in samples:
            match = re.search(rf"{key}\s+(\d+)mW/", line)
            if match:
                samples[key].append(int(match.group(1)))

    output: dict[str, object] = {
        "status": "PASS",
        "sample_count": len(samples["VDD_IN"]),
    }
    for key, values in samples.items():
        if not values:
            output[key] = {"sample_count": 0, "status": "NO_SAMPLES"}
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


def fail(out_path: Path, result: dict, exc: Exception) -> None:
    result["success"] = False
    result["failure"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    result["completed_at_utc"] = utc_now()
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(1)


def make_image():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((112, 112, 335, 335), fill=(192, 0, 0))
    return image


def preprocess(processor, image, prompt: str, device: str):
    import torch

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
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise RuntimeError("processor did not produce the image input contract")
    inputs = inputs.to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    return text, inputs, elapsed


def run_trial(
    model,
    processor,
    image,
    prompt: str,
    device: str,
    protocol: dict,
    visual_timer: EventTimer,
    projector_timers: list[EventTimer],
):
    import torch

    max_new_tokens = int(protocol["workload"]["max_new_tokens"])
    text, inputs, preprocess_seconds = preprocess(
        processor, image, prompt, device
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

        visual_total_ms = sum(visual_timer.drain_ms())
        projector_ms = sum(
            sum(timer.drain_ms()) for timer in projector_timers
        )
        vision_encoder_ms = max(0.0, visual_total_ms - projector_ms)

        prefill_logits = prefill_output.logits[:, -1, :]
        next_token = prefill_logits.argmax(dim=-1, keepdim=True)
        token_ids.append(int(next_token.item()))

        cache = prefill_output.past_key_values
        decode_start = time.monotonic()
        for step in range(1, max_new_tokens):
            logits = prefill_output.logits
            if step == 1:
                logits = prefill_output.logits[:, -1, :]
            else:
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
        decode_seconds = time.monotonic() - decode_start

    generated_text = processor.tokenizer.decode(
        token_ids, skip_special_tokens=True
    )
    generated_seconds = (
        prefill_ms / 1000.0 + decode_seconds
    )
    tokens_per_second = max_new_tokens / generated_seconds

    return {
        "preprocess_seconds": preprocess_seconds,
        "visual_total_ms": visual_total_ms,
        "projector_ms": projector_ms,
        "vision_encoder_ms": vision_encoder_ms,
        "prefill_ms": prefill_ms,
        "decode_seconds": decode_seconds,
        "decode_per_token_seconds": decode_seconds / (max_new_tokens - 1),
        "generated_tokens": max_new_tokens,
        "end_to_end_tokens_per_second": tokens_per_second,
        "generated_token_ids": token_ids,
        "generated_text": generated_text,
        "prefill_logits_finite": bool(
            __import__("torch").isfinite(prefill_logits).all()
        ),
        "input_ids": tensor_facts(inputs["input_ids"]),
        "pixel_values": tensor_facts(inputs["pixel_values"]),
        "image_grid_thw": tensor_facts(inputs["image_grid_thw"]),
        "prompt": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    protocol_path = Path(args.protocol).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text())
    out_path = out_dir / "benchmark_result.json"
    power_log_path = out_dir / "tegrastats.log"
    tegrastats_proc = None
    result = {
        "phase": "Phase 9.1-B",
        "title": "Qwen3-VL PyTorch FP16 baseline benchmark",
        "started_at_utc": utc_now(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "authorization_boundary": {
            "tensorrt": False,
            "onnx": False,
            "quantization": False,
            "optimization": False,
            "backend_modification": False,
        },
        "runtime_dependency_status": {},
        "checkpoint_identity": {},
        "model_evidence": {},
        "memory_evidence": {},
        "warmup_trials": [],
        "measured_trials": [],
        "power_evidence": {},
    }

    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        result["runtime_dependency_status"] = {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "transformers": transformers.__version__,
            "pillow": Image.__version__,
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        result["checkpoint_identity"] = {
            "model_dir": str(model_dir),
            "model_safetensors_sha256": sha256(
                model_dir / "model.safetensors"
            ),
        }
        if result["checkpoint_identity"]["model_safetensors_sha256"] != (
            EXPECTED_MODEL_SHA256
        ):
            raise RuntimeError("pinned model.safetensors SHA-256 mismatch")

        torch.manual_seed(20260915)
        processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=protocol["model"][
                "attention_implementation"
            ],
        )
        model.eval()
        result["model_evidence"] = {
            "load_success": True,
            "model_class": type(model).__name__,
            "model_dtype": str(model.dtype),
            "attention_implementation": protocol["model"][
                "attention_implementation"
            ],
            "parameter_count": sum(
                p.numel() for p in model.parameters()
            ),
            "device": str(next(model.parameters()).device),
        }
        result["memory_evidence"]["after_model_load"] = {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "host_mem_available_bytes": mem_available_bytes(),
            "process_max_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            * 1024,
        }

        visual_timer = EventTimer("visual_total")
        visual_timer.register(model.visual)
        projector_timers = [EventTimer("visual_merger")]
        projector_timers[0].register(model.visual.merger)
        for child in model.visual.deepstack_merger_list:
            timer = EventTimer("deepstack_merger")
            timer.register(child)
            projector_timers.append(timer)

        image = make_image()
        prompt = protocol["workload"]["prompt"]
        warmup_trials = int(protocol["workload"]["warmup_trials"])
        measured_trials = int(protocol["workload"]["measured_trials"])

        tegrastats_proc = start_tegrastats(
            power_log_path,
            int(protocol["power"]["interval_ms"]),
        )
        time.sleep(0.2)

        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup_trials):
            warmup = run_trial(
                model,
                processor,
                image,
                prompt,
                "cuda:0",
                protocol,
                visual_timer,
                projector_timers,
            )
            result["warmup_trials"].append(warmup)

        for index in range(measured_trials):
            trial = run_trial(
                model,
                processor,
                image,
                prompt,
                "cuda:0",
                protocol,
                visual_timer,
                projector_timers,
            )
            trial["trial"] = index + 1
            result["measured_trials"].append(trial)
            result["memory_evidence"][f"after_trial_{index + 1}"] = {
                "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "host_mem_available_bytes": mem_available_bytes(),
                "process_max_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            }

        result["memory_evidence"]["final"] = {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_mem_available_bytes": mem_available_bytes(),
            "process_max_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            * 1024,
        }

        tegrastats_proc.terminate()
        tegrastats_proc.wait(timeout=5)
        tegrastats_proc = None
        result["power_evidence"] = parse_tegrastats(power_log_path)

        aggregates = {}
        for key in [
            "preprocess_seconds",
            "visual_total_ms",
            "projector_ms",
            "vision_encoder_ms",
            "prefill_ms",
            "decode_seconds",
            "decode_per_token_seconds",
            "end_to_end_tokens_per_second",
        ]:
            values = [trial[key] for trial in result["measured_trials"]]
            if key.endswith("_seconds"):
                aggregates[key] = summarize_seconds(values)
            elif key == "end_to_end_tokens_per_second":
                aggregates[key] = summarize_rate(values)
            else:
                aggregates[key] = summary_ms(values)
        result["aggregate_metrics"] = aggregates

        result["success"] = True
        result["completed_at_utc"] = utc_now()
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "success": True,
                    "measured_trials": len(result["measured_trials"]),
                    "mean_prefill_ms": aggregates["prefill_ms"]["mean_ms"],
                    "mean_decode_seconds": aggregates["decode_seconds"][
                        "mean_seconds"
                    ],
                    "mean_tokens_per_second": aggregates[
                        "end_to_end_tokens_per_second"
                    ]["mean_tokens_per_second"],
                    "power_sample_count": result["power_evidence"][
                        "sample_count"
                    ],
                },
                indent=2,
            )
        )
    except Exception as exc:
        if tegrastats_proc is not None:
            tegrastats_proc.terminate()
            tegrastats_proc.wait(timeout=5)
            result["power_evidence"] = parse_tegrastats(power_log_path)
        fail(out_path, result, exc)
    finally:
        for timer in [visual_timer, *projector_timers]:
            timer.remove()


if __name__ == "__main__":
    main()
