#!/usr/bin/env python3
import argparse
import hashlib
import json
import platform
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_MODEL_SHA256 = (
    "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return -1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_facts(value) -> dict:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "is_finite": bool(__import__("torch").isfinite(value).all()),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_result.json"
    result = {
        "phase": "Phase 9.1-A",
        "title": "Qwen3-VL PyTorch FP16 inference smoke test",
        "started_at_utc": utc_now(),
        "authorization_boundary": {
            "tensorrt": False,
            "onnx": False,
            "quantization": False,
            "benchmark_sweep": False,
            "optimization": False,
        },
        "expected_model_sha256": EXPECTED_MODEL_SHA256,
        "runtime_dependency_status": {},
        "checkpoint_identity": {},
        "processor_evidence": {},
        "image_pipeline_evidence": {},
        "model_evidence": {},
        "generation_evidence": {},
        "memory_evidence": {},
    }

    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        result["runtime_dependency_status"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "transformers": transformers.__version__,
            "pillow": Image.__version__,
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        result["memory_evidence"]["before_run"] = {
            "host_mem_available_bytes": mem_available_bytes(),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        }

        result["checkpoint_identity"] = {
            "model_dir": str(model_dir),
            "config_sha256": sha256(model_dir / "config.json"),
            "model_safetensors_sha256": sha256(model_dir / "model.safetensors"),
            "model_safetensors_size_bytes": (model_dir / "model.safetensors").stat().st_size,
        }
        if result["checkpoint_identity"]["model_safetensors_sha256"] != EXPECTED_MODEL_SHA256:
            raise RuntimeError("pinned model.safetensors SHA-256 mismatch")

        torch.manual_seed(args.seed)
        processor_start = time.monotonic()
        processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        result["processor_evidence"] = {
            "load_success": True,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(processor.tokenizer).__name__,
            "image_processor_class": type(processor.image_processor).__name__,
            "load_seconds_wall_clock_diagnostic": time.monotonic() - processor_start,
        }

        image = Image.new("RGB", (448, 448), "white")
        for y in range(112, 336):
            for x in range(112, 336):
                image.putpixel((x, y), (192, 0, 0))
        result["image_pipeline_evidence"]["input_image"] = {
            "mode": image.mode,
            "size": list(image.size),
            "center_pixel_rgb": list(image.getpixel((224, 224))),
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "What color is the square in this image? Answer with one word.",
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise RuntimeError("processor did not produce the image input contract")
        inputs = inputs.to(model_device := "cuda:0")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        result["image_pipeline_evidence"]["processor_outputs"] = {
            key: tensor_facts(value) for key, value in inputs.items()
        }
        result["image_pipeline_evidence"]["chat_template_success"] = True
        result["image_pipeline_evidence"]["prompt"] = text

        load_start = time.monotonic()
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=model_device,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=args.attn_implementation,
        )
        model.eval()
        parameter_count = sum(p.numel() for p in model.parameters())
        dtype_counts: dict[str, int] = {}
        for parameter in model.parameters():
            dtype_counts[str(parameter.dtype)] = (
                dtype_counts.get(str(parameter.dtype), 0) + parameter.numel()
            )
        result["model_evidence"] = {
            "load_success": True,
            "attn_implementation": args.attn_implementation,
            "model_class": type(model).__name__,
            "model_dtype": str(model.dtype),
            "parameter_count": parameter_count,
            "dtype_counts": dtype_counts,
            "load_seconds_wall_clock_diagnostic": time.monotonic() - load_start,
            "device": str(next(model.parameters()).device),
        }
        result["memory_evidence"]["after_model_load"] = {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "host_mem_available_bytes": mem_available_bytes(),
            "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }

        torch.cuda.reset_peak_memory_stats()
        generation_start = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        result["memory_evidence"]["after_generation"] = {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_mem_available_bytes": mem_available_bytes(),
            "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }

        sequence = generated.sequences[0]
        new_token_ids = sequence[inputs["input_ids"].shape[1] :].tolist()
        new_text = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        scores = generated.scores
        result["generation_evidence"] = {
            "success": True,
            "sampling_rule": "greedy",
            "max_new_tokens_requested": args.max_new_tokens,
            "generated_token_count": len(new_token_ids),
            "generated_token_ids": new_token_ids,
            "generated_text": new_text,
            "score_tensors": len(scores),
            "scores_all_finite": all(bool(torch.isfinite(score).all()) for score in scores),
            "generation_seconds_wall_clock_diagnostic": time.monotonic() - generation_start,
        }
        result["success"] = True
        result["completed_at_utc"] = utc_now()
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        fail(out_path, result, exc)


if __name__ == "__main__":
    main()
