#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import tensorrt as trt
from safetensors import safe_open
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLVisionModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enum_name(value):
    return getattr(value, "name", str(value))


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


def tensor_summary(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def output_metrics(reference, actual):
    reference_f32 = reference.detach().float()
    actual_f32 = actual.detach().float()
    difference = actual_f32 - reference_f32
    cosine = torch.nn.functional.cosine_similarity(
        reference_f32.flatten(), actual_f32.flatten(), dim=0
    )
    return {
        "shape_reference": list(reference.shape),
        "shape_tensorrt": list(actual.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_tensorrt": str(actual.dtype),
        "max_absolute_error": difference.abs().max().item(),
        "mean_absolute_error": difference.abs().mean().item(),
        "cosine_similarity": cosine.item(),
        "reference_finite": bool(torch.isfinite(reference_f32).all().item()),
        "tensorrt_finite": bool(torch.isfinite(actual_f32).all().item()),
        "metrics_finite": bool(
            torch.isfinite(torch.tensor([difference.abs().max().item(), difference.abs().mean().item(), cosine.item()])).all().item()
        ),
    }


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model-path", required=True)
    argument_parser.add_argument("--engine-path", required=True)
    argument_parser.add_argument("--result-path", required=True)
    argument_parser.add_argument("--device", default="cuda:0")
    args = argument_parser.parse_args()

    model_path = Path(args.model_path).resolve()
    engine_path = Path(args.engine_path).resolve()
    result_path = Path(args.result_path).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    started_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(20260916)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()

    result = {
        "phase": "Phase 9.2-C2",
        "audit_mode": "PYTORCH_VS_TENSORRT_FP16_CORRECTNESS_ONLY",
        "started_utc": started_utc,
        "hostname": platform.node(),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tensorrt_version": trt.__version__,
        "model_identity": {
            "local_path": str(model_path),
            "weights_path": str(model_path / "model.safetensors"),
            "weights_size_bytes": (model_path / "model.safetensors").stat().st_size,
            "weights_sha256": sha256_file(model_path / "model.safetensors"),
            "config_sha256": sha256_file(model_path / "config.json"),
        },
        "engine_file": {
            "path": str(engine_path),
            "size_bytes": engine_path.stat().st_size,
            "sha256": sha256_file(engine_path),
        },
        "memory": {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
        },
        "boundary": {
            "grid_thw": [1, 28, 28],
            "pixel_values_shape": [784, 1536],
            "pixel_values_dtype": "torch.float16",
            "pixel_values_values": "deterministic linspace [-1,1]",
            "output_names": ["final_hidden", "deepstack_0", "deepstack_1", "deepstack_2"],
        },
        "prohibited_operations_performed": {
            "benchmark_run": False,
            "latency_measured": False,
            "optimization_performed": False,
            "quantization_performed": False,
            "cuda_kernel_modified": False,
            "environment_modified": False,
        },
    }

    try:
        config = Qwen3VLConfig.from_pretrained(str(model_path))
        visual_config = config.vision_config
        visual_config._attn_implementation = "eager"
        visual = Qwen3VLVisionModel(config=visual_config)
        state = {}
        with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual.") :]] = checkpoint.get_tensor(key)
        load_result = visual.load_state_dict(state, strict=True)
        del state
        gc.collect()
        visual.to(device=device, dtype=torch.float16)
        visual.eval()

        grid_thw = torch.tensor([[1, 28, 28]], device=device, dtype=torch.long)
        pixel_values = torch.linspace(
            -1.0, 1.0, 784 * 1536, device=device, dtype=torch.float16
        ).reshape(784, 1536)
        with torch.no_grad():
            final_hidden, deepstack_outputs = visual(pixel_values, grid_thw)
            torch.cuda.synchronize()
        reference_outputs = {
            "final_hidden": final_hidden.detach().float().cpu(),
            "deepstack_0": deepstack_outputs[0].detach().float().cpu(),
            "deepstack_1": deepstack_outputs[1].detach().float().cpu(),
            "deepstack_2": deepstack_outputs[2].detach().float().cpu(),
        }
        result["pytorch_reference"] = {
            "class": visual.__class__.__name__,
            "attention_implementation": visual_config._attn_implementation,
            "parameter_count": sum(parameter.numel() for parameter in visual.parameters()),
            "state_dict_loaded": True,
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "blocks": len(visual.blocks),
            "deepstack_indexes": list(visual.deepstack_visual_indexes),
            "output_shapes": {
                name: tensor_summary(reference_outputs[name]) for name in reference_outputs
            },
            "all_finite": all(
                bool(torch.isfinite(reference_outputs[name]).all().item())
                for name in reference_outputs
            ),
        }
        del visual
        gc.collect()
        torch.cuda.empty_cache()

        runtime_logger = RuntimeLogger()
        runtime = trt.Runtime(runtime_logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("TensorRT engine deserialization returned None")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT execution context creation returned None")

        input_name = "pixel_values"
        output_names = ["final_hidden", "deepstack_0", "deepstack_1", "deepstack_2"]
        context.set_input_shape(input_name, tuple(pixel_values.shape))
        context.set_tensor_address(input_name, int(pixel_values.data_ptr()))
        output_tensors = {}
        for name in output_names:
            shape = tuple(context.get_tensor_shape(name))
            output_tensors[name] = torch.empty(shape, dtype=torch.float16, device=device)
            context.set_tensor_address(name, int(output_tensors[name].data_ptr()))

        execute_returned = context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        tensorrt_outputs = {name: tensor.detach().float().cpu() for name, tensor in output_tensors.items()}

        result["tensorrt_runtime"] = {
            "engine_name": engine.name,
            "engine_num_layers": engine.num_layers,
            "engine_num_io_tensors": engine.num_io_tensors,
            "num_optimization_profiles": engine.num_optimization_profiles,
            "input_shape": list(pixel_values.shape),
            "input_dtype": str(pixel_values.dtype),
            "output_shapes": {
                name: tensor_summary(tensorrt_outputs[name]) for name in output_names
            },
            "execute_async_v3_returned": bool(execute_returned),
            "all_binding_shapes_specified": bool(context.all_binding_shapes_specified),
            "all_shape_inputs_specified": bool(context.all_shape_inputs_specified),
            "device_memory_size_v2": engine.device_memory_size_v2,
            "execution_context_created": True,
            "engine_executed_once": True,
        }
        result["comparison"] = {
            name: output_metrics(reference_outputs[name], tensorrt_outputs[name])
            for name in output_names
        }
        result["comparison_summary"] = {
            "all_output_finite": all(
                result["comparison"][name]["tensorrt_finite"] for name in output_names
            ),
            "all_reference_finite": all(
                result["comparison"][name]["reference_finite"] for name in output_names
            ),
            "all_metrics_finite": all(
                result["comparison"][name]["metrics_finite"] for name in output_names
            ),
            "all_shapes_match": all(
                result["comparison"][name]["shape_reference"]
                == result["comparison"][name]["shape_tensorrt"]
                for name in output_names
            ),
            "max_absolute_error": max(
                result["comparison"][name]["max_absolute_error"] for name in output_names
            ),
            "max_mean_absolute_error": max(
                result["comparison"][name]["mean_absolute_error"] for name in output_names
            ),
            "min_cosine_similarity": min(
                result["comparison"][name]["cosine_similarity"] for name in output_names
            ),
        }
        result["memory"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["memory"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["memory"]["allocated_after_bytes"] = torch.cuda.memory_allocated()
        result["memory"]["reserved_after_bytes"] = torch.cuda.memory_reserved()
        result["runtime_log"] = {
            "record_count": len(runtime_logger.records),
            "severity_counts": dict(
                sorted(Counter(record["severity"] for record in runtime_logger.records).items())
            ),
            "warning_count": sum(record["severity"] == "WARNING" for record in runtime_logger.records),
            "error_count": sum(
                record["severity"] in ("ERROR", "INTERNAL_ERROR")
                for record in runtime_logger.records
            ),
            "nonverbose_records": [
                record for record in runtime_logger.records if record["severity"] != "VERBOSE"
            ],
        }
    except Exception as exc:
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))

    success = (
        "failure" not in result
        and result.get("comparison_summary", {}).get("all_output_finite", False)
        and result.get("comparison_summary", {}).get("all_reference_finite", False)
        and result.get("comparison_summary", {}).get("all_metrics_finite", False)
        and result.get("comparison_summary", {}).get("all_shapes_match", False)
    )
    if not success:
        sys.exit(2)


if __name__ == "__main__":
    main()
