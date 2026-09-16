#!/usr/bin/env python3
import argparse
import gc
import hashlib
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLVisionModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Infinity" if value > 0 else "-Infinity"
    return value


def flatten_tensors(value, path="value"):
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, (tuple, list)):
        tensors = []
        for index, item in enumerate(value):
            tensors.extend(flatten_tensors(item, f"{path}[{index}]"))
        return tensors
    return []


def finite_summary(tensor):
    values = tensor.detach().float()
    element_count = values.numel()
    finite_mask = torch.isfinite(values)
    finite_count = int(finite_mask.sum().item())
    finite_values = values[finite_mask]

    if finite_values.numel():
        finite_mean = float(finite_values.mean().item())
        finite_std = float(finite_values.std(unbiased=True).item())
        finite_max_abs = float(finite_values.abs().max().item())
    else:
        finite_mean = None
        finite_std = None
        finite_max_abs = None

    raw_min = None if element_count == 0 else json_scalar(values.min().item())
    raw_max = None if element_count == 0 else json_scalar(values.max().item())

    return {
        "element_count": element_count,
        "finite_count": finite_count,
        "nonfinite_count": element_count - finite_count,
        "finite_ratio": finite_count / element_count if element_count else 0.0,
        "nan_count": int(torch.isnan(values).sum().item()),
        "positive_inf_count": int(torch.isposinf(values).sum().item()),
        "negative_inf_count": int(torch.isneginf(values).sum().item()),
        "min_raw": raw_min,
        "max_raw": raw_max,
        "mean_finite_only": finite_mean,
        "std_finite_only_sample": finite_std,
        "max_abs_finite_only": finite_max_abs,
        "all_finite": finite_count == element_count,
    }


def output_summary(name, module_type, call_index, inputs, output, output_path, tensor):
    input_tensors = flatten_tensors(inputs, "input")
    input_finite = all(
        bool(torch.isfinite(tensor.detach().float()).all().item())
        for _, tensor in input_tensors
        if tensor.numel()
    )
    summary = finite_summary(tensor)
    summary.update(
        {
            "call_index": call_index,
            "module_name": name,
            "module_type": module_type,
            "output_path": output_path,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "input_finite": input_finite,
        }
    )
    return summary


def attach_hooks(model):
    records = []
    call_counter = itertools.count(1)
    handles = []

    for name, module in model.named_modules():
        if not name:
            continue

        def hook(captured_module, inputs, output, captured_name=name):
            call_index = next(call_counter)
            output_tensors = flatten_tensors(output, "output")
            for output_path, tensor in output_tensors:
                records.append(
                    output_summary(
                        captured_name,
                        captured_module.__class__.__name__,
                        call_index,
                        inputs,
                        output,
                        output_path,
                        tensor,
                    )
                )

        handles.append(module.register_forward_hook(hook))

    return handles, records


def first_nonfinite(records):
    nonfinite_records = [record for record in records if not record["all_finite"]]
    return min(nonfinite_records, key=lambda record: record["call_index"]) if nonfinite_records else None


def create_input(dtype, device):
    element_count = 784 * 1536
    # Direct CUDA FP16 linspace is retained only in the separate input-generation
    # audit; the model passes use a stable FP32 source cast to the target dtype.
    values = torch.linspace(
        -1.0, 1.0, element_count, device=device, dtype=torch.float32
    ).reshape(784, 1536)
    if dtype == torch.float32:
        return values, "direct_cuda_fp32_linspace"
    if dtype == torch.float16:
        return values.to(dtype=torch.float16), "cuda_fp32_linspace_cast_to_fp16"
    raise ValueError(f"Unsupported diagnostic dtype: {dtype}")


def run_dtype_visual(dtype, config, model_path, device):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    memory_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()

    visual = Qwen3VLVisionModel(config=config)
    state = {}
    with safe_open(str(model_path / "model.safetensors"), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith("model.visual."):
                state[key[len("model.visual.") :]] = checkpoint.get_tensor(key)
    load_result = visual.load_state_dict(state, strict=True)
    del state
    gc.collect()
    visual.to(device=device, dtype=dtype)
    visual.eval()
    parameter_count = sum(parameter.numel() for parameter in visual.parameters())

    handles, records = attach_hooks(visual)
    grid_thw = torch.tensor([[1, 28, 28]], device=device, dtype=torch.long)
    pixel_values, input_generation = create_input(dtype, device)

    with torch.no_grad():
        final_hidden, deepstack_outputs = visual(pixel_values, grid_thw)
        torch.cuda.synchronize()

    outputs = {
        "final_hidden": final_hidden,
        "deepstack_0": deepstack_outputs[0],
        "deepstack_1": deepstack_outputs[1],
        "deepstack_2": deepstack_outputs[2],
    }
    output_stats = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            **finite_summary(value),
        }
        for name, value in outputs.items()
    }
    input_stats = finite_summary(pixel_values)
    handles_removed = all(handle.remove() is None for handle in handles)
    memory = {
        "allocated_before_bytes": memory_before,
        "reserved_before_bytes": reserved_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocated_after_bytes": torch.cuda.memory_allocated(),
        "reserved_after_bytes": torch.cuda.memory_reserved(),
    }

    del visual
    del final_hidden
    del deepstack_outputs
    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "dtype": str(dtype),
        "attention_implementation": config._attn_implementation,
        "input_generation": input_generation,
        "parameter_count": parameter_count,
        "state_dict_loaded": True,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "module_hook_count": len(handles),
        "hooks_registered": True,
        "hooks_removed": handles_removed,
        "input": {
            "shape": list(pixel_values.shape),
            "dtype": str(pixel_values.dtype),
            **input_stats,
        },
        "outputs": output_stats,
        "all_outputs_finite": all(value["all_finite"] for value in output_stats.values()),
        "first_nonfinite_output_module": first_nonfinite(records),
        "nonfinite_output_record_count": sum(
            1 for record in records if not record["all_finite"]
        ),
        "module_output_records": records,
        "memory": memory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    result_path = Path(args.result_path).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(device)
    started_utc = datetime.now(timezone.utc).isoformat()
    result = {
        "phase": "Phase 9.2-C2-R1",
        "audit_mode": "PYTORCH_FP32_VS_FP16_NONFINITE_DIAGNOSIS_ONLY",
        "started_utc": started_utc,
        "hostname": platform.node(),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_identity": {
            "local_path": str(model_path),
            "weights_path": str(model_path / "model.safetensors"),
            "weights_size_bytes": (model_path / "model.safetensors").stat().st_size,
            "weights_sha256": sha256_file(model_path / "model.safetensors"),
            "config_sha256": sha256_file(model_path / "config.json"),
        },
        "boundary": {
            "grid_thw": [1, 28, 28],
            "pixel_values_shape": [784, 1536],
            "pixel_values_values": "same deterministic linspace [-1,1] as Phase 9.2-C2",
            "output_names": [
                "final_hidden",
                "deepstack_0",
                "deepstack_1",
                "deepstack_2",
            ],
        },
        "input_generation_audit": {
            "element_count": 784 * 1536,
            "phase_9_2_c2_cuda_fp16_linspace": finite_summary(
                torch.linspace(
                    -1.0,
                    1.0,
                    784 * 1536,
                    device=device,
                    dtype=torch.float16,
                ).reshape(784, 1536)
            ),
            "finite_cast_fp16_from_fp32": finite_summary(
                torch.linspace(
                    -1.0,
                    1.0,
                    784 * 1536,
                    device=device,
                    dtype=torch.float32,
                )
                .reshape(784, 1536)
                .to(dtype=torch.float16)
            ),
            "direct_cuda_fp32_linspace": finite_summary(
                torch.linspace(
                    -1.0,
                    1.0,
                    784 * 1536,
                    device=device,
                    dtype=torch.float32,
                ).reshape(784, 1536)
            ),
        },
        "prohibited_operations_performed": {
            "tensorrt_rebuild": False,
            "tensorrt_execution": False,
            "benchmark_run": False,
            "latency_measured": False,
            "optimization_performed": False,
            "quantization_performed": False,
            "model_modified": False,
            "cuda_kernel_modified": False,
            "environment_modified": False,
        },
    }

    try:
        config = Qwen3VLConfig.from_pretrained(str(model_path))
        config.vision_config._attn_implementation = "eager"
        result["fp32"] = run_dtype_visual(torch.float32, config.vision_config, model_path, device)
        result["fp16"] = run_dtype_visual(torch.float16, config.vision_config, model_path, device)

        fp32_finite = result["fp32"]["all_outputs_finite"]
        fp16_finite = result["fp16"]["all_outputs_finite"]
        if fp32_finite and not fp16_finite:
            fp16_instability = "SUPPORTED_BY_THIS_OBSERVATION"
            input_mismatch = "NOT_SUPPORTED_FOR_FINITE_FP32_REFERENCE"
        elif not fp32_finite:
            fp16_instability = "NOT_ISOLATED"
            input_mismatch = "PARTIALLY_SUPPORTED_INPUT_OR_MODEL_NONFINITE"
        elif fp16_finite:
            fp16_instability = "NOT_REPRODUCED"
            input_mismatch = "NOT_SUPPORTED"
        else:
            fp16_instability = "UNKNOWN"
            input_mismatch = "UNKNOWN"

        result["diagnostic_summary"] = {
            "fp32_all_outputs_finite": fp32_finite,
            "fp16_all_outputs_finite": fp16_finite,
            "fp16_numerical_instability": fp16_instability,
            "input_or_boundary_mismatch": input_mismatch,
            "first_nonfinite_module_fp32": result["fp32"]["first_nonfinite_output_module"],
            "first_nonfinite_module_fp16": result["fp16"]["first_nonfinite_output_module"],
            "specific_submodule_instability": (
                "SUPPORTED_BY_FIRST_NONFINITE_HOOK_OBSERVATION"
                if result["fp16"]["first_nonfinite_output_module"]
                else "NOT_OBSERVED"
            ),
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

    if "failure" in result:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
