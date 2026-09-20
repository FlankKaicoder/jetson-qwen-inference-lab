#!/usr/bin/env python3
"""Compare, benchmark, generate, or profile one Phase 8.5 engine."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import resource
from pathlib import Path

import torch
import tensorrt as trt


def memory_snapshot(stage: str) -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "stage": stage, "cuda_free_bytes": free, "cuda_total_bytes": total,
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


class Runner:
    def __init__(self, path: Path, plugin_so: Path | None = None):
        if plugin_so:
            ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"deserialize failed: {path}")
        self.stream = torch.cuda.current_stream()

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context = self.engine.create_execution_context()
        for name, tensor in inputs.items():
            if not context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"shape rejected: {name}")
            context.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(context.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                if dtype != trt.DataType.HALF:
                    raise RuntimeError(f"unexpected output dtype {name}: {dtype}")
                outputs[name] = torch.empty(shape, device="cuda", dtype=torch.float16)
                context.set_tensor_address(name, outputs[name].data_ptr())
        if not context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("execute failed")
        self.stream.synchronize()
        return outputs


def inputs(kind: str, seed: int = 20260920) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, 8, 1024) if kind == "prefill" else (1, 1, 1024)
    hidden = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16)
    if kind == "prefill":
        pos = torch.arange(8, device="cuda", dtype=torch.long).unsqueeze(0)
        return {"hidden_states": hidden, "position_ids": pos}
    pos = torch.tensor([[8]], device="cuda", dtype=torch.long)
    result = {"hidden_states": hidden, "position_ids": pos}
    for i in range(28):
        result[f"past_k{i}"] = torch.zeros((1, 8, 8, 128), device="cuda", dtype=torch.float16)
        result[f"past_v{i}"] = torch.zeros((1, 8, 8, 128), device="cuda", dtype=torch.float16)
    return result


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.float()
    candidate = candidate.float()
    delta = candidate - reference
    norm = torch.linalg.vector_norm(reference)
    finite_ratio = float(torch.isfinite(candidate).float().mean())
    return {
        "relative_l2": float(torch.linalg.vector_norm(delta) /
                             torch.clamp(norm, min=1.0e-30)),
        "max_abs": float(delta.abs().max()),
        "finite_ratio": finite_ratio,
        "shape_equal": list(reference.shape) == list(candidate.shape),
    }


def compare_outputs(base_outputs: dict[str, torch.Tensor],
                    plugin_outputs: dict[str, torch.Tensor],
                    target_outputs: list[str]) -> dict:
    common = sorted(set(base_outputs).intersection(plugin_outputs))
    all_finite = all(
        bool(torch.isfinite(plugin_outputs[name]).all())
        and bool(torch.isfinite(base_outputs[name]).all())
        for name in common
    )
    hidden = {name: metric(base_outputs[name], plugin_outputs[name])
              for name in common if name.startswith("hidden_l")}
    targets = {name: metric(base_outputs[name], plugin_outputs[name])
               for name in target_outputs if name in base_outputs and name in plugin_outputs}
    kv_names = [name for name in common if name.startswith("present_k") or name.startswith("present_v")]
    kv = {
        "finite_ratio": float(sum(
            bool(torch.isfinite(base_outputs[name]).all()) and
            bool(torch.isfinite(plugin_outputs[name]).all())
            for name in kv_names) / max(len(kv_names), 1)),
        "shape_mismatches": [name for name in kv_names
                             if list(base_outputs[name].shape) != list(plugin_outputs[name].shape)],
    }
    final = hidden.get("hidden_l27")
    target_max = max((row["relative_l2"] for row in targets.values()), default=0.0)
    final_pass = final is not None and final["finite_ratio"] == 1.0 and final["relative_l2"] <= 0.05
    target_pass = all(row["finite_ratio"] == 1.0 and row["relative_l2"] <= 0.001
                      for row in targets.values())
    return {
        "common_output_count": len(common),
        "all_outputs_finite": all_finite,
        "hidden_metrics": hidden,
        "replaced_norm_metrics": targets,
        "kv_summary": kv,
        "replaced_norm_max_relative_l2": target_max,
        "gate_hidden_l27_pass": final_pass,
        "gate_replaced_norm_pass": target_pass,
    }


def do_compare(args) -> dict:
    target_outputs = [line.strip() for line in args.target_outputs.split(",") if line.strip()]
    if args.scale == 0:
        runner = Runner(args.baseline_engine, None)
        base_outputs = {name: tensor.cpu() for name, tensor in runner.run(inputs(args.kind)).items()}
        plugin_outputs = {name: tensor.cpu() for name, tensor in runner.run(inputs(args.kind)).items()}
        result = compare_outputs(base_outputs, plugin_outputs, target_outputs)
        del runner, base_outputs, plugin_outputs
    else:
        runner = Runner(args.baseline_engine, None)
        base_outputs = {name: tensor.cpu() for name, tensor in runner.run(inputs(args.kind)).items()}
        del runner
        gc.collect()
        torch.cuda.empty_cache()
        plugin_runner = Runner(args.plugin_engine, args.plugin_so)
        plugin_outputs = {name: tensor.cpu() for name, tensor in plugin_runner.run(inputs(args.kind)).items()}
        result = compare_outputs(base_outputs, plugin_outputs, target_outputs)
        del plugin_runner, base_outputs, plugin_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def do_measure(args) -> dict:
    plugin_so = args.plugin_so if args.scale > 0 else None
    runner = Runner(args.engine, plugin_so)
    workload = inputs(args.kind)
    samples = []
    for _ in range(args.warmup):
        runner.run(workload)
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(args.repetitions):
        begin.record(runner.stream)
        runner.run(workload)
        end.record(runner.stream)
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    result = {
        "engine": str(args.engine), "kind": args.kind, "scale": args.scale,
        "warmup": args.warmup, "repetitions": args.repetitions,
        "timing_boundary": "CUDA events around execute_async_v3",
        "samples_ms": samples, "mean_ms": sum(samples) / len(samples),
        "min_ms": min(samples), "max_ms": max(samples),
        "memory": memory_snapshot("after_measure"),
    }
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def final_logits(hidden: torch.Tensor, norm_weight: torch.Tensor,
                 head_weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    normalized = hidden.float()
    normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + epsilon)
    normalized = norm_weight.float() * normalized
    return torch.nn.functional.linear(normalized, head_weight.float())


def generate_once_prefill(engine: Path, plugin_so: Path | None,
                          token_ids: torch.Tensor,
                          head: dict[str, torch.Tensor]) -> list[int]:
    runner = Runner(engine, plugin_so)
    hidden = head["model.embed_tokens.weight"][token_ids].unsqueeze(0).to(torch.float16).contiguous()
    tokens = []
    for step in range(8):
        pos = torch.arange(hidden.shape[1], device="cuda", dtype=torch.long).unsqueeze(0)
        outputs = runner.run({"hidden_states": hidden, "position_ids": pos})
        tokens.append(int(torch.argmax(final_logits(outputs["hidden_l27"][:, -1, :],
                                                    head["model.norm.weight"],
                                                    head["model.embed_tokens.weight"])).item()))
        if step < 7:
            next_token = torch.tensor([[tokens[-1]]], device="cuda", dtype=torch.long)
            next_hidden = head["model.embed_tokens.weight"][next_token].to(torch.float16)
            hidden = torch.cat([hidden, next_hidden], dim=1).contiguous()
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return tokens


def generate_once_decode(prefill_engine: Path, decode_engine: Path,
                         plugin_so: Path | None, token_ids: torch.Tensor,
                         head: dict[str, torch.Tensor]) -> list[int]:
    prefill_runner = Runner(prefill_engine, plugin_so)
    hidden = head["model.embed_tokens.weight"][token_ids].unsqueeze(0).to(torch.float16).contiguous()
    pos = torch.arange(token_ids.numel(), device="cuda", dtype=torch.long).unsqueeze(0)
    outputs = prefill_runner.run({"hidden_states": hidden, "position_ids": pos})
    past_k = [outputs[f"present_k{i}"] for i in range(28)]
    past_v = [outputs[f"present_v{i}"] for i in range(28)]
    tokens = [int(torch.argmax(final_logits(outputs["hidden_l27"][:, -1, :],
                                            head["model.norm.weight"],
                                            head["model.embed_tokens.weight"])).item())]
    del prefill_runner
    gc.collect()
    torch.cuda.empty_cache()
    decode_runner = Runner(decode_engine, plugin_so)
    for step in range(8):
        next_token = torch.tensor([[tokens[-1]]], device="cuda", dtype=torch.long)
        next_pos = torch.tensor([[token_ids.numel() + step]], device="cuda", dtype=torch.long)
        next_hidden = head["model.embed_tokens.weight"][next_token].to(torch.float16).contiguous()
        inputs = {"hidden_states": next_hidden, "position_ids": next_pos}
        inputs.update({f"past_k{i}": past_k[i] for i in range(28)})
        inputs.update({f"past_v{i}": past_v[i] for i in range(28)})
        outputs = decode_runner.run(inputs)
        past_k = [outputs[f"present_k{i}"] for i in range(28)]
        past_v = [outputs[f"present_v{i}"] for i in range(28)]
        tokens.append(int(torch.argmax(final_logits(outputs["hidden_l27"][:, -1, :],
                                                    head["model.norm.weight"],
                                                    head["model.embed_tokens.weight"])).item()))
    del decode_runner
    gc.collect()
    torch.cuda.empty_cache()
    return tokens


def do_generation(args) -> dict:
    from phase8_5_prepare_source import load_head_weights
    from transformers import AutoTokenizer

    head = load_head_weights(args.model_dir)
    head = {name: tensor.cuda().to(torch.float16) for name, tensor in head.items()}
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    encoded = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
    token_ids = encoded.input_ids[0, :8].cuda()
    if args.kind == "prefill":
        baseline_tokens = generate_once_prefill(args.baseline_engine, None, token_ids, head)
        candidate_tokens = generate_once_prefill(
            args.baseline_engine if args.scale == 0 else args.plugin_engine,
            args.plugin_so if args.scale > 0 else None, token_ids, head)
    else:
        baseline_tokens = generate_once_decode(args.baseline_prefill_engine,
                                               args.baseline_decode_engine,
                                               None, token_ids, head)
        candidate_tokens = generate_once_decode(
            args.baseline_prefill_engine if args.scale == 0 else args.plugin_prefill_engine,
            args.baseline_decode_engine if args.scale == 0 else args.plugin_decode_engine,
            args.plugin_so if args.scale > 0 else None, token_ids, head)
    del head
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "prompt_token_ids": token_ids.tolist(),
        "generated_tokens": 8, "baseline_tokens": baseline_tokens,
        "candidate_tokens": candidate_tokens,
        "match": baseline_tokens == candidate_tokens,
    }


def do_profile(args) -> dict:
    plugin_so = args.plugin_so if args.scale > 0 else None
    runner = Runner(args.engine, plugin_so)
    workload = inputs(args.kind)
    for _ in range(args.warmup):
        runner.run(workload)
    torch.cuda.synchronize()
    samples = []
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(args.profile_runs):
        begin.record(runner.stream)
        runner.run(workload)
        end.record(runner.stream)
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    torch.cuda.synchronize()
    result = {
        "engine": str(args.engine), "kind": args.kind, "scale": args.scale,
        "warmup": args.warmup, "profile_runs": args.profile_runs,
        "samples_ms": samples, "mean_ms": sum(samples) / len(samples),
    }
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "measure", "generation", "profile"], required=True)
    parser.add_argument("--kind", choices=["prefill", "decode"], required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--baseline-engine", type=Path)
    parser.add_argument("--plugin-engine", type=Path)
    parser.add_argument("--baseline-prefill-engine", type=Path)
    parser.add_argument("--baseline-decode-engine", type=Path)
    parser.add_argument("--plugin-prefill-engine", type=Path)
    parser.add_argument("--plugin-decode-engine", type=Path)
    parser.add_argument("--plugin-so", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--target-outputs", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-runs", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "compare": do_compare,
        "measure": do_measure,
        "generation": do_generation,
        "profile": do_profile,
    }[args.mode](args)
    result["mode"] = args.mode
    result["kind"] = args.kind
    result["scale"] = args.scale
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
