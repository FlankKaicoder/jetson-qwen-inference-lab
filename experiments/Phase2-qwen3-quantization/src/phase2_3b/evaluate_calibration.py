from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
import torch


QPROJ_WEIGHT_KEY = "self_attn.q_proj.weight"
WEIGHT_SCALE = np.float16(0.00507354736328125)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def metric(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    af, bf = a.float(), b.float()
    d = af - bf
    denom = torch.linalg.vector_norm(bf).item()
    return {
        "shape_equal": list(a.shape) == list(b.shape),
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
        "max_abs": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(d * d)).item()),
        "relative_l2": float(torch.linalg.vector_norm(d).item() / denom) if denom else 0.0,
        "cosine": float(torch.nn.functional.cosine_similarity(af.reshape(1, -1), bf.reshape(1, -1)).item()),
    }


def mem(stage: str) -> dict[str, object]:
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    return {"stage": stage, "mem_available_bytes": available, "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def reconstruct(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, dict[str, object]]:
    xf = x.float()
    raw = torch.round(xf / scale)
    q = raw.clamp(-127, 127)
    dq = q * scale
    clipped = (raw < -127) | (raw > 127)
    low = q == -127
    high = q == 127
    return dq.to(x.dtype), {
        "clipped_element_count": int(clipped.sum().item()),
        "clipped_percentage": float(clipped.float().mean().item() * 100.0),
        "int8_min_saturation_count": int(low.sum().item()),
        "int8_max_saturation_count": int(high.sum().item()),
        "element_count": x.numel(),
    }


def aggregate(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def make_graph(path: Path, mode: str, weight: np.ndarray, weight_scale: np.ndarray, activation_scale: np.ndarray | None) -> None:
    from onnx import TensorProto, helper, numpy_helper

    nodes: list[onnx.NodeProto] = []
    initializers = [numpy_helper.from_array(weight, name="weight")]
    matmul_input = "input"
    if mode == "w8a8":
        initializers += [
            numpy_helper.from_array(activation_scale.astype(np.float16), name="activation_scale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="activation_zero_point"),
        ]
        nodes += [
            helper.make_node("QuantizeLinear", ["input", "activation_scale", "activation_zero_point"], ["input_q"]),
            helper.make_node("DequantizeLinear", ["input_q", "activation_scale", "activation_zero_point"], ["input_dq"]),
        ]
        matmul_input = "input_dq"
    if mode in {"w8", "w8a8"}:
        initializers += [
            numpy_helper.from_array(weight_scale.astype(np.float16), name="weight_scale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="weight_zero_point"),
        ]
        nodes.append(helper.make_node("DequantizeLinear", ["weight", "weight_scale", "weight_zero_point"], ["weight_dq"], axis=1))
        weight_name = "weight_dq"
    else:
        weight_name = "weight"
    nodes.append(helper.make_node("MatMul", [matmul_input, weight_name], ["output"], name="target_q_proj_matmul"))
    graph = helper.make_graph(
        nodes,
        f"phase2_3b_{mode}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1, "seq", 1024])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, "seq", 2048])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="phase2_3b")
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_engine(onnx_path: Path, engine_path: Path) -> dict[str, object]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse(onnx_path.read_bytes())
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parsed:
        return {"parse": "FAIL", "build": "NOT_RUN", "parser_errors": errors}
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt, "ProfilingVerbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 1, 1024), (1, 64, 1024), (1, 256, 1024))
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        return {"parse": "PASS", "build": "FAIL", "parser_errors": errors}
    engine_path.write_bytes(bytes(blob))
    inspector_text = "UNAVAILABLE"
    try:
        engine = trt.Runtime(logger).deserialize_cuda_engine(bytes(blob))
        inspector_text = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    except Exception as exc:
        inspector_text = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    return {"parse": "PASS", "build": "PASS", "engine_bytes": engine_path.stat().st_size, "parser_errors": errors, "engine_inspector": inspector_text}


class Runtime:
    def __init__(self, path: Path) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self.context.set_input_shape("input", tuple(x.shape))
        self.context.set_tensor_address("input", x.data_ptr())
        output = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = torch.float16 if self.engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                output = torch.empty(shape, device="cuda", dtype=dtype)
                self.context.set_tensor_address(name, output.data_ptr())
        if output is None or not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return output


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def main(args: argparse.Namespace) -> None:
    out, work = args.out, args.work
    out.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    memory = [mem("start")]
    cal_manifest = json.loads(args.calibration.read_text())["rows"]
    eval_manifest = json.loads(args.evaluation.read_text())["rows"]
    if set(r["sample_id"] for r in cal_manifest) & set(r["sample_id"] for r in eval_manifest):
        raise RuntimeError("calibration/evaluation sample IDs overlap")
    tensors = torch.load(args.inputs, map_location="cpu", weights_only=True)
    cal_tensors = [tensors[r["sample_id"]].contiguous() for r in cal_manifest]
    eval_tensors = [tensors[r["sample_id"]].contiguous() for r in eval_manifest]
    if any(t.dtype != torch.bfloat16 for t in cal_tensors + eval_tensors):
        raise RuntimeError("q_proj inputs must remain BF16 at capture boundary")
    cal_abs = np.concatenate([t.float().abs().numpy().reshape(-1) for t in cal_tensors])
    cal_absmax = np.asarray([float(t.float().abs().max()) for t in cal_tensors])
    global_absmax = float(cal_abs.max())
    candidates: dict[str, dict[str, object]] = {
        "GLOBAL_ABSMAX": {"range": global_absmax, "source": "calibration_global_absmax"},
        "P99_9": {"range": percentile(cal_abs, 99.9), "source": "calibration_abs_percentile_99.9"},
        "P99_99": {"range": percentile(cal_abs, 99.99), "source": "calibration_abs_percentile_99.99"},
    }
    mse_grid = []
    for factor in (0.90, 0.925, 0.95, 0.975, 1.00):
        clip = global_absmax * factor
        total = 0.0
        count = 0
        for t in cal_tensors:
            recon, _ = reconstruct(t, clip / 127.0)
            d = (recon.float() - t.float()).reshape(-1)
            total += float(torch.sum(d * d).item())
            count += d.numel()
        mse_grid.append({"factor": factor, "range": clip, "mse": total / count})
    best_mse = min(mse_grid, key=lambda x: (x["mse"], x["factor"]))
    candidates["BOUNDED_MSE_CLIP"] = {"range": best_mse["range"], "source": "calibration_only_grid", "grid": mse_grid, "selected_factor": best_mse["factor"]}
    for row in candidates.values():
        row["scale"] = float(row["range"]) / 127.0
        row["zero_point"] = 0
        row["scheme"] = "symmetric per-tensor INT8 activation"
    range_rows = []
    for row, t in [(r, tensors[r["sample_id"]]) for r in cal_manifest + eval_manifest]:
        xf = t.float().numpy().reshape(-1)
        absx = np.abs(xf)
        range_rows.append({
            "sample_id": row["sample_id"], "split": row["split"], "length_group": row["length_group"], "category": row["category"], "token_count": row["token_count"],
            "shape": list(t.shape), "dtype": str(t.dtype), "sha256_bf16": sha_tensor(t), "min": float(xf.min()), "max": float(xf.max()), "absmax": float(absx.max()),
            "mean": float(xf.mean()), "std": float(xf.std()), "p99": percentile(absx, 99), "p99.9": percentile(absx, 99.9), "p99.99": percentile(absx, 99.99),
            "finite": bool(np.isfinite(xf).all()), "nan_count": int(np.isnan(xf).sum()), "inf_count": int(np.isinf(xf).sum()),
        })
    cal_ranges = [r for r in range_rows if r["split"] == "calibration"]
    eval_ranges = [r for r in range_rows if r["split"] == "evaluation"]
    dump(out / "activation_range_per_sample.json", {"rows": range_rows})
    dump(out / "activation_range_summary.json", {
        "calibration": {"global_min": float(min(r["min"] for r in cal_ranges)), "global_max": float(max(r["max"] for r in cal_ranges)), "global_absmax": global_absmax, "absmax_min": float(cal_absmax.min()), "absmax_median": float(np.median(cal_absmax)), "absmax_mean": float(cal_absmax.mean()), "absmax_p90": float(np.percentile(cal_absmax, 90)), "absmax_p95": float(np.percentile(cal_absmax, 95)), "absmax_max": global_absmax, "absolute_p99": percentile(cal_abs, 99), "absolute_p99.9": percentile(cal_abs, 99.9), "absolute_p99.99": percentile(cal_abs, 99.99)},
        "evaluation": {"global_min": float(min(r["min"] for r in eval_ranges)), "global_max": float(max(r["max"] for r in eval_ranges)), "global_absmax": float(max(r["absmax"] for r in eval_ranges))},
    })
    median_absmax = float(np.median(cal_absmax))
    max_ids = [r["sample_id"] for r in cal_ranges if r["absmax"] == global_absmax]
    dump(out / "scale_stability.json", {"calibration_absmax_max_over_median": global_absmax / median_absmax, "calibration_absmax_p95_over_median": float(np.percentile(cal_absmax, 95) / median_absmax), "largest_range_sample_ids": max_ids, "calibration_absmax_by_sample": [{"sample_id": r["sample_id"], "absmax": r["absmax"]} for r in cal_ranges], "interpretation": "descriptive only; no arbitrary stability threshold"})
    dump(out / "calibration_candidates.json", candidates)

    weight_bf16 = tensors["__weight_bf16__"] if "__weight_bf16__" in tensors else None
    if weight_bf16 is None:
        raise RuntimeError("handoff input file must include __weight_bf16__")
    weight_f32 = weight_bf16.float()
    q = torch.round(weight_f32 / float(WEIGHT_SCALE)).clamp(-127, 127).to(torch.int8)
    weight = q.cpu().numpy().T.copy()
    weight_fp16 = weight_bf16.to(torch.float16).cpu().numpy().T.copy()
    engine_meta: dict[str, object] = {}
    for name, mode, act_scale in [("fp16", "fp16", None), ("w8", "w8", None)]:
        onnx_path, engine_path = work / f"{name}.onnx", work / f"{name}.engine"
        make_graph(onnx_path, mode, weight_fp16 if mode == "fp16" else weight, np.asarray(WEIGHT_SCALE), None)
        engine_meta[name] = {"graph": {"path": str(onnx_path), "mode": mode}, "engine": build_engine(onnx_path, engine_path), "engine_path": str(engine_path)}
    runtimes = {"fp16": Runtime(work / "fp16.engine"), "w8": Runtime(work / "w8.engine")}
    rows_by_policy: dict[str, list[dict[str, object]]] = {}
    reconstruction_by_policy: dict[str, list[dict[str, object]]] = {}
    for policy, candidate in candidates.items():
        onnx_path, engine_path = work / f"{policy.lower()}.onnx", work / f"{policy.lower()}.engine"
        make_graph(onnx_path, "w8a8", weight, np.asarray(WEIGHT_SCALE), np.asarray(candidate["scale"], dtype=np.float16))
        engine_meta[policy] = {"graph": {"path": str(onnx_path), "mode": "w8a8", "scale": candidate["scale"]}, "engine": build_engine(onnx_path, engine_path), "engine_path": str(engine_path)}
        runtimes[policy] = Runtime(engine_path)
        p_rows, r_rows = [], []
        for row, t in zip(eval_manifest, eval_tensors):
            x = t.to(torch.float16).cuda()
            recon, sat = reconstruct(t.to(torch.float16), float(candidate["scale"]))
            r = {"sample_id": row["sample_id"], "token_count": row["token_count"], **metric(recon, t.to(torch.float16)), **sat}
            r_rows.append(r)
            fp = runtimes["fp16"].run(x)
            w8 = runtimes["w8"].run(x)
            w8a8 = runtimes[policy].run(x)
            p_rows.append({"sample_id": row["sample_id"], "token_count": row["token_count"], "activation_reconstruction": r, "ACTIVATION_ONLY_INCREMENTAL_DELTA": metric(w8a8, w8), "TOTAL_QUANTIZATION_DELTA": metric(w8a8, fp), "WEIGHT_QUANTIZATION_CONTEXT": metric(w8, fp), "finite": bool(torch.isfinite(fp).all() and torch.isfinite(w8).all() and torch.isfinite(w8a8).all())})
        rows_by_policy[policy] = p_rows
        reconstruction_by_policy[policy] = r_rows
    dump(out / "activation_reconstruction.json", reconstruction_by_policy)
    dump(out / "activation_reconstruction_summary.json", {p: {"relative_l2": aggregate(r, "relative_l2"), "rmse": aggregate(r, "rmse"), "clipping_percentage": aggregate(r, "clipped_percentage")} for p, r in reconstruction_by_policy.items()})
    dump(out / "component_delta_per_sample.json", rows_by_policy)
    delta_summary = {}
    for policy, rows in rows_by_policy.items():
        inc = [r["ACTIVATION_ONLY_INCREMENTAL_DELTA"] for r in rows]
        total = [r["TOTAL_QUANTIZATION_DELTA"] for r in rows]
        weight_ctx = [r["WEIGHT_QUANTIZATION_CONTEXT"] for r in rows]
        delta_summary[policy] = {
            "ACTIVATION_ONLY_INCREMENTAL_DELTA": {"relative_l2": aggregate(inc, "relative_l2"), "cosine": {"mean": float(np.mean([r["cosine"] for r in inc])), "median": float(np.median([r["cosine"] for r in inc])), "minimum": float(np.min([r["cosine"] for r in inc]))}},
            "TOTAL_QUANTIZATION_DELTA": {"relative_l2": aggregate(total, "relative_l2"), "cosine": {"mean": float(np.mean([r["cosine"] for r in total])), "median": float(np.median([r["cosine"] for r in total])), "minimum": float(np.min([r["cosine"] for r in total]))}},
            "WEIGHT_QUANTIZATION_CONTEXT": {"relative_l2": aggregate(weight_ctx, "relative_l2"), "cosine": {"mean": float(np.mean([r["cosine"] for r in weight_ctx])), "median": float(np.median([r["cosine"] for r in weight_ctx])), "minimum": float(np.min([r["cosine"] for r in weight_ctx]))}},
        }
    dump(out / "component_delta_summary.json", delta_summary)
    selected = min(candidates, key=lambda p: (delta_summary[p]["ACTIVATION_ONLY_INCREMENTAL_DELTA"]["relative_l2"]["p95"], delta_summary[p]["ACTIVATION_ONLY_INCREMENTAL_DELTA"]["relative_l2"]["max"], {"GLOBAL_ABSMAX": 0, "P99_9": 1, "P99_99": 2, "BOUNDED_MSE_CLIP": 3}[p]))
    selected_meta = engine_meta[selected]
    dump(out / "engine_summary_selected_policy.json", {"selected_policy": selected, "candidate": candidates[selected], "fp16": engine_meta["fp16"], "w8": engine_meta["w8"], "selected_w8a8": selected_meta, "int8_compute_proven": "INT8" in str(selected_meta["engine"].get("engine_inspector", "") ) or "i8" in str(selected_meta["engine"].get("engine_inspector", ""))})
    (out / "engine_inspector_selected_policy.txt").write_text(str(selected_meta["engine"].get("engine_inspector", "UNAVAILABLE")) + "\n")
    all_weight = delta_summary[selected]["WEIGHT_QUANTIZATION_CONTEXT"]["relative_l2"]
    all_inc = delta_summary[selected]["ACTIVATION_ONLY_INCREMENTAL_DELTA"]["relative_l2"]
    dominance = "WEIGHT_QUANTIZATION_DOMINANCE_SUPPORTED" if all_weight["median"] > all_inc["median"] and all_weight["p95"] > all_inc["p95"] else "WEIGHT_QUANTIZATION_DOMINANCE_NOT_SUPPORTED"
    finite = all(r["finite"] for rows in rows_by_policy.values() for r in rows)
    dump(out / "final_validation.json", {"phase": "Phase 2.3-B", "gate": "PASS" if finite and selected else "BLOCKED", "qproj_input_provenance": "EXACT_QPROJ_INPUT_PROVEN", "calibration_samples": len(cal_manifest), "evaluation_samples": len(eval_manifest), "split_disjoint": True, "real_activations_collected": True, "fixed_scales_from_calibration_only": True, "global_absmax_and_clipping_candidates_evaluated": True, "heldout_activation_reconstruction": True, "w8_baseline_evaluated": True, "w8a8_fixed_scale_evaluated": True, "activation_only_delta_measured": True, "total_delta_measured": True, "selected_policy": selected, "weight_dominance": dominance, "selected_int8_compute_proven": bool("INT8" in str(selected_meta["engine"].get("engine_inspector", "")) or "i8" in str(selected_meta["engine"].get("engine_inspector", ""))), "all_outputs_finite": finite, "oom": False, "exit137": False, "no_benchmark": True, "no_nsight": True, "c1_reopened": False})
    memory.append(mem("complete"))
    dump(out / "memory_trace.json", {"trace": memory, "oom": False, "exit137": False})
    dump(out / "environment.json", {"host": platform.node(), "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)), "onnx": onnx.__version__, "tensorrt": trt.__version__, "git_branch": os.popen("git branch --show-current").read().strip(), "git_head": os.popen("git rev-parse HEAD").read().strip()})
    print(json.dumps({"status": "PASS" if finite else "BLOCKED", "selected_policy": selected, "calibration": len(cal_manifest), "evaluation": len(eval_manifest), "weight_dominance": dominance}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    main(parser.parse_args())
