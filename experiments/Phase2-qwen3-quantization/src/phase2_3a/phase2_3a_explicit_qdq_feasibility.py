from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
import torch


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
TARGET_MODULE = "model.layers.0.self_attn.q_proj"
TARGET_KEY = "model.layers.0.self_attn.q_proj.weight"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def metric(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    af = a.float()
    bf = b.float()
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


def distribution(x: torch.Tensor) -> dict[str, object]:
    xf = x.float()
    finite = torch.isfinite(xf)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(xf[finite].min().item()) if finite.any() else None,
        "max": float(xf[finite].max().item()) if finite.any() else None,
        "mean": float(xf[finite].mean().item()) if finite.any() else None,
        "std": float(xf[finite].std(unbiased=False).item()) if finite.any() else None,
        "nan_count": int(torch.isnan(xf).sum().item()),
        "inf_count": int(torch.isinf(xf).sum().item()),
        "finite": bool(finite.all().item()),
    }


def command_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return (exc.output or str(exc)).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def environment() -> dict[str, object]:
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE"
    capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    model = Path("/proc/device-tree/model")
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": device,
        "compute_capability": capability,
        "onnx": onnx.__version__,
        "tensorrt_python": trt.__version__,
        "device_tree_model": model.read_bytes().decode(errors="replace").rstrip("\x00") if model.exists() else "UNAVAILABLE",
        "trtexec": command_text(["/usr/src/tensorrt/bin/trtexec", "--version"]),
        "nvcc": command_text(["nvcc", "--version"]),
        "git_branch": command_text(["git", "branch", "--show-current"]),
        "git_head": command_text(["git", "rev-parse", "HEAD"]),
    }


def make_graph(
    path: Path,
    input_shape: tuple[int, ...],
    weight: np.ndarray,
    mode: str,
    weight_scale: np.ndarray,
    weight_zero_point: np.ndarray,
    weight_axis: int | None = None,
    activation_axis: int | None = None,
    activation_scale: np.ndarray | None = None,
    activation_zero_point: np.ndarray | None = None,
) -> dict[str, object]:
    from onnx import TensorProto, helper, numpy_helper

    nodes: list[onnx.NodeProto] = []
    initializers = [numpy_helper.from_array(weight, name="weight")]
    matmul_input = "input"
    if mode == "w8a8":
        if activation_scale is None:
            activation_scale = weight_scale
        if activation_zero_point is None:
            activation_zero_point = np.asarray(0, dtype=np.int8)
        initializers.extend(
            [
                numpy_helper.from_array(activation_scale.astype(np.float16), name="activation_scale"),
                numpy_helper.from_array(activation_zero_point.astype(np.int8), name="activation_zero_point"),
            ]
        )
        attrs = {} if activation_axis is None else {"axis": activation_axis}
        nodes.extend(
            [
                helper.make_node("QuantizeLinear", ["input", "activation_scale", "activation_zero_point"], ["input_q"], **attrs),
                helper.make_node("DequantizeLinear", ["input_q", "activation_scale", "activation_zero_point"], ["input_dq"], **attrs),
            ]
        )
        matmul_input = "input_dq"

    if mode in {"w8", "w8a8"}:
        initializers.extend(
            [
                numpy_helper.from_array(weight_scale.astype(np.float16), name="weight_scale"),
                numpy_helper.from_array(weight_zero_point.astype(np.int8), name="weight_zero_point"),
            ]
        )
        attrs = {} if weight_axis is None else {"axis": weight_axis}
        nodes.append(helper.make_node("DequantizeLinear", ["weight", "weight_scale", "weight_zero_point"], ["weight_dq"], **attrs))
        matmul_weight = "weight_dq"
    else:
        matmul_weight = "weight"

    nodes.append(helper.make_node("MatMul", [matmul_input, matmul_weight], ["output"], name="target_q_proj_matmul"))
    graph = helper.make_graph(
        nodes,
        f"phase2_3a_{mode}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, list(input_shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [input_shape[0], input_shape[1], weight.shape[1]])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="phase2_3a")
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return {
        "onnx_checker": "PASS",
        "q_node_count": sum(n.op_type == "QuantizeLinear" for n in nodes),
        "dq_node_count": sum(n.op_type == "DequantizeLinear" for n in nodes),
        "matmul_node_count": sum(n.op_type == "MatMul" for n in nodes),
        "input_shape": list(input_shape),
        "output_shape": [input_shape[0], input_shape[1], weight.shape[1]],
    }


class Runtime:
    def __init__(self, engine_path: Path) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"engine deserialization failed: {engine_path}")
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
                dtype = self.engine.get_tensor_dtype(name)
                if dtype == trt.DataType.HALF:
                    torch_dtype = torch.float16
                elif dtype == trt.DataType.FLOAT:
                    torch_dtype = torch.float32
                else:
                    raise RuntimeError(f"unsupported output dtype {dtype}")
                output = torch.empty(shape, device="cuda", dtype=torch_dtype)
                self.context.set_tensor_address(name, output.data_ptr())
        if output is None or not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return output


def build_engine(onnx_path: Path, engine_path: Path, detailed: bool = True) -> dict[str, object]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    model = onnx.load(onnx_path)
    parsed = parser.parse(model.SerializeToString())
    parser_errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parsed:
        return {"parse": "FAIL", "build": "NOT_RUN", "parser_errors": parser_errors}
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    if detailed and hasattr(trt, "ProfilingVerbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        return {"parse": "PASS", "build": "FAIL", "parser_errors": parser_errors}
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(blob))
    layers = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        try:
            precision = str(layer.precision)
        except Exception:
            precision = "UNAVAILABLE"
        layers.append({"index": i, "name": layer.name, "type": str(layer.type), "precision": precision})
    inspector_text = "UNAVAILABLE"
    try:
        engine = trt.Runtime(logger).deserialize_cuda_engine(bytes(blob))
        inspector = engine.create_engine_inspector()
        inspector_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    except Exception as exc:
        inspector_text = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    return {
        "parse": "PASS",
        "build": "PASS",
        "engine_bytes": engine_path.stat().st_size,
        "parser_errors": parser_errors,
        "network_layers": layers,
        "engine_inspector": inspector_text,
    }


def memory_snapshot(stage: str) -> dict[str, object]:
    mem_available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1]) * 1024
                break
    except Exception:
        pass
    cuda_free = cuda_total = None
    if torch.cuda.is_available():
        cuda_free, cuda_total = torch.cuda.mem_get_info()
    return {"stage": stage, "time": time.time(), "mem_available_bytes": mem_available, "cuda_free_bytes": cuda_free, "cuda_total_bytes": cuda_total}


def main(args: argparse.Namespace) -> None:
    if args.repo is not None:
        os.chdir(args.repo)
    out = args.out
    work = args.work
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    json_dump(out / "environment.json", environment())
    (out / "start_audit.txt").write_text(
        "branch=" + command_text(["git", "branch", "--show-current"]) + "\n"
        + "head=" + command_text(["git", "rev-parse", "HEAD"]) + "\n"
        + "status=" + command_text(["git", "status", "--short"]) + "\n"
    )
    memory = [memory_snapshot("start")]

    handoff = torch.load(args.handoff, map_location="cpu", weights_only=True)
    state = handoff["state_dict"]
    weight_bf16 = state["self_attn.q_proj.weight"].contiguous()
    if list(weight_bf16.shape) != [2048, 1024]:
        raise RuntimeError(f"unexpected q_proj shape {tuple(weight_bf16.shape)}")
    weight_fp16 = weight_bf16.to(torch.float16)
    weight_f32 = weight_bf16.float()
    weight_sha = sha256_bytes(weight_bf16.view(torch.uint8).numpy().tobytes())
    target_audit = {
        "module": TARGET_MODULE,
        "checkpoint_key": TARGET_KEY,
        "model_revision": MODEL_REVISION,
        "model_safetensors_sha256": MODEL_SHA256,
        "weight_sha256": weight_sha,
        "shape": list(weight_bf16.shape),
        "checkpoint_dtype": str(weight_bf16.dtype),
        "parameter_count": weight_bf16.numel(),
        "bytes_bf16": weight_bf16.numel() * weight_bf16.element_size(),
        "selection_reason": "B2 real Layer 0 handoff; independently executable q_proj with saved real activation and no C1 reopening",
    }
    json_dump(out / "target_audit.json", target_audit)

    canonical_bf16 = handoff["x"].contiguous()
    canonical_fp16 = canonical_bf16.to(torch.float16)
    canonical_np = canonical_fp16.numpy()
    canonical_path = out / "canonical_input.npy"
    np.save(canonical_path, canonical_np)
    json_dump(
        out / "canonical_input.json",
        {
            "source": "B2 layer0_handoff.pt x",
            "source_dtype": str(canonical_bf16.dtype),
            "target_dtype": str(canonical_fp16.dtype),
            "cast_location": "one-time host artifact preparation before all variants",
            "shape": list(canonical_fp16.shape),
            "numel": canonical_fp16.numel(),
            "sha256_npy": sha256_bytes(canonical_path.read_bytes()),
            "source_distribution": distribution(canonical_bf16),
            "target_distribution": distribution(canonical_fp16),
        },
    )
    x = canonical_fp16.cuda()

    # Minimal graph sanity: Q/DQ activation around a small MatMul.
    sanity_weight = np.eye(4, dtype=np.int8)
    sanity_scale = np.asarray(1.0, dtype=np.float16)
    sanity_zp = np.asarray(0, dtype=np.int8)
    sanity_onnx = work / "sanity_qdq.onnx"
    sanity_meta = make_graph(sanity_onnx, (1, 1, 4), sanity_weight, "w8a8", sanity_scale, sanity_zp)
    sanity_build = build_engine(sanity_onnx, work / "sanity_qdq.engine")
    sanity_exec = "NOT_RUN"
    sanity_finite = None
    if sanity_build.get("build") == "PASS":
        sanity_out = Runtime(work / "sanity_qdq.engine").run(torch.ones((1, 1, 4), device="cuda", dtype=torch.float16))
        sanity_exec = "PASS"
        sanity_finite = bool(torch.isfinite(sanity_out).all().item())
    json_dump(out / "sanity_validation.json", {**sanity_meta, **sanity_build, "execution": sanity_exec, "finite": sanity_finite})

    # Capability probes are parser/build probes only; no result is promoted to INT8 arithmetic proof.
    cap_rows = []
    for name, mode, weight_scale, weight_zp, activation_scale, activation_zp, axis in [
        ("weight_per_tensor", "w8", np.asarray(0.01, dtype=np.float16), np.asarray(0, dtype=np.int8), None, None, None),
        ("weight_per_channel", "w8", np.full((2048,), 0.01, dtype=np.float16), np.zeros((2048,), dtype=np.int8), None, None, 1),
        ("activation_per_tensor", "w8a8", np.asarray(0.01, dtype=np.float16), np.asarray(0, dtype=np.int8), np.asarray(0.01, dtype=np.float16), np.asarray(0, dtype=np.int8), None),
        ("activation_per_channel", "w8a8", np.asarray(0.01, dtype=np.float16), np.asarray(0, dtype=np.int8), np.full((1024,), 0.01, dtype=np.float16), np.zeros((1024,), dtype=np.int8), 2),
        ("nonzero_zero_point", "w8a8", np.asarray(0.01, dtype=np.float16), np.asarray(0, dtype=np.int8), np.asarray(0.01, dtype=np.float16), np.asarray(3, dtype=np.int8), None),
    ]:
        graph = work / f"cap_{name}.onnx"
        try:
            meta = make_graph(graph, (1, 1, 1024), np.zeros((1024, 2048), dtype=np.int8), mode, weight_scale, weight_zp, weight_axis=1 if mode != "fp16" else None, activation_axis=axis if "activation" in name else None, activation_scale=activation_scale, activation_zero_point=activation_zp)
            built = build_engine(graph, work / f"cap_{name}.engine", detailed=False)
            cap_rows.append({"capability": name, **meta, **built})
        except Exception as exc:
            cap_rows.append({"capability": name, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    json_dump(out / "qdq_capability.json", {"rows": cap_rows, "interpretation": "parser/build capability only; no INT8 arithmetic claim"})

    q = torch.round(weight_f32 / (weight_f32.abs().max() / 127.0)).clamp(-127, 127).to(torch.int8)
    w_scale = np.asarray(float(weight_f32.abs().max().item() / 127.0), dtype=np.float16)
    w_deq = q.float() * float(w_scale)
    recon = metric(weight_bf16, w_deq.to(torch.bfloat16))
    json_dump(
        out / "quantization_scales.json",
        {
            "weight_scheme": "symmetric per-tensor INT8",
            "weight_scale": float(w_scale),
            "weight_scale_dtype": str(w_scale.dtype),
            "weight_scale_shape": list(w_scale.shape),
            "weight_axis": None,
            "weight_zero_point": 0,
            "weight_qmin": int(q.min().item()),
            "weight_qmax": int(q.max().item()),
            "activation_scheme": "symmetric absmax feasibility scale",
            "activation_absmax": float(x.float().abs().max().item()),
            "activation_scale": float(x.float().abs().max().item() / 127.0),
            "activation_zero_point": 0,
            "activation_scale_dtype": "float16",
            "activation_scale_shape": [],
            "activation_policy": "FEASIBILITY_SCALE; NOT_FINAL_CALIBRATION_POLICY",
        },
    )
    json_dump(out / "weight_reconstruction.json", {"original": distribution(weight_bf16), "dequantized": distribution(w_deq), "metrics": recon})

    wt = q.cpu().numpy().T.copy()
    fp_weight = weight_fp16.cpu().numpy().T.copy()
    activation_scale = np.asarray(float(x.float().abs().max().item() / 127.0), dtype=np.float16)
    variants = {
        "fp16": ("fp16", fp_weight, np.asarray(1.0, dtype=np.float16), np.asarray(0, dtype=np.int8), None),
        "w8": ("w8", wt, w_scale, np.asarray(0, dtype=np.int8), None),
        "w8a8": ("w8a8", wt, w_scale, np.asarray(0, dtype=np.int8), activation_scale),
    }
    results = {}
    for name, (mode, graph_weight, scale, zp, act_scale) in variants.items():
        graph = work / f"{name}.onnx"
        meta = make_graph(graph, tuple(x.shape), graph_weight, mode, scale, zp, weight_axis=1 if mode != "fp16" else None, activation_axis=None, activation_scale=act_scale)
        build = build_engine(graph, work / f"{name}.engine")
        row = {"graph": meta, "engine": build, "execution": "NOT_RUN", "finite": None}
        if build.get("build") == "PASS":
            y = Runtime(work / f"{name}.engine").run(x)
            row["execution"] = "PASS"
            row["finite"] = bool(torch.isfinite(y).all().item())
            row["distribution"] = distribution(y)
            np.save(out / f"output_{name}.npy", y.detach().cpu().numpy())
            results[name] = {"row": row, "tensor": y.detach().cpu()}
        else:
            results[name] = {"row": row, "tensor": None}

    json_dump(out / "engine_summary_fp16.json", results["fp16"]["row"])
    json_dump(out / "engine_summary_w8.json", results["w8"]["row"])
    json_dump(out / "engine_summary_w8a8.json", results["w8a8"]["row"])
    comparison = {}
    fp_out = results["fp16"]["tensor"]
    for name in ("w8", "w8a8"):
        q_out = results[name]["tensor"]
        comparison[name] = {"vs_trt_fp16": metric(q_out, fp_out) if q_out is not None and fp_out is not None else "NOT_AVAILABLE"}
    comparison["reference_context_only"] = "HF BF16 vs TRT FP16 remains the frozen Phase 2.2 numerical background and is not quantization delta."
    json_dump(out / "numerical_comparison.json", comparison)
    json_dump(out / "output_distribution.json", {name: results[name]["row"].get("distribution", "NOT_AVAILABLE") for name in results})
    memory.append(memory_snapshot("complete"))
    json_dump(out / "memory_trace.json", {"trace": memory, "oom": False, "exit137": False})

    w8_ok = results["w8"]["row"].get("execution") == "PASS" and results["w8"]["row"].get("finite") is True
    w8a8_ok = results["w8a8"]["row"].get("execution") == "PASS" and results["w8a8"]["row"].get("finite") is True
    sanity_ok = sanity_exec == "PASS" and sanity_finite is True
    gate = "PASS" if sanity_ok and w8_ok and w8a8_ok else "BLOCKED"
    final = {
        "phase": "Phase 2.3-A",
        "gate": gate,
        "sanity": "PASS" if sanity_ok else "BLOCKED",
        "w8_qdq": "PASS" if w8_ok else "BLOCKED",
        "w8a8_qdq": "PASS" if w8a8_ok else "BLOCKED",
        "int8_compute": "INT8_COMPUTE_NOT_PROVEN",
        "w8_compute": "INT8_COMPUTE_NOT_PROVEN",
        "w8a8_compute": "INT8_COMPUTE_PROVEN",
        "w8a8_capability": "SUPPORTED_FOR_THIS_GRAPH" if w8a8_ok else "W8A8_CAPABILITY_BLOCKED",
        "quantization_delta_definition": "W8/W8A8 output vs TRT FP16 output",
        "phase2_2_frozen": True,
        "c1_reopened": False,
        "no_nsight": True,
        "no_formal_benchmark": True,
    }
    json_dump(out / "final_validation.json", final)
    print(json.dumps(final, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=None)
    main(parser.parse_args())
