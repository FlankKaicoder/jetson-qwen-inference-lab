"""Authorized C1G Layer 0 internal TensorRT probe.

This script builds a new diagnostic engine for one Qwen3 Layer 0 only.  The
existing B4.2 engine is loaded read-only for the probe-validity comparison and
is never rebuilt or overwritten.  ``--group`` selects the smallest probe
surface (A, B, or C).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
B3 = HERE.parent / "phase2_2b3_real_stack"
import sys
sys.path.insert(0, str(B3))
from portable_qwen3_stack import PortableQwen3Layer


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    x, y = a.detach().float().cpu(), b.detach().float().cpu()
    d = x - y
    xn, yn = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "shape": list(x.shape), "candidate_shape": list(y.shape),
        "shape_equal": list(x.shape) == list(y.shape),
        "dtype": str(a.dtype), "candidate_dtype": str(b.dtype),
        "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
        "max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean()),
        "rmse": float(torch.sqrt((d * d).mean())),
        "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(xn, min=tiny)),
        "cosine": float(torch.dot(x.reshape(-1), y.reshape(-1)) / torch.clamp(xn * yn, min=tiny)),
    }


def mem_snapshot(stage: str) -> dict:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) * 1024
    free, total = torch.cuda.mem_get_info()
    return {
        "stage": stage, "mem_available_bytes": info.get("MemAvailable"),
        "vmrss": next((x.split(":", 1)[1].strip() for x in Path("/proc/self/status").read_text().splitlines() if x.startswith("VmRSS:")), None),
        "cuda_free_bytes": free, "cuda_total_bytes": total,
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def reference_components(layer: PortableQwen3Layer, hidden: torch.Tensor, pos: torch.Tensor) -> dict[str, torch.Tensor]:
    residual = hidden
    input_rms = layer.input_layernorm(hidden)
    b, s, _ = input_rms.shape
    q_proj = layer.self_attn.q_proj(input_rms).view(b, s, layer.q_heads, layer.head_dim)
    k_proj = layer.self_attn.k_proj(input_rms).view(b, s, layer.kv_heads, layer.head_dim)
    v_proj = layer.self_attn.v_proj(input_rms).view(b, s, layer.kv_heads, layer.head_dim)
    q_norm = layer.self_attn.q_norm(q_proj).transpose(1, 2)
    k_norm = layer.self_attn.k_norm(k_proj).transpose(1, 2)
    v_heads = v_proj.transpose(1, 2)
    q_rope = layer._rope(q_norm, pos)
    k_rope = layer._rope(k_norm, pos)
    ka = k_rope[:, :, None].expand(b, layer.kv_heads, layer.groups, s, layer.head_dim).reshape(b, layer.q_heads, s, layer.head_dim)
    va = v_heads[:, :, None].expand(b, layer.kv_heads, layer.groups, s, layer.head_dim).reshape(b, layer.q_heads, s, layer.head_dim)
    scores = torch.matmul(q_rope, ka.transpose(-2, -1)) * (layer.head_dim ** -0.5)
    mask = torch.triu(torch.ones((s, s), device=hidden.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_rope.dtype)
    context = torch.matmul(probs, va).transpose(1, 2).contiguous().reshape(b, s, -1)
    o_proj = layer.self_attn.o_proj(context)
    attn_residual = residual + o_proj
    post_rms = layer.post_attention_layernorm(attn_residual)
    gate = layer.mlp.gate_proj(post_rms)
    up = layer.mlp.up_proj(post_rms)
    silu = torch.nn.functional.silu(gate)
    gate_up = silu * up
    down = layer.mlp.down_proj(gate_up)
    final = attn_residual + down
    return {
        "input_hidden": hidden, "input_rmsnorm": input_rms,
        "q_projection": q_proj, "k_projection": k_proj, "v_projection": v_proj,
        "q_norm": q_norm, "k_norm": k_norm, "q_rope": q_rope, "k_rope": k_rope,
        "attention_scores": scores, "softmax": probs, "attention_context": context,
        "output_projection": o_proj, "attention_residual": attn_residual,
        "post_attention_rmsnorm": post_rms, "gate_projection": gate,
        "up_projection": up, "silu_gate": silu, "gate_up": gate_up,
        "down_projection": down, "final_hidden": final,
    }


GROUP_OUTPUTS = {
    "A": ["input_hidden", "input_rmsnorm", "q_projection", "k_projection", "v_projection",
          "q_norm", "k_norm", "q_rope", "k_rope", "final_hidden"],
    "B": ["attention_scores", "softmax", "attention_context", "output_projection",
          "attention_residual", "final_hidden"],
    "C": ["post_attention_rmsnorm", "gate_projection", "up_projection", "silu_gate",
          "gate_up", "down_projection", "final_hidden"],
}


class DiagnosticLayer(torch.nn.Module):
    def __init__(self, layer: PortableQwen3Layer, group: str):
        super().__init__()
        self.layer = layer
        self.group = group

    def forward(self, hidden, pos):
        vals = reference_components(self.layer, hidden, pos)
        return tuple(vals[name] for name in GROUP_OUTPUTS[self.group])


class TRT:
    def __init__(self, path: Path):
        import tensorrt as trt
        self.trt = trt
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"ENGINE_DESERIALIZE_FAILED:{path}")
        self.stream = torch.cuda.current_stream()

    def contract(self) -> list[dict]:
        return [{"name": self.engine.get_tensor_name(i),
                 "mode": str(self.engine.get_tensor_mode(self.engine.get_tensor_name(i))),
                 "dtype": str(self.engine.get_tensor_dtype(self.engine.get_tensor_name(i))),
                 "shape": list(self.engine.get_tensor_shape(self.engine.get_tensor_name(i)))}
                for i in range(self.engine.num_io_tensors)]

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ctx = self.engine.create_execution_context()
        for name, tensor in inputs.items():
            if not ctx.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"SHAPE_REJECTED:{name}")
            ctx.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != self.trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(ctx.get_tensor_shape(name))
            dtype = torch.float16 if self.engine.get_tensor_dtype(name) == self.trt.DataType.HALF else torch.float32
            outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)
            ctx.set_tensor_address(name, outputs[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("EXECUTE_ASYNC_V3_FAILED")
        self.stream.synchronize()
        return outputs


def load_layer(path: Path) -> PortableQwen3Layer:
    layer = PortableQwen3Layer().to(device="cuda", dtype=torch.float16).eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    layer.load_state_dict({k: v.half() for k, v in state.items()}, strict=True)
    del state
    return layer


def export_onnx(layer, group: str, path: Path, hidden, pos) -> dict:
    import onnx
    module = DiagnosticLayer(layer, group).eval()
    with torch.inference_mode():
        torch.onnx.export(module, (hidden, pos), str(path), opset_version=17,
                          input_names=["hidden_states", "position_ids"],
                          output_names=GROUP_OUTPUTS[group],
                          dynamic_axes={"hidden_states": {0: "batch", 1: "seq"},
                                        "position_ids": {0: "batch", 1: "seq"},
                                        **{n: {0: "batch", 1: "seq"} for n in GROUP_OUTPUTS[group]}})
    model = onnx.load(str(path)); onnx.checker.check_model(model)
    return {"status": "PASS", "bytes": path.stat().st_size,
            "node_count": len(model.graph.node), "initializer_count": len(model.graph.initializer),
            "operators": dict(Counter(x.op_type for x in model.graph.node)),
            "inputs": [x.name for x in model.graph.input],
            "outputs": [x.name for x in model.graph.output]}


def build_engine(onnx_path: Path, engine_path: Path) -> dict:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    network = trt.Builder(logger).create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse(onnx_path.read_bytes())
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not ok:
        raise RuntimeError(f"PARSER_FAILED:{errors}")
    builder = trt.Builder(logger); cfg = builder.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    prof.set_shape("hidden_states", (1, 1, 1024), (1, 8, 1024), (1, 16, 1024))
    prof.set_shape("position_ids", (1, 1), (1, 8), (1, 16))
    cfg.add_optimization_profile(prof)
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        raise RuntimeError("BUILD_FAILED")
    engine_path.write_bytes(bytes(blob))
    return {"status": "PASS", "bytes": engine_path.stat().st_size, "parser_errors": errors}


def main(a: argparse.Namespace) -> None:
    a.out.mkdir(parents=True, exist_ok=True); a.tmp.mkdir(parents=True, exist_ok=True)
    trace = [mem_snapshot("start")]
    payload = torch.load(a.embedding_reference, map_location="cpu", weights_only=True)
    hidden = payload["fp16"]["prefill"].contiguous().cuda()
    del payload
    pos = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
    layer = load_layer(a.layer_file); trace.append(mem_snapshot("layer_loaded"))
    with torch.inference_mode():
        ref = reference_components(layer, hidden, pos)
    trace.append(mem_snapshot("reference_complete"))

    # Existing B4.2 current control and original Layer 0 boundary.
    b4 = TRT(a.b4_engine)
    b4_contract = b4.contract()
    b4_out = b4.run({"hidden_states": hidden, "position_ids": pos})
    if "hidden_l0" not in b4_out:
        raise RuntimeError("B4_HIDDEN_L0_MISSING")
    b4_layer0 = b4_out["hidden_l0"]
    b4_control = metric(ref["final_hidden"], b4_layer0)
    trace.append(mem_snapshot("b4_control_complete"))

    onnx_path = a.tmp / f"layer0_group_{a.group}.onnx"
    engine_path = a.tmp / f"layer0_group_{a.group}.engine"
    onnx_summary = export_onnx(layer, a.group, onnx_path, hidden, pos)
    trace.append(mem_snapshot("onnx_export_complete"))
    engine_summary = build_engine(onnx_path, engine_path)
    trace.append(mem_snapshot("trt_build_complete"))

    diag = TRT(engine_path)
    diag_out = diag.run({"hidden_states": hidden, "position_ids": pos})
    diag_final = diag_out["final_hidden"]
    validity = metric(b4_layer0, diag_final)
    validity["criterion"] = {"relative_l2_max": 0.01, "cosine_min": 0.9999}
    validity["status"] = "PASS" if validity["relative_l2"] <= 0.01 and validity["cosine"] >= 0.9999 else "FAIL"
    trace.append(mem_snapshot("diagnostic_complete"))

    rows = []
    for name in GROUP_OUTPUTS[a.group]:
        rows.append({"probe": name, "component": name, "metric": metric(ref[name], diag_out[name])})
    # Keep final hidden validity separate from attribution rows.
    result = {
        "experiment": "Phase 2.2-C1G", "group": a.group,
        "objective": "Qwen3 Layer 0 internal tensor probe",
        "input_contract": {"shape": list(hidden.shape), "dtype": str(hidden.dtype),
                            "position_ids": [[0,1,2,3,4,5,6,7]], "same_canonical_embedding": True},
        "layer0_structure": {"hidden": 1024, "q_heads": 16, "kv_heads": 8, "head_dim": 128,
                              "rotary_dimension": 128, "rope_theta": 1000000.0, "gqa_repeat": 2,
                              "rmsnorm_eps": 1e-6, "mlp_intermediate": 3072},
        "b4_2_contract": {"input_count": sum(x["mode"].endswith("INPUT") for x in b4_contract),
                          "output_count": sum(x["mode"].endswith("OUTPUT") for x in b4_contract),
                          "outputs": [x["name"] for x in b4_contract if x["mode"].endswith("OUTPUT")]},
        "b4_2_control": b4_control, "probe_validity": validity,
        "portable_reference_checkpoints": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                                 "finite": bool(torch.isfinite(v).all())}
                                             for k, v in ref.items() if k in GROUP_OUTPUTS[a.group]},
        "component_metrics": rows,
        "first_nonzero_difference": None,
        "first_material_divergence": None,
        "diagnostic_engine": {"onnx": str(onnx_path), "engine": str(engine_path),
                               "onnx_summary": onnx_summary, "engine_summary": engine_summary},
        "memory": {"trace": trace, "scope": "DIAGNOSTIC_ONLY", "oom": False, "exit137": False},
    }
    # Compute first non-zero in probe order and largest incremental error.
    nonzero = [r for r in rows if r["metric"]["max_abs"] != 0.0]
    if nonzero:
        result["first_nonzero_difference"] = nonzero[0]["component"]
    increments = []
    previous = 0.0
    for row in rows:
        val = row["metric"]["relative_l2"]
        increments.append({"component": row["component"], "relative_l2": val, "increment": val - previous})
        previous = val
    result["error_growth"] = increments
    result["status"] = "INSTRUMENTATION_PERTURBS_TRT_PATH" if validity["status"] == "FAIL" else "PROBE_VALIDITY_PASS"
    (a.out / f"c1g_layer0_group_{a.group.lower()}_{a.timestamp}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "group": a.group,
                      "probe_validity": validity, "output": str(a.out)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["A", "B", "C"], required=True)
    p.add_argument("--b4-engine", type=Path, required=True)
    p.add_argument("--embedding-reference", type=Path, required=True)
    p.add_argument("--layer-file", type=Path, required=True)
    p.add_argument("--tmp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timestamp", default="20260902T200000Z")
    main(p.parse_args())
