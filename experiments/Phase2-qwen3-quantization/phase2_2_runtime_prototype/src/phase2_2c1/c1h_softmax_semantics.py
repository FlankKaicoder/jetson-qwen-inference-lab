"""C1H: isolate Qwen3 Layer 0 attention softmax numerical semantics.

This diagnostic is intentionally independent of B4.2.  It builds only fresh
micro/probe/Layer-0 engines in a timestamped temporary directory and loads
B4.2 read-only for the unchanged control comparison.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
B3 = HERE.parent / "phase2_2b3_real_stack"
sys.path.insert(0, str(B3))
from portable_qwen3_stack import PortableQwen3Layer


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    x, y = a.detach().float().cpu(), b.detach().float().cpu()
    d = x - y
    xn, yn = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "shape": list(x.shape), "candidate_shape": list(y.shape),
        "shape_equal": list(x.shape) == list(y.shape), "dtype": str(a.dtype),
        "candidate_dtype": str(b.dtype), "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
        "max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean()),
        "rmse": float(torch.sqrt((d * d).mean())),
        "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(xn, min=tiny)),
        "cosine": float(torch.dot(x.reshape(-1), y.reshape(-1)) / torch.clamp(xn * yn, min=tiny)),
        "sha256": sha(a), "candidate_sha256": sha(b),
    }


def snapshot(stage: str) -> dict:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1); info[k] = int(v.split()[0]) * 1024
    free, total = torch.cuda.mem_get_info()
    return {"stage": stage, "mem_available_bytes": info.get("MemAvailable"),
            "vmrss": next((x.split(":", 1)[1].strip() for x in Path("/proc/self/status").read_text().splitlines() if x.startswith("VmRSS:")), None),
            "cuda_free_bytes": free, "cuda_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def components(layer: PortableQwen3Layer, hidden: torch.Tensor, pos: torch.Tensor, softmax_mode: str = "portable") -> dict[str, torch.Tensor]:
    residual = hidden
    rms = layer.input_layernorm(hidden)
    b, s, _ = rms.shape
    q = layer.self_attn.q_proj(rms).view(b, s, layer.q_heads, layer.head_dim)
    k = layer.self_attn.k_proj(rms).view(b, s, layer.kv_heads, layer.head_dim)
    v = layer.self_attn.v_proj(rms).view(b, s, layer.kv_heads, layer.head_dim)
    qn = layer.self_attn.q_norm(q).transpose(1, 2)
    kn = layer.self_attn.k_norm(k).transpose(1, 2)
    vh = v.transpose(1, 2)
    qr, kr = layer._rope(qn, pos), layer._rope(kn, pos)
    ka = kr[:, :, None].expand(b, layer.kv_heads, layer.groups, s, layer.head_dim).reshape(b, layer.q_heads, s, layer.head_dim)
    va = vh[:, :, None].expand(b, layer.kv_heads, layer.groups, s, layer.head_dim).reshape(b, layer.q_heads, s, layer.head_dim)
    raw = torch.matmul(qr, ka.transpose(-2, -1))
    scaled = raw * (layer.head_dim ** -0.5)
    mask = torch.triu(torch.ones((s, s), device=hidden.device, dtype=torch.bool), diagonal=1)
    masked = scaled.masked_fill(mask, torch.finfo(scaled.dtype).min)
    if softmax_mode == "fp32":
        probs = torch.softmax(masked.float(), dim=-1).to(qr.dtype)
    else:
        probs = torch.softmax(masked, dim=-1, dtype=torch.float32).to(qr.dtype)
    ctx = torch.matmul(probs, va).transpose(1, 2).contiguous().reshape(b, s, -1)
    op = layer.self_attn.o_proj(ctx)
    attn_res = residual + op
    post = layer.post_attention_layernorm(attn_res)
    gate, up = layer.mlp.gate_proj(post), layer.mlp.up_proj(post)
    silu, gate_up = torch.nn.functional.silu(gate), torch.nn.functional.silu(gate) * up
    down = layer.mlp.down_proj(gate_up)
    return {"input_rmsnorm": rms, "q_projection": q, "k_projection": k, "v_projection": v,
            "q_norm": qn, "k_norm": kn, "q_rope": qr, "k_rope": kr,
            "qk_raw": raw, "qk_scaled": scaled, "attention_scores": masked,
            "softmax": probs, "attention_context": ctx, "output_projection": op,
            "attention_residual": attn_res, "post_attention_rmsnorm": post,
            "gate_projection": gate, "up_projection": up, "silu_gate": silu,
            "gate_up": gate_up, "down_projection": down, "final_hidden": attn_res + down}


class Probe(torch.nn.Module):
    names = ["qk_raw", "qk_scaled", "attention_scores", "softmax", "attention_context", "output_projection", "attention_residual", "final_hidden"]
    def __init__(self, layer, mode="portable"):
        super().__init__(); self.layer = layer; self.mode = mode
    def forward(self, hidden, pos):
        v = components(self.layer, hidden, pos, self.mode)
        return tuple(v[n] for n in self.names)


class LayerOut(torch.nn.Module):
    def __init__(self, layer, mode): super().__init__(); self.layer = layer; self.mode = mode
    def forward(self, hidden, pos): return components(self.layer, hidden, pos, self.mode)["final_hidden"]


class Micro(torch.nn.Module):
    def __init__(self, mode): super().__init__(); self.mode = mode
    def forward(self, x):
        if self.mode == "fp32": return torch.softmax(x.float(), dim=-1).to(x.dtype)
        return torch.softmax(x, dim=-1)


class TRT:
    def __init__(self, path: Path):
        import tensorrt as trt
        self.trt = trt; self.path = path
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None: raise RuntimeError(f"ENGINE_DESERIALIZE_FAILED:{path}")
        self.stream = torch.cuda.current_stream()
    def run(self, inputs):
        ctx = self.engine.create_execution_context(); out = {}
        for name, t in inputs.items():
            if not ctx.set_input_shape(name, tuple(t.shape)): raise RuntimeError(f"INPUT_SHAPE_REJECTED:{name}")
            ctx.set_tensor_address(name, t.data_ptr())
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != self.trt.TensorIOMode.OUTPUT: continue
            shape = tuple(ctx.get_tensor_shape(name)); dt = self.engine.get_tensor_dtype(name)
            td = torch.float16 if dt == self.trt.DataType.HALF else torch.float32
            out[name] = torch.empty(shape, device="cuda", dtype=td); ctx.set_tensor_address(name, out[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream): raise RuntimeError("EXECUTE_FAILED")
        self.stream.synchronize(); return out


def export_onnx(module, path, hidden, pos, names, micro=False):
    import onnx
    if micro:
        axes = {"scores": {0: "batch", 2: "query", 3: "key"}}
        args = (hidden,); inputs = ["scores"]
    else:
        axes = {"hidden_states": {0: "batch", 1: "seq"}, "position_ids": {0: "batch", 1: "seq"}}
        args = (hidden, pos); inputs = ["hidden_states", "position_ids"]
    for n, rank in names:
        axes[n] = {0: "batch"}
        if rank > 1: axes[n][1] = "heads_or_seq"
    torch.onnx.export(module, args, str(path), opset_version=17,
                      input_names=inputs, output_names=[n for n, _ in names], dynamic_axes=axes)
    m = onnx.load(str(path)); onnx.checker.check_model(m)
    return {"status": "PASS", "bytes": path.stat().st_size, "nodes": len(m.graph.node)}


def build(onnx_path: Path, engine_path: Path, micro=False):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("PARSER_FAILED:" + " | ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
    prof = builder.create_optimization_profile()
    if micro: prof.set_shape("scores", (1, 16, 1, 1), (1, 16, 8, 8), (1, 16, 16, 16))
    else:
        prof.set_shape("hidden_states", (1, 1, 1024), (1, 8, 1024), (1, 16, 1024)); prof.set_shape("position_ids", (1, 1), (1, 8), (1, 16))
    cfg.add_optimization_profile(prof); blob = builder.build_serialized_network(network, cfg)
    if blob is None: raise RuntimeError("BUILD_FAILED")
    engine_path.write_bytes(bytes(blob)); return {"status": "PASS", "bytes": engine_path.stat().st_size}


def load_layer(path):
    layer = PortableQwen3Layer().to(device="cuda", dtype=torch.float16).eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    layer.load_state_dict({k: v.half() for k, v in state.items()}, strict=True); del state
    return layer


def main(a):
    a.tmp.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=True); trace = [snapshot("start")]
    payload = torch.load(a.embedding_reference, map_location="cpu", weights_only=True)
    hidden = payload["fp16"]["prefill"].contiguous().cuda(); del payload
    pos = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
    layer = load_layer(a.layer_file).eval(); ref = components(layer, hidden, pos); trace.append(snapshot("reference_complete"))
    control = TRT(a.b4_engine).run({"hidden_states": hidden, "position_ids": pos})["hidden_l0"]
    control_metric = metric(ref["final_hidden"], control); trace.append(snapshot("b4_control_complete"))

    # Fresh internal probe exposes the exact pre-softmax tensor and all casts.
    probe_names = [(n, 4 if n in ("qk_raw", "qk_scaled", "attention_scores", "softmax") else 3) for n in Probe.names]
    probe_onnx, probe_engine = a.tmp / "c1h_probe.onnx", a.tmp / "c1h_probe.engine"
    ps = export_onnx(Probe(layer, "portable").eval(), probe_onnx, hidden, pos, probe_names); trace.append(snapshot("probe_export"))
    bs = build(probe_onnx, probe_engine); trace.append(snapshot("probe_build"))
    pout = TRT(probe_engine).run({"hidden_states": hidden, "position_ids": pos})
    probe_metrics = {n: metric(ref[n], pout[n]) for n, _ in probe_names}
    pre_summary = {}
    for n in ("qk_raw", "qk_scaled", "attention_scores"):
        t = pout[n].float(); masked = ~torch.isfinite(t)
        pre_summary[n] = {"shape": list(pout[n].shape), "dtype": str(pout[n].dtype), "finite_count": int(torch.isfinite(t).sum()), "inf_count": int(masked.sum()), "min": float(t.min()), "max": float(t.max()), "sha256": sha(pout[n]), "portable_sha256": sha(ref[n])}

    # Same-input micro isolation: portable/current, native TRT, FP32 TRT and oracle.
    scores = ref["attention_scores"].contiguous(); oracle = torch.softmax(scores.float(), dim=-1)
    micro_rows = {}
    for mode in ("native", "fp32"):
        mod = Micro("fp32" if mode == "fp32" else "native").eval(); onnx_path = a.tmp / f"micro_{mode}.onnx"; eng_path = a.tmp / f"micro_{mode}.engine"
        ms = export_onnx(mod, onnx_path, scores, pos, [("softmax", 4)], micro=True); bs2 = build(onnx_path, eng_path, micro=True)
        got = TRT(eng_path).run({"scores": scores})["softmax"]
        micro_rows[mode] = {"onnx": ms, "engine": bs2, "vs_portable": metric(ref["softmax"], got), "vs_oracle": metric(oracle, got), "portable_vs_oracle": metric(oracle, ref["softmax"])}
        del got, mod; gc.collect(); torch.cuda.empty_cache()

    # Independent Layer-0 single-variable FP32-softmax A/B.
    layer_onnx, layer_engine = a.tmp / "layer0_fp32softmax.onnx", a.tmp / "layer0_fp32softmax.engine"
    ls = export_onnx(LayerOut(layer, "fp32").eval(), layer_onnx, hidden, pos, [("final_hidden", 3)]); trace.append(snapshot("layer_variant_export"))
    lbs = build(layer_onnx, layer_engine); trace.append(snapshot("layer_variant_build"))
    variant = TRT(layer_engine).run({"hidden_states": hidden, "position_ids": pos})["final_hidden"]
    layer_ab = {"native_b4_control_vs_reference": control_metric, "fp32_variant_vs_reference": metric(ref["final_hidden"], variant), "native_b4_vs_fp32_variant": metric(control, variant)}
    trace.append(snapshot("layer_variant_complete"))
    native_jump = micro_rows["native"]["vs_oracle"]["relative_l2"]
    fp32_jump = micro_rows["fp32"]["vs_oracle"]["relative_l2"]
    improved_micro = fp32_jump < native_jump * 0.5
    improved_layer = layer_ab["fp32_variant_vs_reference"]["relative_l2"] < control_metric["relative_l2"] * 0.5
    pre_near = probe_metrics["attention_scores"]["relative_l2"] < 1e-4
    if not pre_near: decision = "PRE_SOFTMAX_TRANSFORM_DIVERGENCE_FOUND"
    elif improved_micro and improved_layer: decision = "SOFTMAX_ROOT_CAUSE_CONFIRMED"
    elif improved_micro or improved_layer: decision = "SOFTMAX_MAJOR_SOURCE_CONFIRMED"
    else: decision = "SOFTMAX_HYPOTHESIS_REJECTED"
    result = {"experiment": "Phase 2.2-C1H", "decision": decision, "c1_status": "BLOCKED", "input_contract": {"shape": list(hidden.shape), "dtype": str(hidden.dtype), "position_ids": list(range(8))}, "layer_structure": {"hidden": 1024, "q_heads": 16, "kv_heads": 8, "head_dim": 128, "gqa_repeat": 2, "rotary_dim": 128, "rope_theta": 1000000.0, "rmsnorm_eps": 1e-6}, "b4_2_control": control_metric, "pre_softmax_summary": pre_summary, "probe_metrics": probe_metrics, "micro_isolation": micro_rows, "layer0_ab": layer_ab, "decision_tests": {"pre_inputs_near_identical": pre_near, "micro_fp32_materially_better": improved_micro, "layer0_fp32_materially_better": improved_layer}, "memory": {"trace": trace, "oom": False, "exit137": False}}
    path = a.out / f"c1h_softmax_semantics_{a.timestamp}.json"; path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": decision, "artifact": str(path), "b4_relative_l2": control_metric["relative_l2"], "native_micro_rel_l2": native_jump, "fp32_micro_rel_l2": fp32_jump, "fp32_layer_rel_l2": layer_ab["fp32_variant_vs_reference"]["relative_l2"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--b4-engine", type=Path, required=True); p.add_argument("--embedding-reference", type=Path, required=True); p.add_argument("--layer-file", type=Path, required=True); p.add_argument("--tmp", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--timestamp", required=True); main(p.parse_args())
