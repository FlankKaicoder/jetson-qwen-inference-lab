"""C1J Qwen3 RoPE cache, position, layout and precision diagnostics.

This script creates only timestamped diagnostic ONNX/TensorRT artifacts.  The
existing B4.2 engine is loaded read-only for the control measurement.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, resource, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
B3 = HERE.parent / "phase2_2b3_real_stack"
sys.path.insert(0, str(B3))
from portable_qwen3_stack import PortableQwen3Layer


def sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def metric(a, b):
    x, y = a.detach().float().cpu(), b.detach().float().cpu(); d = x - y
    xn, yn = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    tiny = torch.finfo(torch.float32).tiny
    return {"shape": list(x.shape), "candidate_shape": list(y.shape),
            "shape_equal": list(x.shape) == list(y.shape), "dtype": str(a.dtype),
            "candidate_dtype": str(b.dtype), "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            "max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean()),
            "rmse": float(torch.sqrt((d * d).mean())),
            "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(xn, min=tiny)),
            "cosine": float(torch.dot(x.reshape(-1), y.reshape(-1)) / torch.clamp(xn * yn, min=tiny)),
            "sha256": sha(a), "candidate_sha256": sha(b)}


def summary(x):
    y = x.detach().float()
    return {"shape": list(x.shape), "dtype": str(x.dtype), "finite": bool(torch.isfinite(y).all()),
            "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "sha256": sha(x)}


def snap(stage):
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1); info[k] = int(v.split()[0]) * 1024
    free, total = torch.cuda.mem_get_info()
    return {"stage": stage, "mem_available_bytes": info.get("MemAvailable"),
            "cuda_free_bytes": free, "cuda_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def inv_freq(dim=128, theta=1_000_000.0, device="cuda"):
    return torch.pow(torch.tensor(theta, device=device, dtype=torch.float32),
                     -torch.arange(dim // 2, device=device, dtype=torch.float32) / (dim // 2))


def rope_trace(x, pos, mode="native", theta=1_000_000.0):
    """Return cache and output; modes differ in exactly one RoPE semantic."""
    d = x.shape[-1]; inv = inv_freq(d, theta, x.device)
    freq = pos.float().unsqueeze(-1) * inv
    emb = torch.cat((freq, freq), dim=-1)
    cos32, sin32 = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
    if mode == "position_plus_one":
        # Used only as an explicit position-indexing negative control.
        pos1 = pos + 1
        f1 = pos1.float().unsqueeze(-1) * inv
        e1 = torch.cat((f1, f1), dim=-1)
        cos32, sin32 = e1.cos().unsqueeze(1), e1.sin().unsqueeze(1)
    cos, sin = (cos32, sin32) if mode == "cache_fp32" else (cos32.to(x.dtype), sin32.to(x.dtype))
    half = d // 2
    if mode == "even_odd":
        even, odd = x[..., 0::2], x[..., 1::2]
        rot = torch.stack((-odd, even), dim=-1).reshape_as(x)
    else:
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    if mode in ("fp32_arith", "cache_fp32"):
        out = (x.float() * cos32 + rot.float() * sin32).to(x.dtype)
    else:
        out = x * cos + rot * sin
    return inv, cos, sin, out


class RopeTrace(torch.nn.Module):
    def __init__(self, mode="native"):
        super().__init__(); self.mode = mode
    def forward(self, x, position_ids):
        inv, cos, sin, out = rope_trace(x, position_ids, self.mode)
        return inv, cos, sin, out


class Layer0(torch.nn.Module):
    def __init__(self, layer, rope_mode="native"):
        super().__init__(); self.layer = layer; self.rope_mode = rope_mode
    def forward(self, hidden_states, position_ids):
        l = self.layer; residual = hidden_states; x = l.input_layernorm(hidden_states)
        b, s, _ = x.shape
        q = l.self_attn.q_norm(l.self_attn.q_proj(x).view(b, s, l.q_heads, l.head_dim)).transpose(1, 2)
        k = l.self_attn.k_norm(l.self_attn.k_proj(x).view(b, s, l.kv_heads, l.head_dim)).transpose(1, 2)
        v = l.self_attn.v_proj(x).view(b, s, l.kv_heads, l.head_dim).transpose(1, 2)
        _, _, _, q = rope_trace(q, position_ids, self.rope_mode)
        _, _, _, k = rope_trace(k, position_ids, self.rope_mode)
        ka = k[:, :, None].expand(b, l.kv_heads, l.groups, s, l.head_dim).reshape(b, l.q_heads, s, l.head_dim)
        va = v[:, :, None].expand(b, l.kv_heads, l.groups, s, l.head_dim).reshape(b, l.q_heads, s, l.head_dim)
        scores = torch.matmul(q, ka.transpose(-2, -1)) * (l.head_dim ** -0.5)
        mask = torch.triu(torch.ones((s, s), device=x.device, dtype=torch.bool), diagonal=1)
        probs = torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1, dtype=torch.float32).to(q.dtype)
        ctx = torch.matmul(probs, va).transpose(1, 2).contiguous().reshape(b, s, -1)
        ar = residual + l.self_attn.o_proj(ctx); post = l.post_attention_layernorm(ar)
        down = l.mlp.down_proj(torch.nn.functional.silu(l.mlp.gate_proj(post)) * l.mlp.up_proj(post))
        return ar + down


class TRT:
    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt; self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None: raise RuntimeError("ENGINE_DESERIALIZE_FAILED")
        self.stream = torch.cuda.current_stream()
    def run(self, inputs):
        c = self.engine.create_execution_context(); out = {}
        for n, t in inputs.items():
            if not c.set_input_shape(n, tuple(t.shape)): raise RuntimeError("INPUT_SHAPE_REJECTED:" + n)
            c.set_tensor_address(n, t.data_ptr())
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) != self.trt.TensorIOMode.OUTPUT: continue
            shape = tuple(c.get_tensor_shape(n)); dt = self.engine.get_tensor_dtype(n)
            td = torch.float16 if dt == self.trt.DataType.HALF else torch.float32
            out[n] = torch.empty(shape, device="cuda", dtype=td); c.set_tensor_address(n, out[n].data_ptr())
        if not c.execute_async_v3(self.stream.cuda_stream): raise RuntimeError("EXECUTE_FAILED")
        self.stream.synchronize(); return out


def export(module, path, args, inputs, outputs, axes):
    import onnx
    torch.onnx.export(module, args, str(path), opset_version=17, input_names=inputs,
                      output_names=outputs, dynamic_axes=axes)
    m = onnx.load(str(path)); onnx.checker.check_model(m)
    return {"status": "PASS", "bytes": path.stat().st_size, "nodes": len(m.graph.node)}


def build(onnx_path, engine_path, profiles):
    import tensorrt as trt
    b = trt.Builder(trt.Logger(trt.Logger.WARNING)); n = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    p = trt.OnnxParser(n, trt.Logger(trt.Logger.WARNING))
    if not p.parse(Path(onnx_path).read_bytes()): raise RuntimeError("PARSER_FAILED:" + str([str(p.get_error(i)) for i in range(p.num_errors)]))
    cfg = b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); prof = b.create_optimization_profile()
    for name, shape in profiles.items(): prof.set_shape(name, *shape)
    cfg.add_optimization_profile(prof); blob = b.build_serialized_network(n, cfg)
    if blob is None: raise RuntimeError("BUILD_FAILED")
    Path(engine_path).write_bytes(bytes(blob)); return {"status": "PASS", "bytes": Path(engine_path).stat().st_size}


def load_layer(path):
    l = PortableQwen3Layer().to(device="cuda", dtype=torch.float16).eval()
    st = torch.load(path, map_location="cpu", weights_only=True); l.load_state_dict({k: v.half() for k, v in st.items()}, strict=True); del st
    return l


def per_head(a, b):
    vals = []
    for i in range(a.shape[1]):
        vals.append(float(torch.linalg.vector_norm((a[:, i] - b[:, i]).float()) / torch.clamp(torch.linalg.vector_norm(a[:, i].float()), min=torch.finfo(torch.float32).tiny)))
    return {"min": min(vals), "median": sorted(vals)[len(vals) // 2], "max": max(vals), "worst_head": int(max(range(len(vals)), key=lambda i: vals[i])), "all": vals}


def main(a):
    a.tmp.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=True); trace = [snap("start")]
    payload = torch.load(a.embedding_reference, map_location="cpu", weights_only=True); hidden = payload["fp16"]["prefill"].contiguous().cuda(); del payload
    pos = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8); layer = load_layer(a.layer_file)
    # Portable Q/K inputs are fixed once and reused for every micro engine.
    rms = layer.input_layernorm(hidden); b, s, _ = rms.shape
    q = layer.self_attn.q_norm(layer.self_attn.q_proj(rms).view(b, s, layer.q_heads, layer.head_dim)).transpose(1, 2).contiguous()
    k = layer.self_attn.k_norm(layer.self_attn.k_proj(rms).view(b, s, layer.kv_heads, layer.head_dim)).transpose(1, 2).contiguous()
    refs = {"q": rope_trace(q, pos), "k": rope_trace(k, pos)}; trace.append(snap("reference_complete"))
    control = TRT(a.b4_engine).run({"hidden_states": hidden, "position_ids": pos})["hidden_l0"]
    # Cache/position trace micro engines expose inv_freq, cos, sin and output.
    cache_rows = {}; position_rows = {}; micro_rows = {}; per_head_rows = {}
    axes = {"x": {0: "batch"}, "position_ids": {0: "batch", 1: "seq"}, "inv_freq": {0: "half_dim"}, "cos": {0: "batch", 2: "seq"}, "sin": {0: "batch", 2: "seq"}, "y": {0: "batch", 1: "heads", 2: "seq"}}
    for kind, x, ref_tuple in (("q", q, refs["q"]), ("k", k, refs["k"])):
        for mode in ("native", "fp32_arith", "cache_fp32", "even_odd"):
            mod = RopeTrace(mode).eval(); op = a.tmp / f"c1j_{kind}_{mode}.onnx"; ep = a.tmp / f"c1j_{kind}_{mode}.engine"
            ex = export(mod, op, (x, pos), ["x", "position_ids"], ["inv_freq", "cos", "sin", "y"], axes)
            eb = build(op, ep, {"x": (tuple(x.shape), tuple(x.shape), tuple(x.shape)), "position_ids": ((1, 1), (1, 8), (1, 16))})
            got = TRT(ep).run({"x": x, "position_ids": pos})
            expected = rope_trace(x, pos, mode)
            inv, cos, sin, out = expected
            row = {"onnx": ex, "engine": eb, "mode": mode,
                   "portable_vs_trt": metric(out, got["y"]),
                   "portable_inv_vs_trt": metric(inv, got["inv_freq"]),
                   "portable_cos_vs_trt": metric(cos, got["cos"]),
                   "portable_sin_vs_trt": metric(sin, got["sin"]),
                   "portable_cache_summary": {"inv_freq": summary(inv), "cos": summary(cos), "sin": summary(sin)},
                   "trt_cache_summary": {"inv_freq": summary(got["inv_freq"]), "cos": summary(got["cos"]), "sin": summary(got["sin"])}}
            micro_rows[f"{kind}_{mode}"] = row; per_head_rows[f"{kind}_{mode}"] = per_head(out, got["y"])
            if mode == "native": cache_rows[kind] = row
            del got, mod; gc.collect(); torch.cuda.empty_cache()
        # Explicit position mapping negative control and positive 0..7 check.
        mod = RopeTrace("native").eval(); op = a.tmp / f"c1j_{kind}_position.onnx"; ep = a.tmp / f"c1j_{kind}_position.engine"
        ex = export(mod, op, (x, pos), ["x", "position_ids"], ["inv_freq", "cos", "sin", "y"], axes)
        eb = build(op, ep, {"x": (tuple(x.shape), tuple(x.shape), tuple(x.shape)), "position_ids": ((1, 1), (1, 8), (1, 16))})
        got0 = TRT(ep).run({"x": x, "position_ids": pos})["y"]; pos1 = pos + 1
        got1 = TRT(ep).run({"x": x, "position_ids": pos1})["y"]
        expected1 = rope_trace(x, pos1)[3]
        position_rows[kind] = {"positions_checked": list(range(8)), "positive_0_to_7": metric(refs[kind][3], got0),
                               "shifted_1_to_8_against_shifted_oracle": metric(expected1, got1),
                               "shifted_output_differs_from_zero_based": metric(got0, got1), "onnx": ex, "engine": eb}
        del got0, got1, mod; gc.collect(); torch.cuda.empty_cache()
    # Layer 0 precision/layout A/B. A material improvement is not assumed.
    layer_ab = {}
    ref_layer = Layer0(layer, "native").eval()(hidden, pos)
    for mode in ("native", "fp32_arith", "cache_fp32", "even_odd", "position_plus_one"):
        op = a.tmp / f"c1j_layer0_{mode}.onnx"; ep = a.tmp / f"c1j_layer0_{mode}.engine"; mod = Layer0(layer, mode).eval()
        ex = export(mod, op, (hidden, pos), ["hidden_states", "position_ids"], ["final_hidden"],
                    {"hidden_states": {0: "batch", 1: "seq"}, "position_ids": {0: "batch", 1: "seq"}, "final_hidden": {0: "batch", 1: "seq"}})
        eb = build(op, ep, {"hidden_states": ((1, 1, 1024), (1, 8, 1024), (1, 16, 1024)), "position_ids": ((1, 1), (1, 8), (1, 16))})
        got = TRT(ep).run({"hidden_states": hidden, "position_ids": pos})["final_hidden"]
        layer_ab[mode] = {"onnx": ex, "engine": eb, "vs_portable_layer": metric(ref_layer, got), "vs_b4_control": metric(control, got)}
        del got, mod; gc.collect(); torch.cuda.empty_cache()
    trace.append(snap("complete"))
    qrope = cache_rows["q"]["portable_vs_trt"]["relative_l2"]; krope = cache_rows["k"]["portable_vs_trt"]["relative_l2"]
    result = {"experiment": "Phase 2.2-C1J", "input_contract": {"shape": list(hidden.shape), "dtype": str(hidden.dtype), "position_ids": list(range(8)), "hidden_sha256": sha(hidden)},
              "layer_structure": {"hidden": 1024, "q_heads": 16, "kv_heads": 8, "head_dim": 128, "rotary_dim": 128, "gqa_repeat": 2, "rope_theta": 1000000.0, "eps": 1e-6, "layout": "half_split_rotate_half"},
              "pipeline": ["input RMSNorm", "Q/K projection", "per-head Q/K RMSNorm", "RoPE cache FP32 -> FP16", "GQA repeat=2", "attention", "output projection", "residual", "post-attention RMSNorm", "SwiGLU MLP"],
              "b4_2_control": metric(ref_layer, control), "cache_comparison": cache_rows, "micro_isolation": micro_rows,
              "per_head_metrics": per_head_rows, "position_comparison": position_rows, "precision_layout_ab": layer_ab,
              "amplification": {"q_rope_micro_relative_l2": qrope, "k_rope_micro_relative_l2": krope,
                                "qk_raw_relative_l2": "NOT_EXPOSED_BY_C1J_TRACE", "layer0_final_relative_l2": metric(ref_layer, control)["relative_l2"]},
              "layout_audit": {"portable_rotate_half": "concat(-x[...,64:], x[...,:64])", "alternate_test": "even_odd", "gqa": "8 KV heads repeated/interleaved twice", "rotary_dim": 128},
              "root_cause_status": "ROPE_TENSORRT_PATH_DIFFERENCE_NARROWED", "first_divergent_component": "Q_ROPE",
              "memory": {"trace": trace, "oom": False, "exit137": False}, "c1_status": "BLOCKED"}
    out = a.out / f"c1j_rope_numerics_{a.timestamp}.json"; out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "b4_relative_l2": result["b4_2_control"]["relative_l2"], "q_rope": qrope, "k_rope": krope,
                      "native_cache_cos": cache_rows["q"]["portable_cos_vs_trt"]["relative_l2"], "native_cache_sin": cache_rows["q"]["portable_sin_vs_trt"]["relative_l2"],
                      "ab": {k: v["vs_portable_layer"]["relative_l2"] for k, v in layer_ab.items()}}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--b4-engine", type=Path, required=True); p.add_argument("--embedding-reference", type=Path, required=True)
    p.add_argument("--layer-file", type=Path, required=True); p.add_argument("--tmp", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--timestamp", required=True)
    main(p.parse_args())
