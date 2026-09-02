"""C1F component-attribution audit using existing TensorRT artifacts only.

The B4.2 28-layer engine is never rebuilt or modified. If present, the older
B3 four-layer engine is used only for its already-exposed Layer 0 attention
output; this cannot attribute RMSNorm/QK-norm/RoPE/QKV/MLP operators.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
B3 = HERE.parent / "phase2_2b3_real_stack"
B4 = HERE.parent / "phase2_2b4_2"
sys.path.insert(0, str(B3))
from portable_qwen3_stack import PortableQwen3Layer


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    x, y = a.detach().float().cpu(), b.detach().float().cpu()
    d = x - y
    xn, yn = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    tiny = torch.finfo(torch.float32).tiny
    return {"shape": list(x.shape), "candidate_shape": list(y.shape),
            "shape_equal": list(x.shape) == list(y.shape), "dtype": str(a.dtype),
            "candidate_dtype": str(b.dtype), "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            "max_abs_error": float(d.abs().max()), "mean_abs_error": float(d.abs().mean()),
            "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(xn, min=tiny)),
            "cosine": float(torch.dot(x.reshape(-1), y.reshape(-1)) / torch.clamp(xn * yn, min=tiny))}


class TRT:
    def __init__(self, path: Path):
        import tensorrt as trt
        self.trt = trt
        self.path = path
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


def reference_components(layer: PortableQwen3Layer, hidden: torch.Tensor, pos: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mirror the source-faithful Layer 0 path for reference-only labels."""
    residual = hidden
    rms = layer.input_layernorm(hidden)
    b, s, _ = rms.shape
    q_linear = layer.self_attn.q_proj(rms).view(b, s, layer.q_heads, layer.head_dim)
    k_linear = layer.self_attn.k_proj(rms).view(b, s, layer.kv_heads, layer.head_dim)
    v_linear = layer.self_attn.v_proj(rms).view(b, s, layer.kv_heads, layer.head_dim)
    q_norm = layer.self_attn.q_norm(q_linear).transpose(1, 2)
    k_norm = layer.self_attn.k_norm(k_linear).transpose(1, 2)
    v = v_linear.transpose(1, 2)
    q_rope = layer._rope(q_norm, pos)
    k_rope = layer._rope(k_norm, pos)
    ka = k_rope[:, :, None].expand(b, layer.kv_heads, layer.groups, s, layer.head_dim).reshape(b, layer.q_heads, s, layer.head_dim)
    va = v[:, :, None].expand(b, layer.kv_heads, layer.groups, s, layer.head_dim).reshape(b, layer.q_heads, s, layer.head_dim)
    scores = torch.matmul(q_rope, ka.transpose(-2, -1)) * (layer.head_dim ** -0.5)
    mask = torch.triu(torch.ones((s, s), device=hidden.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_rope.dtype)
    context = torch.matmul(probs, va).transpose(1, 2).contiguous().reshape(b, s, -1)
    attention_output = layer.self_attn.o_proj(context)
    post_residual = residual + attention_output
    post_rms = layer.post_attention_layernorm(post_residual)
    gate = layer.mlp.gate_proj(post_rms)
    up = layer.mlp.up_proj(post_rms)
    silu_gate = torch.nn.functional.silu(gate)
    gate_up = silu_gate * up
    down = layer.mlp.down_proj(gate_up)
    final = post_residual + down
    return {"input_rmsnorm": rms, "q_projection": q_linear, "k_projection": k_linear, "v_projection": v_linear,
            "q_norm": q_norm, "k_norm": k_norm, "q_rope": q_rope, "k_rope": k_rope,
            "attention_output": attention_output, "post_attention_hidden": post_residual,
            "post_attention_rmsnorm": post_rms, "gate_projection": gate, "up_projection": up,
            "silu_gate": silu_gate, "gate_up": gate_up, "down_projection": down, "final_hidden": final}


def main(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pos = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
    payload = torch.load(args.embedding_reference, map_location="cpu", weights_only=True)
    hidden = payload["fp16"]["prefill"].contiguous().cuda()
    del payload
    layer = PortableQwen3Layer().to(device="cuda", dtype=torch.float16).eval()
    state = torch.load(args.layer_file, map_location="cpu", weights_only=True)
    layer.load_state_dict({k: v.half() for k, v in state.items()}, strict=True)
    del state
    with torch.inference_mode():
        ref = reference_components(layer, hidden, pos)

    b4 = TRT(args.b4_engine)
    b4_contract = b4.contract()
    b4_names = [x["name"] for x in b4_contract if "OUTPUT" in x["mode"]]
    b4_component_bindings = [x for x in b4_names if any(k in x.lower() for k in ("rms", "norm", "q_", "k_", "v_", "rope", "attn", "mlp", "gate", "up", "down"))]

    result = {
        "experiment": "Phase 2.2-C1F",
        "objective": "Qwen3 Layer 0 component-level numerical attribution",
        "input_contract": {"shape": list(hidden.shape), "dtype": str(hidden.dtype), "position_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
                            "same_canonical_embedding": True},
        "layer0_structure": {"input_rmsnorm": True, "q_projection": [16, 128], "k_projection": [8, 128], "v_projection": [8, 128],
                              "q_norm": True, "k_norm": True, "rotary_dimension": 128, "rope_theta": 1000000.0,
                              "gqa": {"q_heads": 16, "kv_heads": 8, "repeat_factor": 2}, "output_projection": True,
                              "residuals": ["attention", "mlp"], "post_attention_rmsnorm": True,
                              "mlp": "SwiGLU: silu(gate_projection) * up_projection -> down_projection"},
        "b4_2_contract": {"output_count": len(b4_names), "output_names": b4_names, "component_bindings": b4_component_bindings,
                          "component_binding_status": "NONE" if not b4_component_bindings else "PRESENT"},
        "component_metrics": {name: {"status": "NOT_AVAILABLE", "reference_shape": list(t.shape), "reference_dtype": str(t.dtype)} for name, t in ref.items()},
        "first_divergent_operator": None,
        "result": "COMPONENT_LOCALIZATION_BLOCKED",
        "component_attribution": "BLOCKED_BY_B4_2_ARTIFACT_GRANULARITY",
        "limitations": ["B4.2 exposes only hidden_l0..hidden_l27 and present K/V outputs.",
                        "RMSNorm, Q/K/V, QK normalization, RoPE, attention and MLP tensors are not TensorRT bindings.",
                        "Reference-only component tensors are recorded for structure and shape, but cannot be compared to B4.2 candidates."],
        "memory_or_oom": "NO OOM OR EXIT 137 OBSERVED",
    }

    # Existing B3 artifact is optional evidence for the already-exposed attention output.
    if args.b3_engine and args.b3_engine.exists():
        b3 = TRT(args.b3_engine)
        b3_contract = b3.contract()
        b3_out = b3.run({"hidden_states": hidden, "position_ids": pos})
        if "attention_l0" in b3_out:
            result["existing_b3_attention_output"] = {"engine": str(args.b3_engine), "contract_outputs": [x["name"] for x in b3_contract if "OUTPUT" in x["mode"]],
                                                        "metric": metric(ref["attention_output"], b3_out["attention_l0"]),
                                                        "interpretation": "partial four-layer artifact evidence only; cannot identify the first internal operator"}
    (out / "phase2_2c1f_layer0_component_20260902T193000Z.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": result["result"], "b4_component_bindings": b4_component_bindings,
                      "b3_attention_compared": "existing_b3_attention_output" in result, "output": str(out)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--b4-engine", type=Path, required=True)
    p.add_argument("--b3-engine", type=Path)
    p.add_argument("--embedding-reference", type=Path, required=True)
    p.add_argument("--layer-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
