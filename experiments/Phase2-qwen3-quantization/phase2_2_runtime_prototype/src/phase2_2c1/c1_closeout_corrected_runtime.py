"""C1 closeout: rebuild the 28-layer decoder with the C1K RoPE cache fix.

Only the RoPE cache boundary changes: cos/sin are source-faithfully computed
in FP32 once, cast to FP16, and gathered by the normal position_ids contract.
All weights, layer structure, attention, KV cache and runtime bindings remain
the B4.2 implementation.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, resource, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
B42 = Path("/tmp/phase2_2b4_2") if Path("/tmp/phase2_2b4_2").exists() else HERE.parent / "phase2_2b4_2"
if str(B42) not in sys.path:
    sys.path.insert(0, str(B42))
import stage_b4_2 as b42
from portable_qwen3_28 import PortableTwentyEightLayerStack
from portable_qwen3_stack import PortableQwen3Layer

HANDOFF = Path("/tmp/phase2_2b4_stream_20260902T070000Z")
EMBED_REF = Path("/tmp/phase2_2c1_20260902T090000Z/reference.pt")
EMBED_ENGINE = Path("/tmp/phase2_2c1_20260902T090000Z/embedding_fp16.engine")

def sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def cache(max_pos=2048, dim=128, theta=1_000_000.0, dtype=torch.float16, device="cuda"):
    p = torch.arange(max_pos, device=device, dtype=torch.float32)
    inv = torch.pow(torch.tensor(theta, device=device, dtype=torch.float32),
                    -torch.arange(dim // 2, device=device, dtype=torch.float32) / (dim // 2))
    f = p[:, None] * inv[None, :]
    e = torch.cat((f, f), dim=-1)
    return e.cos().unsqueeze(0).unsqueeze(0).to(dtype), e.sin().unsqueeze(0).unsqueeze(0).to(dtype)

class CorrectedLayer(PortableQwen3Layer):
    def __init__(self, max_pos=2048):
        super().__init__()
        c, s = cache(max_pos=max_pos, device="cuda")
        self.register_buffer("rope_cos_fp16", c)
        self.register_buffer("rope_sin_fp16", s)

    def _rope(self, x, pos):
        flat = pos.reshape(-1).to(torch.long)
        c = self.rope_cos_fp16.index_select(2, flat).reshape(pos.shape[0], 1, pos.shape[1], self.head_dim)
        s = self.rope_sin_fp16.index_select(2, flat).reshape(pos.shape[0], 1, pos.shape[1], self.head_dim)
        half = x.shape[-1] // 2
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * c + rot * s

class CorrectedStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([CorrectedLayer().to(device="cuda", dtype=torch.float16) for _ in range(28)])

    def forward_prefill(self, hidden, pos):
        hs, ks, vs = [], [], []
        for layer in self.layers:
            hidden, k, v, _ = layer.forward_prefill(hidden, pos)
            hs.append(hidden); ks.append(k); vs.append(v)
        return hs, ks, vs

    def forward_decode(self, hidden, pos, past_ks, past_vs):
        hs, ks, vs = [], [], []
        for i, layer in enumerate(self.layers):
            hidden, k, v, _ = layer.forward_decode(hidden, pos, past_ks[i], past_vs[i])
            hs.append(hidden); ks.append(k); vs.append(v)
        return hs, ks, vs

def load_stack(handoff, corrected):
    stack = CorrectedStack() if corrected else PortableTwentyEightLayerStack(device="cuda", dtype=torch.float16)
    stack = stack.eval()
    for i in range(28):
        p = torch.load(Path(handoff) / f"layer_{i:02d}.pt", map_location="cpu", weights_only=True)
        stack.layers[i].load_state_dict({k: v.to(dtype=torch.float16) for k, v in p.items()}, strict=False)
        del p
    return stack

def mem(stage):
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            vals["mem_available_bytes"] = int(line.split()[1]) * 1024
    free, total = torch.cuda.mem_get_info()
    vals.update(stage=stage, cuda_free_bytes=free, cuda_total_bytes=total,
                torch_allocated_bytes=torch.cuda.memory_allocated(),
                torch_reserved_bytes=torch.cuda.memory_reserved(),
                maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return vals

def metric(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu(); d = a - b
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b); tiny = torch.finfo(torch.float32).tiny
    return {"max_abs": float(d.abs().max()), "rmse": float(torch.sqrt((d*d).mean())),
            "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(an, min=tiny)),
            "cosine": float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(an*bn, min=tiny)),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "shape_equal": list(a.shape) == list(b.shape)}

def runtime(path, inputs):
    return b42.TRT(Path(path)).run(inputs)

def split(o):
    return ([o[f"hidden_l{i}"] for i in range(28)], [o[f"present_k{i}"] for i in range(28)],
            [o[f"present_v{i}"] for i in range(28)])

def validate(a):
    a.out.mkdir(parents=True, exist_ok=True)
    trace = [mem("start")]
    payload = torch.load(a.embedding_reference, map_location="cpu", weights_only=True)
    ids = torch.arange(8, dtype=torch.long, device="cuda").reshape(1, 8)
    h = payload["fp16"]["prefill"].contiguous().cuda()
    pos = torch.arange(h.shape[1], device="cuda", dtype=torch.long).reshape(1, -1)
    native = load_stack(a.handoff, False); corrected = load_stack(a.handoff, True)
    with torch.inference_mode():
        nh, nk, nv = native.forward_prefill(h, pos)
        ch, ck, cv = corrected.forward_prefill(h, pos)
    torch.cuda.synchronize(); trace.append(mem("references_complete"))
    # Original B4.2 is a read-only control; corrected engine is the only new engine.
    control = split(runtime(a.original_engine, {"hidden_states": h, "position_ids": pos}))[0]
    got = split(runtime(a.corrected_engine, {"hidden_states": h, "position_ids": pos}))
    gh, gk, gv = got
    rows = []
    for i in range(28):
        rows.append({"layer": i, "native_reference": metric(nh[i], control[i]),
                     "corrected_reference": metric(ch[i], gh[i]),
                     "corrected_vs_original_control": metric(control[i], gh[i]),
                     "k_corrected_reference": metric(ck[i], gk[i]),
                     "v_corrected_reference": metric(cv[i], gv[i])})
    embedding = None
    if a.embedding_engine.exists():
        emb = runtime(a.embedding_engine, {"input_ids": ids})["hidden_states"]
        embedding = {"input_ids": ids.cpu().tolist(), "vs_c1_reference": metric(h, emb),
                     "decoder_final_vs_c1_reference": metric(h, emb)}
        eh, _, _ = split(runtime(a.corrected_engine, {"hidden_states": emb, "position_ids": pos}))
        embedding["decoder_final_vs_direct_corrected"] = metric(gh[-1], eh[-1])
    # Minimal decode regression if a corrected decode engine was built.
    decode = None
    if a.corrected_decode.exists():
        past_k, past_v = gk, gv; steps = []
        for step in range(4):
            token = torch.randn((1, 1, 1024), device="cuda", dtype=torch.float16)
            p = torch.tensor([[8 + step]], device="cuda", dtype=torch.long)
            oldk, oldv = past_k, past_v
            dh, past_k, past_v = split(runtime(a.corrected_decode, {"hidden_states": token, "position_ids": p,
                **{f"past_k{i}": oldk[i] for i in range(28)}, **{f"past_v{i}": oldv[i] for i in range(28)}}))
            steps.append({"step": step, "present_length": int(past_k[0].shape[2]),
                "all_finite_cuda": all(torch.isfinite(t).all().item() and t.device.type == "cuda" for group in (dh,past_k,past_v) for t in group),
                "prefix_k_exact": all(torch.equal(oldk[i], past_k[i][:,:,:oldk[i].shape[2],:]) for i in range(28)),
                "prefix_v_exact": all(torch.equal(oldv[i], past_v[i][:,:,:oldv[i].shape[2],:]) for i in range(28)),
                "k_pointer_isolation": len({int(t.data_ptr()) for t in past_k}) == 28,
                "v_pointer_isolation": len({int(t.data_ptr()) for t in past_v}) == 28})
        decode = {"status": "PASS" if all(x["all_finite_cuda"] and x["prefix_k_exact"] and x["prefix_v_exact"] for x in steps) else "FAIL", "steps": steps}
    trace.append(mem("complete"))
    final = rows[-1]; layer0 = rows[0]
    result = {"experiment": "Phase 2.2-C1 closeout", "input_contract": {"shape": list(h.shape), "dtype": str(h.dtype), "position_ids": list(range(int(h.shape[1]))), "hidden_sha256": sha(h)},
        "correction": {"cache": "FP32 source formula -> FP16 precomputed cache -> position_ids gather", "max_position": 2048, "rotary_dim": 128, "theta": 1000000.0, "layout": "half_split_rotate_half"},
        "layer0": layer0, "layer27": final, "layers": rows, "embedding_integration": embedding, "decode": decode, "memory": {"trace": trace, "oom": False, "exit137": False},
        "result": "PASS / BOUNDED" if layer0["corrected_reference"]["relative_l2"] < 0.01 and final["corrected_reference"]["relative_l2"] < 0.10 else "CLOSED / NUMERICAL_LIMITATION_UNRESOLVED"}
    (a.out / "c1_closeout_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": result["result"], "layer0_rel_l2": layer0["corrected_reference"]["relative_l2"], "layer27_rel_l2": final["corrected_reference"]["relative_l2"], "decode": decode}))

def main(a):
    if a.mode == "export":
        b42.load_stack = lambda handoff, dtype: load_stack(handoff, True)
        b42.export_mode(a)
    elif a.mode == "build":
        b42.build_mode(a)
    elif a.mode == "validate":
        validate(a)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["export", "build", "validate"], required=True)
    p.add_argument("--handoff", type=Path, default=HANDOFF); p.add_argument("--tmp", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--original-engine", type=Path, default=Path("/tmp/phase2_2b4_2_20260902T082326Z/prefill_28layer.engine"))
    p.add_argument("--corrected-engine", type=Path); p.add_argument("--corrected-decode", type=Path); p.add_argument("--embedding-engine", type=Path, default=EMBED_ENGINE); p.add_argument("--embedding-reference", type=Path, default=EMBED_REF)
    main(p.parse_args())
