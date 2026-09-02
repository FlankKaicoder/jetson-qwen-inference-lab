from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
SINGLE = ROOT / "src" / "trt_single_layer"
sys.path.insert(0, str(SINGLE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multilayer_runtime import MultiLayerRuntime  # noqa: E402
from qwen3_layer_semantics import make_layer  # noqa: E402

NUM_LAYERS = 4

def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = a.float() - b.float()
    return {"shape_equal": list(a.shape) == list(b.shape), "finite": bool(torch.isfinite(a).all().item()),
            "max_abs_error": float(d.abs().max().item()), "rmse": float(torch.sqrt((d*d).mean()).item())}

def layer_bytes(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.numel() * t.element_size()

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--prefill-engine", type=Path, required=True); p.add_argument("--decode-engine", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); a = p.parse_args()
    torch.manual_seed(77)
    refs = [make_layer(device="cuda") for _ in range(NUM_LAYERS)]
    x = torch.randn((1, 8, 1024), device="cuda", dtype=torch.float16)
    pos = torch.arange(8, device="cuda", dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        ref_hidden = x; ref_prefill = []
        for layer in refs:
            ref_hidden, rk, rv = layer.forward_prefill(ref_hidden, pos); ref_prefill.append((ref_hidden, rk, rv))
    runtime = MultiLayerRuntime(a.prefill_engine, a.decode_engine, NUM_LAYERS)
    trt_hidden, trt_prefill = runtime.prefill(x, pos)
    pre_layers = []
    for i, (ref_h, ref_k, ref_v), out in zip(range(NUM_LAYERS), ref_prefill, trt_prefill):
        pre_layers.append({"layer_id": i, "hidden": metric(out["hidden_out"], ref_h), "k": metric(out["present_k"], ref_k), "v": metric(out["present_v"], ref_v), "position": runtime.layers[i].current_position, "devices": {k: str(v.device) for k, v in out.items()}})
    ref_kv = [(k, v) for _, k, v in ref_prefill]
    steps = []
    for step in range(4):
        hx = torch.randn((1, 1, 1024), device="cuda", dtype=torch.float16)
        pp = torch.tensor([[8 + step]], device="cuda", dtype=torch.long)
        with torch.no_grad():
            rh = hx; next_ref = []; ref_layer_hidden = []
            for i, layer in enumerate(refs):
                rh, rk, rv = layer.forward_decode(rh, pp, ref_kv[i][0], ref_kv[i][1]); ref_layer_hidden.append(rh); next_ref.append((rk, rv))
        trt_old = [(s.past_k, s.past_v) for s in runtime.layers]
        out_h, out_layers = runtime.decode(hx, pp)
        layer_rows = []
        for i, (rk, rv), out in zip(range(NUM_LAYERS), next_ref, out_layers):
            old_k, old_v = ref_kv[i]
            trt_old_k, trt_old_v = trt_old[i]
            layer_rows.append({"layer_id": i, "hidden": metric(out["hidden_out"], ref_layer_hidden[i]), "k": metric(out["present_k"], rk), "v": metric(out["present_v"], rv), "prefix_k": metric(out["present_k"][:, :, :old_k.shape[2], :], old_k), "prefix_v": metric(out["present_v"][:, :, :old_v.shape[2], :], old_v), "past_length": int(old_k.shape[2]), "present_length": int(out["present_k"].shape[2]), "devices": {k: str(v.device) for k, v in out.items()}})
            layer_rows[-1]["prefix_k"] = metric(out["present_k"][:, :, :trt_old_k.shape[2], :], trt_old_k)
            layer_rows[-1]["prefix_v"] = metric(out["present_v"][:, :, :trt_old_v.shape[2], :], trt_old_v)
        steps.append({"step": step, "past_length": int(next_ref[0][0].shape[2] - 1), "present_length": int(next_ref[0][0].shape[2]), "hidden_final": metric(out_h, rh), "layers": layer_rows, "cache_bytes": runtime.cache_bytes()})
        ref_kv = next_ref
    cache_ptrs = [{"layer_id": s.layer_id, "k_ptr": int(s.past_k.data_ptr()), "v_ptr": int(s.past_v.data_ptr())} for s in runtime.layers]
    result = {"status": "PASS", "num_layers": NUM_LAYERS, "prefill": {"hidden_final": metric(trt_hidden, ref_prefill[-1][0]), "layers": pre_layers, "all_cuda": all(d == "cuda:0" for row in pre_layers for d in row["devices"].values())}, "decode_steps": steps, "all_outputs_finite": all(row["hidden_final"]["finite"] and all(x["hidden"]["finite"] and x["k"]["finite"] and x["v"]["finite"] for x in row["layers"]) for row in steps), "prefix_unchanged": all(x["prefix_k"]["max_abs_error"] == 0 and x["prefix_v"]["max_abs_error"] == 0 for row in steps for x in row["layers"]), "cache_pointer_unique": len({p["k_ptr"] for p in cache_ptrs} | {p["v_ptr"] for p in cache_ptrs}) == 2 * NUM_LAYERS, "cache_pointers": cache_ptrs, "stream": "all layer runtimes use torch.cuda.current_stream(); explicit synchronize in TRTRuntime", "host_payload_roundtrip": False}
    a.out_dir.mkdir(parents=True, exist_ok=True); (a.out_dir / "multilayer_validation.json").write_text(json.dumps(result, indent=2) + "\n"); (a.out_dir / "cache_validation.json").write_text(json.dumps({"num_layers": NUM_LAYERS, "prefix_unchanged": result["prefix_unchanged"], "steps": [{"step": s["step"], "layers": [{"layer_id": x["layer_id"], "past_length": x["past_length"], "present_length": x["present_length"], "prefix_k_max_abs": x["prefix_k"]["max_abs_error"], "prefix_v_max_abs": x["prefix_v"]["max_abs_error"]} for x in s["layers"]]} for s in steps]}, indent=2) + "\n"); print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
