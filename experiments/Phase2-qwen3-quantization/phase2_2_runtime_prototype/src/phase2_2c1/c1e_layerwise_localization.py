"""C1E-only numerical localization for the existing 28-layer decoder engine.

This script deliberately performs no export, build, or runtime modification. It
reuses the B4.2 engine and C1 handoff/reference artifacts and records every
exposed hidden_l0..hidden_l27 output.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
B4 = HERE.parent / "phase2_2b4_2"
sys.path.insert(0, str(B4))
from portable_qwen3_28 import PortableTwentyEightLayerStack


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a = reference.detach().float().cpu()
    b = candidate.detach().float().cpu()
    d = a - b
    an = torch.linalg.vector_norm(a)
    bn = torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "shape": list(a.shape),
        "candidate_shape": list(b.shape),
        "shape_equal": list(a.shape) == list(b.shape),
        "dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_abs_error": float(d.abs().max()),
        "mean_abs_error": float(d.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(an, min=tiny)),
        "cosine": float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(an * bn, min=tiny)),
    }


def sha_tensor(t: torch.Tensor) -> str:
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class TRT:
    def __init__(self, path: Path):
        import tensorrt as trt

        self.trt = trt
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"ENGINE_DESERIALIZE_FAILED:{path}")
        self.stream = torch.cuda.current_stream()

    def run(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = self.engine.create_execution_context()
        inputs = {"hidden_states": hidden, "position_ids": position_ids}
        for name, tensor in inputs.items():
            if not ctx.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"SHAPE_REJECTED:{name}:{tuple(tensor.shape)}")
            ctx.set_tensor_address(name, tensor.data_ptr())
        outputs: dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != self.trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(ctx.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if dtype != self.trt.DataType.HALF:
                raise RuntimeError(f"UNEXPECTED_OUTPUT_DTYPE:{name}:{dtype}")
            outputs[name] = torch.empty(shape, device="cuda", dtype=torch.float16)
            ctx.set_tensor_address(name, outputs[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("EXECUTE_ASYNC_V3_FAILED")
        self.stream.synchronize()
        return outputs


def load_stack(handoff: Path) -> PortableTwentyEightLayerStack:
    stack = PortableTwentyEightLayerStack(device="cuda", dtype=torch.float16).eval()
    for i in range(28):
        payload = torch.load(handoff / f"layer_{i:02d}.pt", map_location="cpu", weights_only=True)
        stack.layers[i].load_state_dict({k: v.half() for k, v in payload.items()}, strict=True)
        del payload
        gc.collect()
    return stack


def main(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pos = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)

    # D0 is retained as a control, using a deterministic random hidden tensor.
    torch.manual_seed(20260902)
    control = torch.randn((1, 8, 1024), device="cuda", dtype=torch.float16).contiguous()
    stack = load_stack(args.handoff)
    with torch.inference_mode():
        control_ref, _, _ = stack.forward_prefill(control, pos)
    trt = TRT(args.decoder_engine)
    control_out = trt.run(control, pos)
    d0 = metric(control_ref[-1], control_out["hidden_l27"])
    d0["status"] = "PASS" if d0["relative_l2"] <= 0.10 and d0["cosine"] >= 0.99 else "FAIL"
    del control, control_ref, control_out
    gc.collect()
    torch.cuda.empty_cache()

    payload = torch.load(args.embedding_reference, map_location="cpu", weights_only=True)
    canonical = payload["fp16"]["prefill"].contiguous().cuda()
    del payload
    with torch.inference_mode():
        reference_hidden, _, _ = stack.forward_prefill(canonical, pos)
    trt_outputs = trt.run(canonical, pos)

    rows = []
    for i in range(28):
        candidate = trt_outputs[f"hidden_l{i}"]
        row = {"layer": i, "reference": {"shape": list(reference_hidden[i].shape), "dtype": str(reference_hidden[i].dtype)},
               "candidate": {"shape": list(candidate.shape), "dtype": str(candidate.dtype), "sha256_raw": sha_tensor(candidate)},
               "metric": metric(reference_hidden[i], candidate)}
        rows.append(row)

    first_nonzero = next((r["layer"] for r in rows if r["metric"]["max_abs_error"] > 0.0), None)
    first_rel = next((r["layer"] for r in rows if r["metric"]["relative_l2"] > 0.10), None)
    first_cos = next((r["layer"] for r in rows if r["metric"]["cosine"] < 0.99), None)
    result = {
        "experiment": "Phase 2.2-C1E",
        "objective": "Layerwise decoder numerical divergence localization",
        "status": "FIRST_DIVERGENCE_FOUND" if first_nonzero is not None else "LAYERWISE_LOCALIZATION_BLOCKED",
        "inputs": {"shape": [1, 8, 1024], "dtype": "torch.float16", "position_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
                   "decoder_engine": str(args.decoder_engine), "embedding_reference": str(args.embedding_reference), "handoff": str(args.handoff)},
        "d0_b4_2_control": d0,
        "layerwise": rows,
        "thresholds": {"relative_l2_alert": 0.10, "cosine_alert": 0.99},
        "first_nonzero_difference_layer": first_nonzero,
        "first_relative_l2_gt_0_10_layer": first_rel,
        "first_cosine_lt_0_99_layer": first_cos,
        "component_attribution": "NOT PERFORMED: engine exposes layer hidden outputs only; RMSNorm/attention/MLP intermediates are unavailable without rebuilding the engine",
        "memory_or_oom": "NO OOM OR EXIT 137 OBSERVED",
    }
    (out / "phase2_2c1e_layerwise_20260902T190000Z.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "first_nonzero": first_nonzero, "first_relative_l2_gt_0_10": first_rel,
                      "first_cosine_lt_0_99": first_cos, "output": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-engine", type=Path, required=True)
    parser.add_argument("--embedding-reference", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
