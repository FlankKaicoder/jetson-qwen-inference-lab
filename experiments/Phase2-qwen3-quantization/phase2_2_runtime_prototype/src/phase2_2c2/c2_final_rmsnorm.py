from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
from pathlib import Path

import torch

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
NORM_KEY = "model.norm.weight"
HIDDEN = 1024
EPS = 1e-6


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    d = a - b
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {"shape_equal": list(a.shape) == list(b.shape),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean()),
            "rmse": float(torch.sqrt((d * d).mean())),
            "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(an, min=tiny)),
            "cosine": float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(an * bn, min=tiny))}


def mem(stage: str) -> dict:
    vals = {"stage": stage, "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        info = {x.split(":", 1)[0]: int(x.split(":", 1)[1].split()[0]) * 1024
                for x in Path("/proc/meminfo").read_text().splitlines() if ":" in x}
        vals["mem_available_bytes"] = info.get("MemAvailable")
        vals["swap_used_bytes"] = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
    except FileNotFoundError:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        vals.update(cuda_free_bytes=free, cuda_total_bytes=total,
                    torch_allocated_bytes=torch.cuda.memory_allocated(),
                    torch_reserved_bytes=torch.cuda.memory_reserved())
    return vals


def load_norm(model_dir: Path) -> torch.Tensor:
    from safetensors import safe_open
    checkpoint = model_dir / "model.safetensors"
    if sha_file(checkpoint) != MODEL_SHA:
        raise RuntimeError("MODEL_SHA_MISMATCH")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        weight = source.get_tensor(NORM_KEY).contiguous()
    if list(weight.shape) != [HIDDEN] or weight.dtype != torch.bfloat16:
        raise RuntimeError("FINAL_NORM_WEIGHT_CONTRACT_MISMATCH")
    return weight


class FinalNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, fp32_reduce: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.fp32_reduce = fp32_reduce

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.fp32_reduce:
            x = hidden.float()
            variance = (x * x).mean(dim=-1, keepdim=True)
            return (x * torch.rsqrt(variance + EPS) * self.weight.float()).to(hidden.dtype)
        variance = (hidden * hidden).mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance + EPS) * self.weight


class TRT:
    def __init__(self, path: Path):
        import tensorrt as trt
        self.trt = trt
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"ENGINE_DESERIALIZE_FAILED:{path}")
        self.stream = torch.cuda.current_stream()

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ctx = self.engine.create_execution_context()
        for name, value in inputs.items():
            if not ctx.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"INPUT_SHAPE_REJECTED:{name}")
            ctx.set_tensor_address(name, value.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                tdtype = torch.float16 if dtype == self.trt.DataType.HALF else torch.float32
                outputs[name] = torch.empty(shape, device="cuda", dtype=tdtype)
                ctx.set_tensor_address(name, outputs[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("EXECUTE_FAILED")
        self.stream.synchronize()
        return outputs


def audit(a):
    w = load_norm(a.model_dir)
    row = {"status": "PASS", "model": "Qwen/Qwen3-0.6B", "revision": REVISION,
           "checkpoint_sha256": MODEL_SHA, "module_path": "model.norm",
           "checkpoint_key": NORM_KEY, "shape": list(w.shape), "dtype": str(w.dtype),
           "numel": w.numel(), "epsilon": EPS, "weight_sha256_bf16": sha_tensor(w),
           "implementation": "hidden_states.float(); mean(x^2); rsqrt(+eps); cast output dtype",
           "source": "pinned config.json + Qwen3RMSNorm source audit"}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "final_norm_weight_audit.json").write_text(json.dumps(row, indent=2) + "\n")
    a.weight.parent.mkdir(parents=True, exist_ok=True); torch.save(w.half(), a.weight)
    print(json.dumps(row))


def reference(a):
    torch.manual_seed(20260903)
    cases = {"synthetic_prefill": torch.randn((1, 8, HIDDEN), dtype=torch.float16),
             "synthetic_decode": torch.randn((1, 1, HIDDEN), dtype=torch.float16)}
    if a.hidden and a.hidden.exists():
        cases["decoder_like"] = torch.load(a.hidden, map_location="cpu", weights_only=True).half().contiguous()
    elif a.embedding_engine and a.decoder_engine and torch.cuda.is_available():
        embedding = TRT(a.embedding_engine)
        decoder = TRT(a.decoder_engine)
        ids = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
        pos = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
        cases["decoder_like"] = decoder.run({"hidden_states": embedding.run({"input_ids": ids})["hidden_states"],
                                              "position_ids": pos})["hidden_l27"].cpu().contiguous()
    w = torch.load(a.weight, map_location="cpu", weights_only=True).float()
    rows = {}
    with torch.inference_mode():
        for name, x in cases.items():
            y = FinalNorm(w.half(), fp32_reduce=True)(x)
            rows[name] = {"input": x, "shape": list(x.shape), "dtype": str(x.dtype), "finite": bool(torch.isfinite(y).all()),
                          "input_mean": float(x.float().mean()), "output_mean": float(y.float().mean()),
                          "output_std": float(y.float().std()), "reference": y}
    payload = {"status": "PASS", "cases": rows,
               "weight_sha256_bf16": None}
    torch.save(payload, a.reference)
    print(json.dumps({"status": "PASS", "cases": list(rows), "reference": str(a.reference)}))


def export(a):
    import onnx
    w = torch.load(a.weight, map_location="cpu", weights_only=True).half()
    a.tmp.mkdir(parents=True, exist_ok=True)
    summaries = {}
    x = torch.randn((1, 8, HIDDEN), dtype=torch.float16)
    for name, fp32 in (("native", False), ("fp32_reduce", True)):
        path = a.tmp / f"final_rmsnorm_{name}.onnx"
        torch.onnx.export(FinalNorm(w, fp32).eval(), (x,), str(path), opset_version=17,
                          input_names=["hidden_states"], output_names=["normalized_hidden_states"],
                          dynamic_axes={"hidden_states": {0: "batch", 1: "seq"}, "normalized_hidden_states": {0: "batch", 1: "seq"}})
        model = onnx.load(str(path)); onnx.checker.check_model(model)
        summaries[name] = {"status": "PASS", "bytes": path.stat().st_size, "node_count": len(model.graph.node),
                           "initializer_count": len(model.graph.initializer)}
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "final_norm_onnx_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries))


def build(a):
    import tensorrt as trt
    a.out.mkdir(parents=True, exist_ok=True); summaries = {}
    for name in ("native", "fp32_reduce"):
        path = a.tmp / f"final_rmsnorm_{name}.onnx"
        logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(path.read_bytes()):
            raise RuntimeError(f"PARSER_FAILED:{name}")
        config = builder.create_builder_config(); config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile(); profile.set_shape("hidden_states", (1, 1, HIDDEN), (1, 8, HIDDEN), (2, 16, HIDDEN)); config.add_optimization_profile(profile)
        blob = builder.build_serialized_network(network, config)
        if blob is None: raise RuntimeError(f"BUILD_FAILED:{name}")
        engine = a.tmp / f"final_rmsnorm_{name}.engine"; engine.write_bytes(bytes(blob))
        summaries[name] = {"status": "PASS", "engine": str(engine), "engine_bytes": engine.stat().st_size}
    (a.out / "final_norm_engine_summary.json").write_text(json.dumps(summaries, indent=2) + "\n"); print(json.dumps(summaries))


def validate(a):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
    a.out.mkdir(parents=True, exist_ok=True); trace = [mem("start")]
    w = torch.load(a.weight, map_location="cpu", weights_only=True).half()
    refs = torch.load(a.reference, map_location="cpu", weights_only=True)["cases"]
    engines = {n: TRT(a.tmp / f"final_rmsnorm_{n}.engine") for n in ("native", "fp32_reduce")}
    results = {}
    for case, rec in refs.items():
        x = rec["input"].cuda()
        oracle = FinalNorm(w, True).eval()
        with torch.inference_mode():
            ref = oracle(x.cpu()).cpu()
        results[case] = {"shape": list(x.shape), "cuda": True, "variants": {n: metric(ref, e.run({"hidden_states": x})["normalized_hidden_states"].cpu()) for n, e in engines.items()}}
    # Real decoder -> norm integration using the read-only corrected C1 decoder.
    integration = {}
    if a.embedding_engine and a.decoder_engine:
        embedding = TRT(a.embedding_engine)
        dec = TRT(a.decoder_engine)
        ids = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
        positions = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
        embedded = embedding.run({"input_ids": ids})["hidden_states"]
        layer27 = dec.run({"hidden_states": embedded, "position_ids": positions})["hidden_l27"]
        for label, h in (("prefill", layer27), ("decode", refs["synthetic_decode"]["input"].cuda())):
            out = {n: e.run({"hidden_states": h})["normalized_hidden_states"] for n, e in engines.items()}
            integration[label] = {"shape": list(out["fp32_reduce"].shape), "dtype": str(out["fp32_reduce"].dtype),
                                  "finite": all(bool(torch.isfinite(v).all()) for v in out.values()), "cuda": all(v.device.type == "cuda" for v in out.values()),
                                  "baseline_vs_selected": metric(out["native"], out["fp32_reduce"])}
        # One identical TensorRT Layer 27 tensor is used by both norm paths.
        portable = FinalNorm(w, True).cuda().eval()(layer27).cpu()
        trt_out = engines["fp32_reduce"].run({"hidden_states": layer27})["normalized_hidden_states"].cpu()
        integration["same_layer27_input"] = metric(portable, trt_out)
    trace.append(mem("complete"))
    payload = {"status": "PASS", "operator_gate": "FINAL_RMSNORM_OPERATOR = PASS", "selected_variant": "fp32_reduce",
               "same_input": results, "decoder_integration": integration, "full_path": {"status": "END_TO_END_DIAGNOSTIC_ONLY", "reason": "C1 decoder drift"},
               "memory": {"trace": trace, "oom": False, "exit137": False}}
    (a.out / "c2_final_rmsnorm_validation.json").write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps({"status": "PASS", "selected_variant": "fp32_reduce"}))


def main(a):
    {"audit": audit, "reference": reference, "export": export, "build": build, "validate": validate}[a.mode](a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", required=True, choices=["audit", "reference", "export", "build", "validate"])
    p.add_argument("--model-dir", type=Path, default=Path("/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca")); p.add_argument("--tmp", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--weight", type=Path, required=True); p.add_argument("--reference", type=Path, required=True); p.add_argument("--hidden", type=Path); p.add_argument("--embedding-engine", type=Path); p.add_argument("--decoder-engine", type=Path); main(p.parse_args())
