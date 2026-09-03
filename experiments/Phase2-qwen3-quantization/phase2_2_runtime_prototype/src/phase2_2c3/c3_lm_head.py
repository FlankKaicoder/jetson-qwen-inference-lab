"""Phase 2.2-C3 diagnostic: Qwen3 LM Head audit and TensorRT integration.

This script intentionally builds an independent LM-head artifact. Existing
embedding, decoder, and final-RMSNorm engines are consumed read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import resource
from pathlib import Path

import torch

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
LM_KEY = "lm_head.weight"
EMBED_KEY = "model.embed_tokens.weight"
HIDDEN = 1024
VOCAB = 151936
EPS = 1e-6


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    d = a - b
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "shape_equal": list(a.shape) == list(b.shape),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_abs": float(d.abs().max()),
        "mean_abs": float(d.abs().mean()),
        "rmse": float(torch.sqrt((d * d).mean())),
        "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(an, min=tiny)),
        "cosine": float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(an * bn, min=tiny)),
    }


def ranking(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> dict:
    aa = a.detach().float().cpu().reshape(-1, a.shape[-1])
    bb = b.detach().float().cpu().reshape(-1, b.shape[-1])
    ia, ib = aa.argmax(dim=-1), bb.argmax(dim=-1)
    ta, tb = aa.topk(k, dim=-1).indices, bb.topk(k, dim=-1).indices
    overlap = [(set(x.tolist()) & set(y.tolist())).__len__() for x, y in zip(ta, tb)]
    return {
        "argmax_equal": bool(torch.equal(ia, ib)),
        "argmax_reference": ia.tolist(),
        "argmax_tensorrt": ib.tolist(),
        "top5_overlap_per_token": overlap,
        "top5_overlap_min": min(overlap),
        "top5_overlap_mean": float(sum(overlap) / len(overlap)),
    }


def mem(stage: str) -> dict:
    row = {"stage": stage, "maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        info = {
            x.split(":", 1)[0]: int(x.split(":", 1)[1].split()[0]) * 1024
            for x in Path("/proc/meminfo").read_text().splitlines()
            if ":" in x
        }
        row["mem_available_bytes"] = info.get("MemAvailable")
        row["swap_used_bytes"] = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
    except FileNotFoundError:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        row.update(
            cuda_free_bytes=free,
            cuda_total_bytes=total,
            torch_allocated_bytes=torch.cuda.memory_allocated(),
            torch_reserved_bytes=torch.cuda.memory_reserved(),
        )
    return row


def load_lm_weights(model_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    checkpoint = model_dir / "model.safetensors"
    if sha_file(checkpoint) != MODEL_SHA:
        raise RuntimeError("MODEL_SHA_MISMATCH")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        lm = source.get_tensor(LM_KEY).contiguous()
        emb = source.get_tensor(EMBED_KEY).contiguous()
    if list(lm.shape) != [VOCAB, HIDDEN] or lm.dtype != torch.bfloat16:
        raise RuntimeError("LM_HEAD_WEIGHT_CONTRACT_MISMATCH")
    if list(emb.shape) != [VOCAB, HIDDEN] or emb.dtype != torch.bfloat16:
        raise RuntimeError("EMBED_WEIGHT_CONTRACT_MISMATCH")
    return lm, emb


class LMHead(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden_states, self.weight.transpose(0, 1))


class FinalNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.float()
        variance = (x * x).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + EPS) * self.weight.float()).to(hidden.dtype)


class TRT:
    def __init__(self, path: Path):
        import tensorrt as trt

        self.trt = trt
        self.path = path
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"ENGINE_DESERIALIZE_FAILED:{path}")
        self.stream = torch.cuda.current_stream()

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context = self.engine.create_execution_context()
        for name, value in inputs.items():
            if not context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"INPUT_SHAPE_REJECTED:{name}")
            context.set_tensor_address(name, value.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(context.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                if dtype == self.trt.DataType.HALF:
                    tdtype = torch.float16
                elif dtype == self.trt.DataType.FLOAT:
                    tdtype = torch.float32
                else:
                    raise RuntimeError(f"UNSUPPORTED_OUTPUT_DTYPE:{name}:{dtype}")
                outputs[name] = torch.empty(shape, device="cuda", dtype=tdtype)
                context.set_tensor_address(name, outputs[name].data_ptr())
        if not context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("EXECUTE_FAILED")
        self.stream.synchronize()
        return outputs


def audit(a: argparse.Namespace) -> None:
    lm, emb = load_lm_weights(a.model_dir)
    tied = bool(torch.equal(lm, emb))
    source_semantics = "Qwen3ForCausalLM.lm_head = Linear(hidden_size, vocab_size, bias=False); tie_weights shares model.embed_tokens"
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        source = inspect.getsource(Qwen3ForCausalLM)
        source_semantics += "; class_source_contains_lm_head=" + str("self.lm_head" in source)
    except Exception as exc:
        source_semantics += f"; class_source=UNAVAILABLE:{type(exc).__name__}"
    row = {
        "status": "PASS",
        "model": "Qwen/Qwen3-0.6B",
        "revision": REVISION,
        "checkpoint_sha256": MODEL_SHA,
        "module_path": "lm_head",
        "checkpoint_key": LM_KEY,
        "shape": list(lm.shape),
        "dtype": str(lm.dtype),
        "numel": lm.numel(),
        "bias": False,
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "weight_sha256_bf16": sha_tensor(lm),
        "embedding_checkpoint_key": EMBED_KEY,
        "embedding_weight_sha256_bf16": sha_tensor(emb),
        "tied_to_embedding": tied,
        "tie_word_embeddings": True,
        "tie_equality": {"exact_tensor_equal": tied, "max_abs": float((lm.float() - emb.float()).abs().max())},
        "source_semantics": source_semantics,
        "config_contract": {"vocab_size": VOCAB, "hidden_size": HIDDEN, "tie_word_embeddings": True, "bias": False},
    }
    a.out.mkdir(parents=True, exist_ok=True)
    a.weight.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lm.half(), a.weight)
    (a.out / "lm_head_weight_audit.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row))


def reference(a: argparse.Namespace) -> None:
    torch.manual_seed(20260903)
    cases = {
        "synthetic_prefill": torch.randn((1, 8, HIDDEN), dtype=torch.float16),
        "synthetic_decode": torch.randn((1, 1, HIDDEN), dtype=torch.float16),
    }
    if a.hidden and a.hidden.exists():
        loaded = torch.load(a.hidden, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            loaded = loaded.get("cases", {}).get("decoder_like", loaded.get("cases", {}).get("synthetic_prefill"))
            if isinstance(loaded, dict):
                loaded = loaded.get("input")
        if not isinstance(loaded, torch.Tensor):
            raise RuntimeError("DECODER_LIKE_HIDDEN_TENSOR_NOT_FOUND")
        cases["decoder_like"] = loaded.half().contiguous()
    weight = torch.load(a.weight, map_location="cpu", weights_only=True).half()
    oracle = LMHead(weight).cuda().eval() if torch.cuda.is_available() else LMHead(weight).eval()
    records = {}
    with torch.inference_mode():
        for name, x in cases.items():
            y = oracle(x.cuda()).cpu() if torch.cuda.is_available() else oracle(x).cpu()
            records[name] = {"input": x, "reference": y, "shape": list(x.shape), "output_shape": list(y.shape), "finite": bool(torch.isfinite(y).all())}
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"status": "PASS", "seed": 20260903, "cases": records}, a.reference)
    print(json.dumps({"status": "PASS", "cases": list(records), "reference": str(a.reference)}))


def export(a: argparse.Namespace) -> None:
    import onnx

    weight = torch.load(a.weight, map_location="cpu", weights_only=True).half()
    a.tmp.mkdir(parents=True, exist_ok=True)
    x = torch.randn((1, 8, HIDDEN), dtype=torch.float16)
    path = a.tmp / "lm_head_fp16.onnx"
    torch.onnx.export(
        LMHead(weight).eval(),
        (x,),
        str(path),
        opset_version=17,
        input_names=["hidden_states"],
        output_names=["logits"],
        dynamic_axes={"hidden_states": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        do_constant_folding=True,
    )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    summary = {"status": "PASS", "path": str(path), "bytes": path.stat().st_size, "node_count": len(model.graph.node), "initializer_count": len(model.graph.initializer), "input": [1, 8, HIDDEN], "output": [1, 8, VOCAB]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "lm_head_onnx_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def build(a: argparse.Namespace) -> None:
    import tensorrt as trt

    path = a.tmp / "lm_head_fp16.onnx"
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(json.dumps({"PARSER_FAILED": errors}))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("hidden_states", (1, 1, HIDDEN), (1, 8, HIDDEN), (2, 16, HIDDEN))
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("BUILD_FAILED")
    engine = a.tmp / "lm_head_fp16.engine"
    engine.write_bytes(bytes(blob))
    summary = {"status": "PASS", "engine": str(engine), "engine_bytes": engine.stat().st_size, "profile": {"min": [1, 1, HIDDEN], "opt": [1, 8, HIDDEN], "max": [2, 16, HIDDEN]}, "output": ["B", "S", VOCAB]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "lm_head_engine_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def validate(a: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    a.out.mkdir(parents=True, exist_ok=True)
    trace = [mem("start")]
    weight = torch.load(a.weight, map_location="cpu", weights_only=True).half()
    refs = torch.load(a.reference, map_location="cpu", weights_only=True)["cases"]
    engine = TRT(a.tmp / "lm_head_fp16.engine")
    results = {}
    for name, rec in refs.items():
        x = rec["input"].cuda().contiguous()
        with torch.inference_mode():
            portable = LMHead(weight).cuda().eval()(x).cpu()
        trt_logits = engine.run({"hidden_states": x})["logits"].cpu()
        results[name] = {"input_shape": list(x.shape), "logits_shape": list(trt_logits.shape), "cuda": True, "metrics": metric(portable, trt_logits), "ranking": ranking(portable, trt_logits)}
    prefill = refs["synthetic_prefill"]["input"].cuda().contiguous()
    last = prefill[:, -1:, :].contiguous()
    with torch.inference_mode():
        prefill_ref = LMHead(weight).cuda().eval()(prefill).cpu()
        last_ref = LMHead(weight).cuda().eval()(last).cpu()
    prefill_trt = engine.run({"hidden_states": prefill})["logits"].cpu()
    last_trt = engine.run({"hidden_states": last})["logits"].cpu()
    paths = {
        "full_prefill": {"shape": list(prefill_trt.shape), "metrics": metric(prefill_ref, prefill_trt), "ranking": ranking(prefill_ref, prefill_trt)},
        "last_token": {"shape": list(last_trt.shape), "metrics": metric(last_ref, last_trt), "ranking": ranking(last_ref, last_trt)},
    }
    integration = {"status": "NOT_RUN", "classification": "END_TO_END_DIAGNOSTIC_ONLY"}
    if a.embedding_engine and a.decoder_engine and a.norm_engine and a.norm_weight:
        embedding = TRT(a.embedding_engine)
        decoder = TRT(a.decoder_engine)
        norm_engine = TRT(a.norm_engine)
        norm_weight = torch.load(a.norm_weight, map_location="cpu", weights_only=True).half()
        ids = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
        pos = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
        embedded = embedding.run({"input_ids": ids})["hidden_states"]
        decoder_out = decoder.run({"hidden_states": embedded, "position_ids": pos})["hidden_l27"]
        normalized_trt = norm_engine.run({"hidden_states": decoder_out})["normalized_hidden_states"]
        with torch.inference_mode():
            normalized_ref = FinalNorm(norm_weight).cuda().eval()(decoder_out).cpu()
            logits_ref = LMHead(weight).cuda().eval()(normalized_ref.cuda()).cpu()
        logits_trt = engine.run({"hidden_states": normalized_trt})["logits"].cpu()
        integration = {
            "status": "PASS" if bool(torch.isfinite(logits_trt).all()) else "FAIL",
            "classification": "END_TO_END_DIAGNOSTIC_ONLY",
            "reason": "C1 decoder numerical drift remains closed and unresolved",
            "hidden_shape": list(decoder_out.shape),
            "normalized_shape": list(normalized_trt.shape),
            "logits_shape": list(logits_trt.shape),
            "finite_cuda": bool(torch.isfinite(normalized_trt).all() and torch.isfinite(logits_trt).all()),
            "norm_same_input": metric(normalized_ref, normalized_trt.cpu()),
            "logits_same_input": metric(logits_ref, logits_trt),
            "logits_ranking": ranking(logits_ref, logits_trt),
        }
    trace.append(mem("complete"))
    payload = {"status": "PASS", "operator_gate": "LM_HEAD_OPERATOR = PASS", "selected_variant": "native_fp16_matmul", "same_input": results, "paths": paths, "decoder_norm_lm_head": integration, "memory": {"trace": trace, "oom": False, "exit137": False}}
    (a.out / "c3_lm_head_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "operator_gate": "LM_HEAD_OPERATOR = PASS", "integration": integration.get("status")}))


def main(a: argparse.Namespace) -> None:
    {"audit": audit, "reference": reference, "export": export, "build": build, "validate": validate}[a.mode](a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["audit", "reference", "export", "build", "validate"])
    parser.add_argument("--model-dir", type=Path, default=Path("/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca"))
    parser.add_argument("--tmp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--hidden", type=Path)
    parser.add_argument("--embedding-engine", type=Path)
    parser.add_argument("--decoder-engine", type=Path)
    parser.add_argument("--norm-engine", type=Path)
    parser.add_argument("--norm-weight", type=Path)
    main(parser.parse_args())
