"""Phase 2.2-C4: deterministic greedy sampling integration.

The sampler is deliberately host-side CPU/NumPy argmax. TensorRT and all
historical engines are consumed read-only; no generation loop is implemented.
"""
from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np
import torch

HIDDEN = 1024
VOCAB = 151936
EPS = 1e-6
SEED = 20260903


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


def margin_info(logits: torch.Tensor) -> dict:
    x = logits.detach().float().cpu().reshape(-1, logits.shape[-1])
    values, indices = torch.topk(x, 2, dim=-1)
    margins = values[:, 0] - values[:, 1]
    return {
        "top1_ids": indices[:, 0].tolist(),
        "top2_ids": indices[:, 1].tolist(),
        "top1_values": values[:, 0].tolist(),
        "top2_values": values[:, 1].tolist(),
        "top1_top2_margin": margins.tolist(),
        "minimum_margin": float(margins.min()),
    }


def reference_sampler(logits: torch.Tensor) -> torch.Tensor:
    """Portable/reference semantics: first index wins on exact ties."""
    return logits.detach().float().cpu().argmax(dim=-1)


def cpu_sampler(logits: torch.Tensor) -> tuple[torch.Tensor, int]:
    """C4 backend: copy logits to host and use NumPy argmax."""
    host = logits.detach().float().cpu().numpy()
    return torch.from_numpy(np.argmax(host, axis=-1).astype(np.int64)), int(host.nbytes)


def sampler_compare(logits: torch.Tensor) -> dict:
    ref = reference_sampler(logits)
    got, host_bytes = cpu_sampler(logits)
    return {
        "reference_ids": ref.reshape(-1).tolist(),
        "sampler_ids": got.reshape(-1).tolist(),
        "exact_integer_agreement": bool(torch.equal(ref, got)),
        "shape": list(ref.shape),
        "host_transfer_bytes": host_bytes,
        "margin": margin_info(logits),
        "valid_range": bool(((got >= 0) & (got < VOCAB)).all()),
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
        row.update(cuda_free_bytes=free, cuda_total_bytes=total,
                   torch_allocated_bytes=torch.cuda.memory_allocated(),
                   torch_reserved_bytes=torch.cuda.memory_reserved())
    return row


class LMHead(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.weight.transpose(0, 1))


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


def synthetic_cases() -> dict[str, torch.Tensor]:
    clear = torch.zeros((1, 1, VOCAB), dtype=torch.float32)
    clear[0, 0, 12345] = 4.0
    clear[0, 0, 54321] = 1.0
    near = torch.zeros((1, 1, VOCAB), dtype=torch.float32)
    near[0, 0, 23456] = 1.0
    near[0, 0, 23457] = 0.9999
    tie = torch.zeros((1, 1, VOCAB), dtype=torch.float32)
    tie[0, 0, 34567] = 2.0
    tie[0, 0, 34568] = 2.0
    return {"clear_winner": clear, "near_tie": near, "exact_tie": tie}


def load_c3(a: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    data = torch.load(a.c3_reference, map_location="cpu", weights_only=True)
    weight = torch.load(a.lm_weight, map_location="cpu", weights_only=True).half()
    cases = data["cases"]
    prefill = cases["synthetic_prefill"]["input"].half().contiguous()
    return weight, {"decode": cases["synthetic_decode"]["input"].half().contiguous(),
                    "last_token": prefill[:, -1:, :].contiguous()}


def run(a: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    a.out.mkdir(parents=True, exist_ok=True)
    trace = [mem("start")]
    weight, c3_hidden = load_c3(a)
    portable_head = LMHead(weight).cuda().eval()

    edge = {}
    for name, logits in synthetic_cases().items():
        edge[name] = {
            "sampler": sampler_compare(logits),
            "tie_policy": "first index wins (torch.argmax and NumPy argmax)",
        }

    engine = TRT(a.lm_engine)
    c3_results = {}
    with torch.inference_mode():
        for name, hidden_cpu in c3_hidden.items():
            hidden = hidden_cpu.cuda().contiguous()
            portable_logits = portable_head(hidden)
            trt_logits = engine.run({"hidden_states": hidden})["logits"]
            portable_ids = reference_sampler(portable_logits)
            trt_ids, host_bytes = cpu_sampler(trt_logits)
            c3_results[name] = {
                "hidden_shape": list(hidden.shape),
                "logits_shape": list(trt_logits.shape),
                "logits_metrics": metric(portable_logits, trt_logits),
                "portable": sampler_compare(portable_logits),
                "trt_logits_to_c4_sampler": {
                    "portable_ids": portable_ids.reshape(-1).tolist(),
                    "trt_ids": trt_ids.reshape(-1).tolist(),
                    "exact_integer_agreement": bool(torch.equal(portable_ids, trt_ids)),
                    "host_transfer_bytes": host_bytes,
                    "portable_margin": margin_info(portable_logits),
                    "trt_margin": margin_info(trt_logits),
                    "valid_range": bool(((trt_ids >= 0) & (trt_ids < VOCAB)).all()),
                },
            }

    integration = {
        "status": "NOT_RUN",
        "classification": "END_TO_END_DIAGNOSTIC_ONLY",
    }
    if a.embedding_engine and a.decoder_engine and a.norm_engine and a.norm_weight:
        embedding = TRT(a.embedding_engine)
        decoder = TRT(a.decoder_engine)
        norm_engine = TRT(a.norm_engine)
        norm_weight = torch.load(a.norm_weight, map_location="cpu", weights_only=True).half()
        ids = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
        positions = torch.arange(8, device="cuda", dtype=torch.long).reshape(1, 8)
        embedded = embedding.run({"input_ids": ids})["hidden_states"]
        decoder_out = decoder.run({"hidden_states": embedded, "position_ids": positions})["hidden_l27"]
        normalized_trt = norm_engine.run({"hidden_states": decoder_out})["normalized_hidden_states"]
        last_hidden_trt = normalized_trt[:, -1:, :].contiguous()
        trt_logits = engine.run({"hidden_states": last_hidden_trt})["logits"]
        with torch.inference_mode():
            normalized_portable = FinalNorm(norm_weight).cuda().eval()(decoder_out)[:, -1:, :]
            portable_logits = portable_head(normalized_portable)
        portable_ids = reference_sampler(portable_logits)
        trt_ids, host_bytes = cpu_sampler(trt_logits)
        integration = {
            "status": "PASS" if bool(torch.isfinite(trt_logits).all()) else "FAIL",
            "classification": "END_TO_END_DIAGNOSTIC_ONLY",
            "reason": "C1 decoder numerical drift remains closed and unresolved",
            "input_ids": ids.cpu().tolist(),
            "last_input_token_id": int(ids[0, -1].item()),
            "embedded_shape": list(embedded.shape),
            "decoder_hidden_shape": list(decoder_out.shape),
            "normalized_shape": list(last_hidden_trt.shape),
            "logits_shape": list(trt_logits.shape),
            "finite_cuda": bool(torch.isfinite(normalized_trt).all() and torch.isfinite(trt_logits).all()),
            "portable_next_token": portable_ids.reshape(-1).tolist(),
            "trt_next_token": trt_ids.reshape(-1).tolist(),
            "agreement": bool(torch.equal(portable_ids, trt_ids)),
            "portable_margin": margin_info(portable_logits),
            "trt_margin": margin_info(trt_logits),
            "logits_metrics_same_hidden": metric(portable_logits, trt_logits),
            "host_transfer_bytes": host_bytes,
            "valid_range": bool(((trt_ids >= 0) & (trt_ids < VOCAB)).all()),
            "optional_token_decode": {"status": "NOT_RUN", "reason": "not a C4 gate"},
        }
    trace.append(mem("complete"))
    all_pass = all(v["sampler"]["exact_integer_agreement"] for v in edge.values())
    all_pass = all_pass and all(v["trt_logits_to_c4_sampler"]["exact_integer_agreement"] for v in c3_results.values())
    all_pass = all_pass and integration["status"] == "PASS" and integration["valid_range"]
    payload = {
        "status": "PASS" if all_pass else "BLOCKED",
        "gate": "PASS" if all_pass else "BLOCKED",
        "sampler_backend": "CPU_NUMPY_ARGMAX",
        "greedy_semantics": "next_token_id = argmax(logits, dim=-1); first index wins exact ties",
        "vocab_size": VOCAB,
        "synthetic": edge,
        "c3_same_input": c3_results,
        "full_runtime_single_step": integration,
        "memory": {"trace": trace, "oom": False, "exit137": False},
    }
    (a.out / "c4_greedy_sampling_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "gate": payload["gate"], "backend": payload["sampler_backend"], "integration": integration["status"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--c3-reference", type=Path, required=True)
    p.add_argument("--lm-weight", type=Path, required=True)
    p.add_argument("--lm-engine", type=Path, required=True)
    p.add_argument("--embedding-engine", type=Path)
    p.add_argument("--decoder-engine", type=Path)
    p.add_argument("--norm-engine", type=Path)
    p.add_argument("--norm-weight", type=Path)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
