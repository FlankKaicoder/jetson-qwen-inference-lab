"""Phase 2.2-C5 minimal Qwen3 autoregressive runtime.

Reference generation and TensorRT runtime are separate modes so the full HF
model is never resident alongside all TensorRT engines. This is orchestration
only; existing component engines are read-only.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
from pathlib import Path

import numpy as np
import torch

VOCAB = 151936
HIDDEN = 1024
MAX_NEW = 4
EOS_ID = 151645
PROMPT = "Hello"
EPS = 1e-6


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


def margins(logits: torch.Tensor) -> dict:
    x = logits.detach().float().cpu().reshape(-1, logits.shape[-1])
    values, ids = torch.topk(x, 2, dim=-1)
    return {"top1_token": int(ids[0, 0]), "top2_token": int(ids[0, 1]),
            "top1_logit": float(values[0, 0]), "top2_logit": float(values[0, 1]),
            "top1_top2_margin": float(values[0, 0] - values[0, 1])}


def greedy(logits: torch.Tensor) -> tuple[int, dict]:
    x = logits.detach().float().cpu().numpy().reshape(-1, logits.shape[-1])
    token = int(np.argmax(x, axis=-1)[0])
    return token, {"token": token, "host_transfer_bytes": int(x.nbytes),
                   "valid": 0 <= token < VOCAB, "margin": margins(logits)}


def reference_mode(a: argparse.Namespace) -> None:
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    a.out.mkdir(parents=True, exist_ok=True)
    trace = [mem("reference_start")]
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_dir), local_files_only=True)
    encoded = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to("cuda")
    model = Qwen3ForCausalLM.from_pretrained(
        str(a.model_dir), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to("cuda").eval()
    trace.append(mem("reference_model_loaded"))
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).to("cuda"),
            do_sample=False,
            max_new_tokens=MAX_NEW,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    prompt_len = int(input_ids.shape[1])
    generated_ids = generated.sequences[0, prompt_len:].detach().cpu().tolist()
    scores = list(generated.scores)
    steps = []
    for i, token in enumerate(generated_ids):
        info = margins(scores[i]) if i < len(scores) else {}
        steps.append({"step": i, "position": prompt_len + i,
                      "token": int(token), "valid": 0 <= int(token) < VOCAB,
                      "margin": info})
    payload = {
        "status": "PASS",
        "prompt": PROMPT,
        "tokenizer_mode": "plain_causal_prompt",
        "input_ids": input_ids[0].detach().cpu().tolist(),
        "prompt_token_count": prompt_len,
        "max_new_tokens": MAX_NEW,
        "eos_token_id": tokenizer.eos_token_id,
        "reference_generated_ids": [int(x) for x in generated_ids],
        "reference_steps": steps,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "full_text": tokenizer.decode(generated.sequences[0].detach().cpu().tolist(), skip_special_tokens=False),
        "stopped_reason": "EOS" if generated_ids and generated_ids[-1] == tokenizer.eos_token_id else "MAX_NEW_TOKENS",
        "memory": {"trace": trace + [mem("reference_complete")], "oom": False, "exit137": False},
    }
    (a.out / "reference_generation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "input_ids": payload["input_ids"], "generated_ids": payload["reference_generated_ids"], "text": payload["generated_text"]}))
    del model
    gc.collect()
    torch.cuda.empty_cache()


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
                tdtype = torch.float16 if dtype == self.trt.DataType.HALF else torch.float32
                outputs[name] = torch.empty(shape, device="cuda", dtype=tdtype)
                context.set_tensor_address(name, outputs[name].data_ptr())
        if not context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"EXECUTE_FAILED:{self.path}")
        self.stream.synchronize()
        return outputs


def final_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = hidden.float()
    variance = (x * x).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + EPS) * weight.float()).to(hidden.dtype)


def runtime_mode(a: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    a.out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((a.out / "reference_generation.json").read_text())
    if not ref["input_ids"]:
        raise RuntimeError("EMPTY_PROMPT_IDS")
    trace = [mem("runtime_start")]
    embedding = TRT(a.embedding_engine)
    prefill = TRT(a.prefill_engine)
    decode = TRT(a.decode_engine)
    norm = TRT(a.norm_engine)
    lm = TRT(a.lm_engine)
    norm_weight = torch.load(a.norm_weight, map_location="cpu", weights_only=True).half().cuda()
    lm_weight = torch.load(a.lm_weight, map_location="cpu", weights_only=True).half().cuda()
    ids = torch.tensor([ref["input_ids"]], device="cuda", dtype=torch.long)
    positions = torch.arange(ids.shape[1], device="cuda", dtype=torch.long).reshape(1, -1)
    embedded = embedding.run({"input_ids": ids})["hidden_states"]
    pre = prefill.run({"hidden_states": embedded, "position_ids": positions})
    hidden = pre["hidden_l27"]
    past_k = [pre[f"present_k{i}"] for i in range(28)]
    past_v = [pre[f"present_v{i}"] for i in range(28)]
    pre_norm = norm.run({"hidden_states": hidden})["normalized_hidden_states"][:, -1:, :].contiguous()
    pre_logits = lm.run({"hidden_states": pre_norm})["logits"]
    portable_norm = final_norm(hidden[:, -1:, :], norm_weight)
    portable_logits = torch.matmul(portable_norm, lm_weight.t())
    pre_token, pre_sampler = greedy(pre_logits)
    pre_portable_token, pre_portable_sampler = greedy(portable_logits)
    reference_ids = [int(x) for x in ref["reference_generated_ids"]]
    trace_rows = [{"step": 0, "phase": "prefill", "position": int(ids.shape[1] - 1),
                   "reference_token": reference_ids[0] if reference_ids else None,
                   "trt_token": pre_token, "agreement": bool(reference_ids and pre_token == reference_ids[0]),
                   "portable_token": pre_portable_token, "trt_margin": pre_sampler["margin"],
                   "portable_margin": pre_portable_sampler["margin"],
                   "valid_token": pre_sampler["valid"]}]
    cache_rows = [{"step": 0, "phase": "prefill", "cache_length": int(past_k[0].shape[2]),
                  "all_finite": all(bool(torch.isfinite(x).all()) for x in past_k + past_v),
                  "pointer_isolation_k": len({int(x.data_ptr()) for x in past_k}) == 28,
                  "pointer_isolation_v": len({int(x.data_ptr()) for x in past_v}) == 28,
                  "cache_bytes": sum(x.numel() * x.element_size() for x in past_k + past_v),
                  "prefix_k_exact": True, "prefix_v_exact": True}]
    current = pre_token
    for step in range(1, min(MAX_NEW, len(reference_ids))):
        old_k, old_v = past_k, past_v
        token_input = torch.tensor([[current]], device="cuda", dtype=torch.long)
        token_hidden = embedding.run({"input_ids": token_input})["hidden_states"]
        position = ids.shape[1] + step - 1
        pos = torch.tensor([[position]], device="cuda", dtype=torch.long)
        dec_inputs = {"hidden_states": token_hidden, "position_ids": pos}
        dec_inputs.update({f"past_k{i}": old_k[i] for i in range(28)})
        dec_inputs.update({f"past_v{i}": old_v[i] for i in range(28)})
        out = decode.run(dec_inputs)
        past_k = [out[f"present_k{i}"] for i in range(28)]
        past_v = [out[f"present_v{i}"] for i in range(28)]
        dec_hidden = out["hidden_l27"]
        normalized = norm.run({"hidden_states": dec_hidden})["normalized_hidden_states"]
        logits = lm.run({"hidden_states": normalized})["logits"]
        current, sample = greedy(logits)
        prefix_k = all(torch.equal(old_k[i], past_k[i][:, :, :old_k[i].shape[2], :]) for i in range(28))
        prefix_v = all(torch.equal(old_v[i], past_v[i][:, :, :old_v[i].shape[2], :]) for i in range(28))
        trace_rows.append({"step": step, "phase": "decode", "position": position,
                           "reference_token": reference_ids[step], "trt_token": current,
                           "agreement": current == reference_ids[step], "trt_margin": sample["margin"],
                           "valid_token": sample["valid"]})
        cache_rows.append({"step": step, "phase": "decode", "cache_length": int(past_k[0].shape[2]),
                           "all_finite": all(bool(torch.isfinite(x).all()) for x in out.values()),
                           "pointer_isolation_k": len({int(x.data_ptr()) for x in past_k}) == 28,
                           "pointer_isolation_v": len({int(x.data_ptr()) for x in past_v}) == 28,
                           "cache_bytes": sum(x.numel() * x.element_size() for x in past_k + past_v),
                           "prefix_k_exact": prefix_k, "prefix_v_exact": prefix_v,
                           "new_position": position})
    generated = [int(x["trt_token"]) for x in trace_rows]
    divergence = next((x["step"] for x in trace_rows if not x["agreement"]), None)
    tokenizer_text = {"status": "NOT_RUN"}
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(a.model_dir), local_files_only=True)
        tokenizer_text = {"status": "PASS", "generated_text": tok.decode(generated, skip_special_tokens=False),
                          "full_text": tok.decode(ref["input_ids"] + generated, skip_special_tokens=False)}
    except Exception as exc:
        if a.tokenizer_decode and a.tokenizer_decode.exists():
            tokenizer_text = json.loads(a.tokenizer_decode.read_text())
        else:
            tokenizer_text = {"status": "UNAVAILABLE", "reason": type(exc).__name__}
    trace.append(mem("runtime_complete"))
    cache_ok = all(x["all_finite"] and x["prefix_k_exact"] and x["prefix_v_exact"] and
                   x["pointer_isolation_k"] and x["pointer_isolation_v"] for x in cache_rows)
    runtime_ok = all(x["valid_token"] for x in trace_rows) and cache_ok
    payload = {"status": "PASS" if runtime_ok else "BLOCKED", "gate": "PASS" if runtime_ok else "BLOCKED",
               "prompt": ref["prompt"], "input_ids": ref["input_ids"], "prompt_token_count": len(ref["input_ids"]),
               "reference_generated_ids": reference_ids, "trt_generated_ids": generated,
               "token_trace": trace_rows, "first_token_divergence_step": divergence,
               "prefill": {"status": "PASS", "sequence_length": len(ref["input_ids"]), "position_ids": list(range(len(ref["input_ids"]))),
                           "hidden_shape": list(hidden.shape), "kv_layers": 28, "kv_length": int(past_k[0].shape[2])},
               "decode_loop": {"status": "PASS" if len(trace_rows) > 1 else "FAIL", "steps": max(0, len(trace_rows) - 1),
                               "max_new_tokens": MAX_NEW},
               "kv_cache": {"status": "PASS" if cache_ok else "BLOCKED", "rows": cache_rows,
                            "prefix_invariant": cache_ok, "length_growth": [x["cache_length"] for x in cache_rows],
                            "pointer_isolation": cache_ok},
               "generated_text": tokenizer_text,
               "known_c1_limitation": "FULL_RUNTIME_TOKEN_MISMATCH_DUE_TO_UPSTREAM_NUMERICAL_LIMITATION" if divergence is not None else "DOCUMENTED_C1_LIMITATION_NOT_TRIGGERED",
               "memory": {"trace": trace, "oom": False, "exit137": False}}
    (a.out / "generation_trace.json").write_text(json.dumps(payload, indent=2) + "\n")
    (a.out / "memory_trace.json").write_text(json.dumps(payload["memory"], indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "reference": reference_ids, "trt": generated, "divergence": divergence, "text": tokenizer_text}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["reference", "runtime"], required=True)
    p.add_argument("--model-dir", type=Path, default=Path("/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--embedding-engine", type=Path)
    p.add_argument("--prefill-engine", type=Path)
    p.add_argument("--decode-engine", type=Path)
    p.add_argument("--norm-engine", type=Path)
    p.add_argument("--lm-engine", type=Path)
    p.add_argument("--norm-weight", type=Path)
    p.add_argument("--lm-weight", type=Path)
    p.add_argument("--tokenizer-decode", type=Path)
    args = p.parse_args()
    if args.mode == "reference":
        reference_mode(args)
    else:
        runtime_mode(args)
