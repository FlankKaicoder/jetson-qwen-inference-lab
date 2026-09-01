#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    ("arithmetic", "What is 2 + 3? Reply with only the number."),
    ("chinese", "中国的首都是哪里？只回答城市名。"),
    ("cuda", "In one short sentence, what does CUDA stand for?"),
]


def meminfo():
    values = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = int(value.strip().split()[0]) * 1024
    return values


def checkpoint(stage):
    system = meminfo()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            **system,
            "SwapUsed": system["SwapTotal"] - system["SwapFree"],
        },
        "torch_cuda_allocator": {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        },
        "cuda_mem_info": {"free": free_bytes, "total": total_bytes},
    }


def chat_input(tokenizer, prompt):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(rendered, return_tensors="pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshot = str(Path(args.snapshot).resolve())
    output = Path(args.output).resolve()
    result = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": snapshot,
        "protocol": {
            "batch_size": 1,
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 32,
            "attention_implementation": "eager",
            "purpose": "deterministic functional smoke test, not a performance or quality benchmark",
        },
        "memory_checkpoints": [],
    }

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")

    torch.cuda.reset_peak_memory_stats()
    result["memory_checkpoints"].append(checkpoint("M0_before_model_load"))

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    torch.cuda.synchronize()
    result["memory_checkpoints"].append(checkpoint("M1_after_model_load"))

    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    config = model.config
    result["runtime"] = {
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    result["tokenizer"] = {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_present": bool(tokenizer.chat_template),
    }
    result["model"] = {
        "class": type(model).__name__,
        "model_type": config.model_type,
        "architectures": config.architectures,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_devices": devices,
        "parameter_dtypes": dtypes,
        "hf_device_map": getattr(model, "hf_device_map", None),
    }
    if devices != ["cuda:0"]:
        raise RuntimeError(f"unexpected parameter placement: {devices}")
    if dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"unexpected parameter dtypes: {dtypes}")

    forward_inputs = chat_input(tokenizer, "Hello.")
    forward_inputs = {key: value.to("cuda:0") for key, value in forward_inputs.items()}
    with torch.inference_mode():
        logits = model(**forward_inputs, use_cache=False).logits
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(logits).all().item())
    result["forward"] = {
        "input_tokens": int(forward_inputs["input_ids"].shape[-1]),
        "input_shape": list(forward_inputs["input_ids"].shape),
        "attention_mask_shape": list(forward_inputs["attention_mask"].shape),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "finite": finite,
        "logits_min": float(logits.float().min().item()),
        "logits_max": float(logits.float().max().item()),
    }
    if not finite:
        raise RuntimeError("forward logits contain NaN or Inf")
    del logits
    result["memory_checkpoints"].append(checkpoint("M2_after_forward"))

    result["generations"] = []
    for index, (case, prompt) in enumerate(PROMPTS, start=1):
        inputs = chat_input(tokenizer, prompt)
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        new_ids = generated[0, input_tokens:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        eos_reached = bool(len(new_ids) and int(new_ids[-1]) == tokenizer.eos_token_id)
        generation = {
            "case": case,
            "prompt": prompt,
            "input_tokens": input_tokens,
            "generated_tokens": int(len(new_ids)),
            "stop_reason": "eos" if eos_reached else "max_new_tokens",
            "output": text,
            "nonempty": bool(text),
            "token_ids_valid": bool(torch.all((new_ids >= 0) & (new_ids < len(tokenizer))).item()),
        }
        result["generations"].append(generation)
        if not generation["nonempty"] or not generation["token_ids_valid"]:
            raise RuntimeError(f"invalid generation for {case}")
        result["memory_checkpoints"].append(checkpoint(f"M{index + 2}_after_generation_{index}"))

    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["status"] = "PASS"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
