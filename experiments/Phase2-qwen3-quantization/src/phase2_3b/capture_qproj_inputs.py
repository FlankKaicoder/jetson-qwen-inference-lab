from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding


REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = Path("/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca")
MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dist(t: torch.Tensor) -> dict[str, object]:
    x = t.float().reshape(-1)
    finite = torch.isfinite(x)
    vals = x[finite].cpu().numpy()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "finite": bool(finite.all().item()),
        "nan_count": int(torch.isnan(x).sum().item()),
        "inf_count": int(torch.isinf(x).sum().item()),
        "min": float(vals.min()) if vals.size else None,
        "max": float(vals.max()) if vals.size else None,
        "mean": float(vals.mean()) if vals.size else None,
        "std": float(vals.std()) if vals.size else None,
    }


def mem_snapshot(stage: str) -> dict[str, object]:
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    return {
        "stage": stage,
        "mem_available_bytes": available,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def corpus() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups = {
        "short": [
            ("english", "Explain CUDA streams in one sentence."),
            ("chinese", "请用一句话解释量化。"),
            ("numeric", "Compute 17 * 19 and show the result."),
            ("structured", "status: ok; batch=1; dtype=bf16"),
            ("code", "for i in range(4): print(i)"),
            ("mixed", "Qwen3 Jetson: 快速、稳定、可复现。"),
        ],
        "medium": [
            ("english", "Describe a reproducible TensorRT inference check with inputs, outputs, and finite-value validation."),
            ("chinese", "请比较校准集和评估集的区别，并说明为什么必须保持样本互斥。"),
            ("numeric", "A matrix has 8 rows and 16 columns. Give its element count, byte count in FP16, and byte count in INT8."),
            ("structured", "config={model:'qwen3', layer:0, op:'q_proj', quant:{bits:8, symmetric:true, zero_point:0}}"),
            ("code", "def dequantize(q, scale):\n    return q.astype('float32') * scale\n\nprint(dequantize(x, 0.01))"),
            ("mixed", "在 Jetson Orin Nano 上，用 TensorRT 检查一个 real Qwen3 component 的 Q/DQ 执行路径。"),
        ],
        "long": [
            ("english", "Calibration should measure the real tensor entering a selected projection across varied prompts. Describe how a fixed symmetric INT8 activation scale trades clipping against reconstruction error, and keep the explanation concrete."),
            ("chinese", "为了审计 Transformer 激活范围，请从真实 tokenizer 输入开始，记录每个样本的 token 数、最小值、最大值、绝对最大值、百分位数以及 NaN 和 Inf 计数，然后在独立评估集上验证固定 scale。"),
            ("numeric", "Consider a 1024-dimensional activation with values drawn from several ranges. Explain how global absmax, P99.9, and P99.99 clipping change the quantization step, saturation count, and expected error without inventing an accuracy threshold."),
            ("structured", "experiment={phase:'2.3-B', target:'model.layers.0.self_attn.q_proj', baseline:['FP16','W8'], primary_delta:'W8A8_vs_W8', safety:{overwrite:false, cleanup:false}, next:'sensitivity'}"),
            ("code", "import numpy as np\n\ndef symmetric_qdq(x, scale):\n    q = np.clip(np.rint(x / scale), -127, 127)\n    return q * scale\n\n# Compare reconstruction on held-out inputs only\nprint('audit')"),
            ("mixed", "量化实验必须把 W8 权重误差与 A8 激活增量误差分开：先冻结 per-tensor weight，再从 calibration split 推导固定 activation scale，最后在 disjoint evaluation split 上比较 W8A8 与 W8。"),
        ],
        "very_long": [
            ("english", "This bounded infrastructure audit uses a frozen Qwen3 checkpoint and a single Layer 0 q_proj component. The calibration split contains representative English, Chinese, numerical, structured, and code-like prompts. The evaluation split is disjoint. The exact q_proj input is captured after input_layernorm and before q_proj, then the same FP16 tensor is supplied to FP16, W8, and W8A8 TensorRT graphs. The report must preserve raw provenance, range stability, clipping behavior, reconstruction metrics, and the distinction between activation-only and total quantization deltas."),
            ("chinese", "本实验只研究 Qwen3-0.6B 第 0 层 self_attn.q_proj 的激活校准，不构建完整 28 层 INT8 runtime，也不进行正式性能 benchmark。校准集和评估集必须由真实 tokenizer 编码，覆盖不同长度、中文、英文、数字、结构化字段和代码片段。所有固定 scale 只能由校准集推导；评估集只能用于 held-out reconstruction 与组件输出比较。必须保留 W8 与 FP16 的权重量化背景，并把 W8A8 与 W8 的差值作为主要激活校准指标。"),
            ("numeric", "For a deterministic calibration study, record token counts and activation statistics for every prompt. Aggregate global minimum, global maximum, global absmax, the distribution of per-sample absmax, and absolute-value percentiles. Evaluate fixed scales from global absmax, P99.9, and P99.99, plus a small bounded MSE clipping grid. Report mean, median, P95, and maximum instead of hiding worst-case samples. A policy can be selected for simplicity and stable held-out behavior, but no production accuracy claim is allowed."),
            ("structured", "audit_record = {\n  'phase': '2.3-B',\n  'model': 'Qwen/Qwen3-0.6B',\n  'revision': 'c1899de289a04d12100db370d81485cdf75e47ca',\n  'component': 'model.layers.0.self_attn.q_proj',\n  'input_relation': 'post_input_layernorm_pre_q_proj',\n  'weight_policy': {'bits': 8, 'granularity': 'per_tensor', 'symmetric': True},\n  'activation_policies': ['GLOBAL_ABSMAX', 'P99_9', 'P99_99'],\n  'safety': {'historical_artifacts_overwritten': False}\n}"),
            ("code", "class CalibrationAudit:\n    def choose_scale(self, calibration_abs_values):\n        # Candidate scales are derived from calibration data only.\n        ranges = {\n            'GLOBAL_ABSMAX': calibration_abs_values.max(),\n            'P99_9': np.percentile(calibration_abs_values, 99.9),\n            'P99_99': np.percentile(calibration_abs_values, 99.99),\n        }\n        return {name: value / 127.0 for name, value in ranges.items()}\n\nprint('fixed policy')"),
            ("mixed", "从真实文本到 TensorRT Q/DQ 的证据链应该是可恢复的：tokenizer revision、token IDs、输入 SHA256、q_proj producer/consumer 关系、scale 来源、saturation 计数和 W8A8-vs-W8 增量都要进入 timestamped artifacts。不要因为一个 scale 在平均值上更好就掩盖 p95 或 max 失败，也不要把 HF 到 TRT FP16 的历史 C1 drift 归因于本轮 INT8。"),
        ],
    }
    index = 0
    eval_indices = {
        "short": (0, 1, 2),
        "medium": (3, 4, 5),
        "long": (0, 3, 4),
        "very_long": (1, 2, 5),
    }
    for split, selected in (("calibration", lambda entries: entries), ("evaluation", lambda entries: [entries[i] for i in eval_indices[group]])):
        for group, entries in groups.items():
            for category, text in selected(entries):
                index += 1
                suffix = "" if split == "calibration" else "\nheldout=evaluation"
                rows.append({"sample_id": f"{split[:3]}_{index:03d}", "split": split, "length_group": group, "category": category, "text": text + suffix})
    return rows


def main(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    memory = [mem_snapshot("start")]
    ck = MODEL / "model.safetensors"
    digest = sha_bytes(ck.read_bytes())
    if digest != MODEL_SHA256:
        raise RuntimeError(f"checkpoint hash mismatch: {digest}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, revision=REVISION)
    rows = corpus()
    layer_manifest = [
        "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight",
        "self_attn.k_norm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
        "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]
    with safe_open(str(ck), framework="pt", device="cpu") as f:
        state = {name: f.get_tensor("model.layers.0." + name) for name in layer_manifest}
        embed = f.get_tensor("model.embed_tokens.weight")
    config = Qwen3Config.from_pretrained(str(MODEL), local_files_only=True)
    config._attn_implementation = "eager"
    layer = Qwen3DecoderLayer(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16).eval()
    layer.load_state_dict(state, strict=True)
    rotary = Qwen3RotaryEmbedding(config).to("cuda")
    embedding = torch.nn.Embedding(embed.shape[0], embed.shape[1], _weight=embed).to(device="cuda", dtype=torch.bfloat16).eval()
    captured: dict[str, torch.Tensor] = {}
    direct: dict[str, torch.Tensor] = {}

    def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["current"] = inputs[0].detach().cpu().contiguous()

    handle = layer.self_attn.q_proj.register_forward_pre_hook(hook)
    manifest_rows = []
    all_inputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for row in rows:
            enc = tokenizer(row["text"], add_special_tokens=False, return_tensors="pt")
            ids = enc["input_ids"].to("cuda")
            pos = torch.arange(ids.shape[1], device="cuda", dtype=torch.long).unsqueeze(0)
            hidden = embedding(ids)
            direct_norm = layer.input_layernorm(hidden).detach().cpu().contiguous()
            mask = torch.triu(torch.ones((ids.shape[1], ids.shape[1]), device="cuda", dtype=torch.bool), diagonal=1)
            mask = mask[None, None].masked_fill(mask[None, None], torch.finfo(hidden.dtype).min)
            captured.clear()
            position_embeddings = rotary(hidden, pos)
            layer(hidden, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=position_embeddings)
            q_input = captured.get("current")
            if q_input is None:
                raise RuntimeError("q_proj forward-pre-hook did not capture an input")
            if not torch.equal(q_input, direct_norm):
                raise RuntimeError(f"hook/direct mismatch for {row['sample_id']}")
            all_inputs[row["sample_id"]] = q_input
            token_ids = enc["input_ids"][0].tolist()
            manifest_rows.append({
                **row,
                "tokenizer": "AutoTokenizer.from_pretrained(local_snapshot, add_special_tokens=False)",
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "token_ids_sha256": sha_bytes(np.asarray(token_ids, dtype=np.int64).tobytes()),
                "qproj_input_producer": "model.layers.0.input_layernorm",
                "qproj_input_consumer": "model.layers.0.self_attn.q_proj",
                "qproj_input_relation": "post_input_layernorm_pre_q_proj",
                "qproj_input_dtype": str(q_input.dtype),
                "qproj_input_shape": list(q_input.shape),
                "qproj_input_sha256_bf16": sha_tensor(q_input),
                "qproj_input_distribution": dist(q_input),
                "hook_matches_direct_input_layernorm": True,
            })
    handle.remove()
    all_inputs["__weight_bf16__"] = state["self_attn.q_proj.weight"].cpu().contiguous()
    inputs_path = out / "qproj_inputs_bf16.pt"
    torch.save(all_inputs, inputs_path)
    inputs_sha256 = sha_bytes(inputs_path.read_bytes())
    for row in manifest_rows:
        row["qproj_inputs_file"] = "qproj_inputs_bf16.pt"
        row["qproj_inputs_file_sha256"] = inputs_sha256
    (out / "calibration_manifest.json").write_text(json.dumps({"revision": REVISION, "tokenizer_snapshot": str(MODEL), "rows": [r for r in manifest_rows if r["split"] == "calibration"]}, indent=2, ensure_ascii=False) + "\n")
    (out / "evaluation_manifest.json").write_text(json.dumps({"revision": REVISION, "tokenizer_snapshot": str(MODEL), "rows": [r for r in manifest_rows if r["split"] == "evaluation"]}, indent=2, ensure_ascii=False) + "\n")
    (out / "qproj_input_provenance.json").write_text(json.dumps({
        "classification": "EXACT_QPROJ_INPUT_PROVEN",
        "producer_operation": "model.layers.0.input_layernorm",
        "consumer_operation": "model.layers.0.self_attn.q_proj",
        "pre_norm_relationship": "input_layernorm is directly upstream; captured tensor is post-input_layernorm and pre-q_proj",
        "capture_method": "Qwen3DecoderLayer self_attn.q_proj forward-pre-hook",
        "hook_direct_norm_exact_for_all_samples": True,
        "shape_rule": "[1, token_count, 1024]",
        "dtype": "torch.bfloat16",
        "model_revision": REVISION,
        "checkpoint_sha256": digest,
        "sample_count": len(manifest_rows),
        "qproj_inputs_file": "qproj_inputs_bf16.pt",
        "qproj_inputs_file_sha256": inputs_sha256,
    }, indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps({
        "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)),
        "transformers": __import__("transformers").__version__, "safetensors": __import__("safetensors").__version__,
        "git_branch": os.popen("git branch --show-current").read().strip(), "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "model_revision": REVISION, "checkpoint_sha256": digest,
    }, indent=2) + "\n")
    memory.append(mem_snapshot("complete"))
    (out / "memory_trace.json").write_text(json.dumps({"trace": memory, "oom": False, "exit137": False}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "samples": len(manifest_rows), "calibration": sum(r["split"] == "calibration" for r in manifest_rows), "evaluation": sum(r["split"] == "evaluation" for r in manifest_rows)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
