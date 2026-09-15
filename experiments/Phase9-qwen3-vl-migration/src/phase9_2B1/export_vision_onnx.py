#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLVisionModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class StaticQwen3VLVision(nn.Module):
    """Fixed single-image export wrapper for model.visual."""

    def __init__(self, visual: Qwen3VLVisionModel, grid_thw: torch.Tensor):
        super().__init__()
        if grid_thw.tolist() != [[1, 28, 28]]:
            raise ValueError("This export boundary only supports grid_thw=[1,28,28]")
        self.visual = visual
        deepstack_indexes = list(visual.deepstack_visual_indexes)
        if deepstack_indexes != [5, 11, 17]:
            raise ValueError(f"Unexpected deepstack indexes: {deepstack_indexes}")
        with torch.no_grad():
            pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
            rotary = visual.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary, rotary), dim=-1)
        self.register_buffer("pos_embeds", pos_embeds.detach(), persistent=False)
        self.register_buffer("cos_emb", emb.cos().detach(), persistent=False)
        self.register_buffer("sin_emb", emb.sin().detach(), persistent=False)

    def forward(self, pixel_values: torch.Tensor):
        hidden_states = self.visual.patch_embed(pixel_values)
        hidden_states = hidden_states + self.pos_embeds

        deepstack_outputs = []
        for layer_index, block in enumerate(self.visual.blocks):
            hidden_states = self._block_forward(block, hidden_states)
            if layer_index == 5:
                deepstack_outputs.append(self.visual.deepstack_merger_list[0](hidden_states))
            elif layer_index == 11:
                deepstack_outputs.append(self.visual.deepstack_merger_list[1](hidden_states))
            elif layer_index == 17:
                deepstack_outputs.append(self.visual.deepstack_merger_list[2](hidden_states))

        final_hidden = self.visual.merger(hidden_states)
        return final_hidden, deepstack_outputs[0], deepstack_outputs[1], deepstack_outputs[2]

    def _block_forward(self, block, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = block.norm1(hidden_states)
        hidden_states = residual + self._attention_forward(block.attn, hidden_states)

        residual = hidden_states
        hidden_states = block.norm2(hidden_states)
        return residual + block.mlp(hidden_states)

    def _attention_forward(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = attn.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3)
        query_states, key_states, value_states = qkv.unbind(0)

        query_states, key_states = self._apply_vision_rope(
            query_states, key_states, self.cos_emb, self.sin_emb
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * attn.scaling
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return attn.proj(attn_output)

    @staticmethod
    def _apply_vision_rope(q, k, cos, sin):
        original_q_dtype = q.dtype
        original_k_dtype = k.dtype
        q = q.float()
        k = k.float()
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed.to(original_q_dtype), k_embed.to(original_k_dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16"])
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output_path).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    manifest = {
        "phase": "Phase 9.2-B1",
        "scope": "Qwen3-VL vision encoder ONNX export feasibility only",
        "audit_mode": "FIXED_SINGLE_IMAGE_ONNX_EXPORT_ATTEMPT",
        "export_success": False,
        "onnx_file_written": False,
        "prohibited_operations_performed": {
            "tensorrt_build": False,
            "benchmark": False,
            "quantization": False,
            "optimization": False,
            "cuda_modification": False,
            "environment_modification": False,
        },
    }
    started_at = time.time()
    try:
        import transformers

        manifest["runtime"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device_name": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "dtype": args.dtype,
            "opset_version": args.opset,
            "device": args.device,
        }
        torch.manual_seed(20260916)

        weights_path = model_path / "model.safetensors"
        config_path = model_path / "config.json"
        manifest["model_identity"] = {
            "local_path": str(model_path),
            "weights_path": str(weights_path),
            "weights_size_bytes": weights_path.stat().st_size,
            "weights_sha256": sha256_file(weights_path),
            "config_sha256": sha256_file(config_path),
        }

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device(args.device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)

        config = Qwen3VLConfig.from_pretrained(str(model_path))
        visual_config = config.vision_config
        visual_config._attn_implementation = "eager"
        visual = Qwen3VLVisionModel(config=visual_config)
        state = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual.") :]] = f.get_tensor(key)
        load_result = visual.load_state_dict(state, strict=True)
        visual.to(device=device, dtype=torch.float16)
        visual.eval()

        grid_thw = torch.tensor([[1, 28, 28]], device=device, dtype=torch.long)
        wrapper = StaticQwen3VLVision(visual, grid_thw).to(device=device)
        wrapper.eval()
        pixel_values = torch.linspace(
            -1.0, 1.0, 784 * 1536, device=device, dtype=torch.float16
        ).reshape(784, 1536)

        manifest["boundary"] = {
            "grid_thw": [1, 28, 28],
            "pixel_values_shape": list(pixel_values.shape),
            "pixel_values_dtype": str(pixel_values.dtype),
            "pixel_values_values": "deterministic linspace [-1,1], not an image or correctness workload",
            "sequence_length": 784,
            "merged_sequence_length": 196,
            "output_names": ["final_hidden", "deepstack_0", "deepstack_1", "deepstack_2"],
        }
        manifest["visual_module"] = {
            "class": visual.__class__.__name__,
            "parameter_count": sum(p.numel() for p in visual.parameters()),
            "state_dict_loaded": True,
            "state_dict_missing_keys": list(load_result.missing_keys),
            "state_dict_unexpected_keys": list(load_result.unexpected_keys),
            "blocks": len(visual.blocks),
            "deepstack_indexes": list(visual.deepstack_visual_indexes),
        }
        manifest["wrapper_buffers"] = {
            "pos_embeds": {"shape": list(wrapper.pos_embeds.shape), "dtype": str(wrapper.pos_embeds.dtype)},
            "cos_emb": {"shape": list(wrapper.cos_emb.shape), "dtype": str(wrapper.cos_emb.dtype)},
            "sin_emb": {"shape": list(wrapper.sin_emb.shape), "dtype": str(wrapper.sin_emb.dtype)},
        }
        manifest["memory"] = {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
        }

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (pixel_values,),
                str(output_path),
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=False,
                input_names=["pixel_values"],
                output_names=["final_hidden", "deepstack_0", "deepstack_1", "deepstack_2"],
                dynamic_axes=None,
                training=torch.onnx.TrainingMode.EVAL,
            )
        if not output_path.exists():
            raise RuntimeError("torch.onnx.export returned without creating the output file")

        manifest["export_success"] = True
        manifest["onnx_file_written"] = True
        manifest["onnx_output"] = {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }
    except Exception as exc:
        manifest["export_success"] = False
        manifest["onnx_file_written"] = output_path.exists()
        manifest["export_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if torch.cuda.is_available():
            try:
                device = torch.device(args.device)
                manifest["memory"]["allocated_after_bytes"] = torch.cuda.memory_allocated(device)
                manifest["memory"]["reserved_after_bytes"] = torch.cuda.memory_reserved(device)
                manifest["memory"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                manifest["memory"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            except Exception:
                pass
        manifest["elapsed_seconds"] = round(time.time() - started_at, 6)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))

    if not manifest["export_success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
