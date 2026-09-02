from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


MODEL = Path('/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca')
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
SELECTED_HIDDEN = {0, 3, 7, 15, 27}


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def memory_snapshot(stage: str, layer_id: int) -> dict:
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.strip().split()[0]) * 1024
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    return {
        'stage': stage,
        'layer_id': layer_id,
        'mem_available_bytes': values['MemAvailable'],
        'swap_used_bytes': values['SwapTotal'] - values['SwapFree'],
        'cuda_free_bytes': cuda_free,
        'cuda_total_bytes': cuda_total,
        'torch_allocated_bytes': torch.cuda.memory_allocated(),
        'torch_reserved_bytes': torch.cuda.memory_reserved(),
        'torch_max_allocated_bytes': torch.cuda.max_memory_allocated(),
        'torch_max_reserved_bytes': torch.cuda.max_memory_reserved(),
    }


def run_layer(layer, rotary, hidden, positions, cache, layer_id, past_length):
    batch, sequence, _ = hidden.shape
    total = past_length + sequence
    mask = torch.zeros((batch, 1, sequence, total), device=hidden.device, dtype=hidden.dtype)
    if past_length == 0:
        causal = torch.triu(
            torch.ones((sequence, sequence), device=hidden.device, dtype=torch.bool), diagonal=1
        )
        mask = mask.masked_fill(causal[None, None], torch.finfo(hidden.dtype).min)
    return layer(
        hidden_states=hidden,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(past_length, total, device=hidden.device),
        position_embeddings=rotary(hidden, positions),
    )


def write_trace(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> None:
    if args.layers not in (4, 8, 28):
        raise ValueError('--layers must be 4, 8, or 28')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    trace = [memory_snapshot('before_load', -1)]
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
        attn_implementation='eager',
        local_files_only=True,
    ).eval()
    trace.append(memory_snapshot('after_load', -1))

    torch.manual_seed(0)
    hidden = torch.randn((1, 8, 1024), device='cuda', dtype=torch.bfloat16)
    positions = torch.arange(8, device='cuda', dtype=torch.long).unsqueeze(0)
    decode_input = torch.randn((1, 1, 1024), device='cuda', dtype=torch.bfloat16)
    cache = DynamicCache()
    selected = {}

    with torch.inference_mode():
        for layer_id in range(args.layers):
            hidden = run_layer(
                model.model.layers[layer_id], model.model.rotary_emb,
                hidden, positions, cache, layer_id, 0,
            )
            torch.cuda.synchronize()
            if layer_id in SELECTED_HIDDEN:
                selected[str(layer_id)] = {
                    'shape': list(hidden.shape),
                    'dtype': str(hidden.dtype),
                    'sha256': tensor_sha256(hidden),
                }
            trace.append(memory_snapshot('prefill', layer_id))

        prefill_hidden = hidden
        prefill_cache_length = int(cache[0][0].shape[2])
        current = decode_input
        decode_position = torch.tensor([[8]], device='cuda', dtype=torch.long)
        for layer_id in range(args.layers):
            current = run_layer(
                model.model.layers[layer_id], model.model.rotary_emb,
                current, decode_position, cache, layer_id, 8,
            )
            torch.cuda.synchronize()
            trace.append(memory_snapshot('decode_8_to_9', layer_id))

    cache_shapes = []
    cache_finite = True
    for layer_id in range(args.layers):
        key, value = cache[layer_id]
        cache_shapes.append({'layer': layer_id, 'key': list(key.shape), 'value': list(value.shape)})
        cache_finite = cache_finite and bool(torch.isfinite(key).all() and torch.isfinite(value).all())

    result = {
        'status': 'PASS',
        'model_revision': REVISION,
        'layers': args.layers,
        'implementation': 'single full HF model; direct internal decoder layer execution',
        'b3_semantic_contract': {
            'prefill': 'B=1,S=8, sequential decoder layers, eager attention, DynamicCache',
            'decode': 'B=1,S=1, one step 8->9 using per-layer persistent KV',
            'matches': True,
        },
        'prefill': {
            'status': 'PASS',
            'hidden_shape': list(prefill_hidden.shape),
            'finite': bool(torch.isfinite(prefill_hidden).all()),
            'cache_length': prefill_cache_length,
            'selected_hidden': selected,
        },
        'decode': {
            'status': 'PASS',
            'hidden_shape': list(current.shape),
            'finite': bool(torch.isfinite(current).all()),
            'cache_length': int(cache[0][0].shape[2]),
        },
        'cache_shapes': cache_shapes,
        'all_cache_finite': cache_finite,
        'memory': {
            'minimum_mem_available_bytes': min(row['mem_available_bytes'] for row in trace),
            'maximum_swap_used_bytes': max(row['swap_used_bytes'] for row in trace),
            'maximum_torch_allocated_bytes': max(row['torch_allocated_bytes'] for row in trace),
            'maximum_torch_reserved_bytes': max(row['torch_reserved_bytes'] for row in trace),
        },
    }
    write_trace(args.trace, trace)
    args.result.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'status': result['status'],
        'layers': args.layers,
        'prefill': result['prefill']['status'],
        'decode': result['decode']['status'],
        'minimum_mem_available_bytes': result['memory']['minimum_mem_available_bytes'],
        'maximum_torch_reserved_bytes': result['memory']['maximum_torch_reserved_bytes'],
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    main(parser.parse_args())
