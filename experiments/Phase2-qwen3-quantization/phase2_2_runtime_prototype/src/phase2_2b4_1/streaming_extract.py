from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


MODEL = Path('/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca')
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
EXPECTED_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
REQUIRED = [
    'input_layernorm.weight',
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight',
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'self_attn.q_norm.weight',
    'self_attn.k_norm.weight',
    'post_attention_layernorm.weight',
    'mlp.gate_proj.weight',
    'mlp.up_proj.weight',
    'mlp.down_proj.weight',
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def memory_snapshot() -> dict:
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.strip().split()[0]) * 1024
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    return {
        'mem_available_bytes': values['MemAvailable'],
        'swap_used_bytes': values['SwapTotal'] - values['SwapFree'],
        'cuda_free_bytes': cuda_free,
        'cuda_total_bytes': cuda_total,
        'torch_allocated_bytes': torch.cuda.memory_allocated(),
        'torch_reserved_bytes': torch.cuda.memory_reserved(),
    }


def main(args: argparse.Namespace) -> None:
    args.handoff.mkdir(parents=True, exist_ok=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL / 'model.safetensors'
    model_digest = sha256_file(checkpoint)
    if model_digest != EXPECTED_SHA:
        raise RuntimeError(f'checkpoint SHA mismatch: {model_digest}')

    before = memory_snapshot()
    layers = []
    for layer_id in range(28):
        state = {}
        tensor_rows = []
        with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
            for name in REQUIRED:
                checkpoint_key = f'model.layers.{layer_id}.{name}'
                tensor = source.get_tensor(checkpoint_key).cpu().contiguous()
                state[name] = tensor
                tensor_rows.append({
                    'checkpoint_key': checkpoint_key,
                    'module_key': name,
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'numel': tensor.numel(),
                    'bytes': tensor.numel() * tensor.element_size(),
                    'sha256': tensor_sha256(tensor),
                })

        layer_path = args.handoff / f'layer_{layer_id:02d}.pt'
        torch.save(state, layer_path)
        row = {
            'layer_id': layer_id,
            'file': layer_path.name,
            'file_size': layer_path.stat().st_size,
            'file_sha256': sha256_file(layer_path),
            'tensor_keys': [item['module_key'] for item in tensor_rows],
            'tensors': tensor_rows,
            'tensor_bytes': sum(item['bytes'] for item in tensor_rows),
            'memory_after_save': memory_snapshot(),
        }
        layers.append(row)
        print(json.dumps({'layer': layer_id, 'file_size': row['file_size'],
                          'mem_available_bytes': row['memory_after_save']['mem_available_bytes']}), flush=True)
        del tensor, state, tensor_rows
        gc.collect()

    after = memory_snapshot()
    manifest = {
        'status': 'PASS',
        'model_revision': REVISION,
        'model_sha256': model_digest,
        'handoff_directory': str(args.handoff),
        'layer_count': len(layers),
        'total_file_bytes': sum(row['file_size'] for row in layers),
        'total_tensor_bytes': sum(row['tensor_bytes'] for row in layers),
        'all_file_sha256_present': all(len(row['file_sha256']) == 64 for row in layers),
        'before': before,
        'after': after,
        'minimum_mem_available_bytes': min(
            [before['mem_available_bytes'], after['mem_available_bytes']]
            + [row['memory_after_save']['mem_available_bytes'] for row in layers]
        ),
        'layers': layers,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({
        'status': 'PASS',
        'layers': len(layers),
        'total_file_bytes': manifest['total_file_bytes'],
        'minimum_mem_available_bytes': manifest['minimum_mem_available_bytes'],
    }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    main(parser.parse_args())
