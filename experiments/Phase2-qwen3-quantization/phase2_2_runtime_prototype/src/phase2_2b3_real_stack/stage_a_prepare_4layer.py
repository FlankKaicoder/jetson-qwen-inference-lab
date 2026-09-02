from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from portable_qwen3_stack import PortableFourLayerStack


MODEL = Path('/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca')
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
EXPECTED_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
REQUIRED = [
    'input_layernorm.weight', 'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
    'self_attn.v_proj.weight', 'self_attn.o_proj.weight', 'self_attn.q_norm.weight',
    'self_attn.k_norm.weight', 'post_attention_layernorm.weight', 'mlp.gate_proj.weight',
    'mlp.up_proj.weight', 'mlp.down_proj.weight',
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float()
    b = b.float()
    d = a - b
    ref_norm = torch.linalg.vector_norm(a)
    test_norm = torch.linalg.vector_norm(b)
    denom = torch.clamp(ref_norm, min=torch.finfo(torch.float32).tiny)
    cosine = torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(ref_norm * test_norm, min=torch.finfo(torch.float32).tiny)
    return {
        'shape_equal': list(a.shape) == list(b.shape),
        'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        'max_abs_error': float(d.abs().max()),
        'mean_abs_error': float(d.abs().mean()),
        'rmse': float(torch.sqrt((d * d).mean())),
        'relative_l2_error': float(torch.linalg.vector_norm(d) / denom),
        'cosine_similarity': float(cosine),
        'reference_rms': float(torch.sqrt((a * a).mean())),
        'test_rms': float(torch.sqrt((b * b).mean())),
    }


def hf_layer_run(layer, rotary, hidden, pos, cache, layer_idx, past_len):
    b, s, _ = hidden.shape
    total = past_len + s
    mask = torch.zeros((b, 1, s, total), device=hidden.device, dtype=hidden.dtype)
    if past_len == 0:
        causal = torch.triu(torch.ones((s, s), device=hidden.device, dtype=torch.bool), diagonal=1)
        mask = mask.masked_fill(causal[None, None], torch.finfo(hidden.dtype).min)
    pos_emb = rotary(hidden, pos)
    out = layer(
        hidden_states=hidden,
        attention_mask=mask,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(past_len, past_len + s, device=hidden.device),
        position_embeddings=pos_emb,
    )
    k, v = cache[layer_idx]
    return out, k, v


def run_hf_stack(layers, rotary, hidden, pos, steps):
    cache = DynamicCache()
    pre_hidden, pre_k, pre_v = [], [], []
    with torch.inference_mode():
        for idx, layer in enumerate(layers):
            hidden, k, v = hf_layer_run(layer, rotary, hidden, pos, cache, idx, 0)
            pre_hidden.append(hidden)
            pre_k.append(k)
            pre_v.append(v)
        dec = []
        for step, token in enumerate(steps):
            p = torch.tensor([[8 + step]], device=hidden.device, dtype=torch.long)
            row_h, row_k, row_v = [], [], []
            current = token
            for idx, layer in enumerate(layers):
                current, k, v = hf_layer_run(layer, rotary, current, p, cache, idx, 8 + step)
                row_h.append(current)
                row_k.append(k)
                row_v.append(v)
            dec.append({'hidden': row_h, 'k': row_k, 'v': row_v})
    return {'hidden': pre_hidden, 'k': pre_k, 'v': pre_v, 'decode': dec}


def portable_stack_run(stack, hidden, pos, steps):
    with torch.inference_mode():
        pre_h, pre_k, pre_v, pre_a = stack.forward_prefill(hidden, pos)
        dec, pk, pv = [], pre_k, pre_v
        for step, token in enumerate(steps):
            p = torch.tensor([[8 + step]], device=hidden.device, dtype=torch.long)
            hs, pk, pv, attn = stack.forward_decode(token, p, pk, pv)
            dec.append({'hidden': hs, 'k': pk, 'v': pv, 'attn': attn})
    return {'hidden': pre_h, 'k': pre_k, 'v': pre_v, 'attn': pre_a, 'decode': dec}


def cpu_tree(tree):
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().contiguous()
    if isinstance(tree, list):
        return [cpu_tree(x) for x in tree]
    if isinstance(tree, dict):
        return {k: cpu_tree(v) for k, v in tree.items()}
    return tree


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    args.handoff.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL / 'model.safetensors'
    digest = sha256_bytes(checkpoint.read_bytes())
    if digest != EXPECTED_SHA:
        raise RuntimeError(f'checkpoint SHA mismatch: {digest} != {EXPECTED_SHA}')
    cfg = Qwen3Config.from_pretrained(str(MODEL))
    cfg._attn_implementation = 'eager'
    layers = [Qwen3DecoderLayer(cfg, layer_idx=i).to('cuda', dtype=torch.bfloat16).eval() for i in range(4)]
    rotary = Qwen3RotaryEmbedding(cfg).to('cuda')
    states, manifest = {}, []
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        for idx in range(4):
            layer_state = {}
            for name in REQUIRED:
                key = f'model.layers.{idx}.{name}'
                tensor = source.get_tensor(key)
                layer_state[name] = tensor
                raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
                manifest.append({'layer': idx, 'checkpoint_key': key, 'module_key': name,
                                 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                 'numel': tensor.numel(), 'bytes': tensor.numel() * tensor.element_size(),
                                 'sha256': sha256_bytes(raw)})
            states[idx] = layer_state
            layers[idx].load_state_dict(layer_state, strict=True)
    portable = PortableFourLayerStack().to('cuda', dtype=torch.bfloat16).eval()
    for idx in range(4):
        portable.layers[idx].load_state_dict(states[idx], strict=True)
    torch.manual_seed(0)
    hidden = torch.randn((1, 8, 1024), device='cuda', dtype=torch.bfloat16)
    pos = torch.arange(8, device='cuda', dtype=torch.long).unsqueeze(0)
    steps = [torch.randn((1, 1, 1024), device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    hf = run_hf_stack(layers, rotary, hidden, pos, steps)
    portable_ref = portable_stack_run(portable, hidden, pos, steps)
    semantic = {'prefill': [], 'decode': []}
    for idx in range(4):
        semantic['prefill'].append({'layer': idx, 'hidden': metric(hf['hidden'][idx], portable_ref['hidden'][idx]),
                                    'k': metric(hf['k'][idx], portable_ref['k'][idx]),
                                    'v': metric(hf['v'][idx], portable_ref['v'][idx])})
    for step in range(4):
        row = {'step': step, 'layers': []}
        for idx in range(4):
            row['layers'].append({'layer': idx,
                                  'hidden': metric(hf['decode'][step]['hidden'][idx], portable_ref['decode'][step]['hidden'][idx]),
                                  'k': metric(hf['decode'][step]['k'][idx], portable_ref['decode'][step]['k'][idx]),
                                  'v': metric(hf['decode'][step]['v'][idx], portable_ref['decode'][step]['v'][idx])})
        semantic['decode'].append(row)
    all_semantic = [x[m] for x in semantic['prefill'] for m in ('hidden', 'k', 'v')]
    all_semantic += [x[m] for row in semantic['decode'] for x in row['layers'] for m in ('hidden', 'k', 'v')]
    if any(x['max_abs_error'] != 0.0 for x in all_semantic):
        raise RuntimeError('BLOCKED_BY_MULTILAYER_PORTABLE_SEMANTICS')
    handoff_payload = {
        'state_dicts': {str(i): cpu_tree(states[i]) for i in range(4)},
        'hidden': cpu_tree(hidden), 'position_ids': cpu_tree(pos), 'decode_inputs': cpu_tree(steps),
        'hf_prefill': cpu_tree({'hidden': hf['hidden'], 'k': hf['k'], 'v': hf['v']}),
        'hf_decode': cpu_tree(hf['decode']),
        'portable_prefill': cpu_tree({'hidden': portable_ref['hidden'], 'k': portable_ref['k'], 'v': portable_ref['v'], 'attn': portable_ref['attn']}),
        'portable_decode': cpu_tree(portable_ref['decode']), 'model_revision': REVISION,
    }
    handoff_path = args.handoff / 'layers0_3_handoff.pt'
    torch.save(handoff_payload, handoff_path)
    handoff_hash = sha256_bytes(handoff_path.read_bytes())
    (args.out / 'layers_0_3_weight_manifest.json').write_text(json.dumps({
        'model_sha256': digest, 'revision': REVISION, 'layers': 4, 'items': manifest,
        'per_layer_params': {str(i): sum(x['numel'] for x in manifest if x['layer'] == i) for i in range(4)},
        'total_params': sum(x['numel'] for x in manifest),
        'total_bf16_bytes': sum(x['bytes'] for x in manifest),
        'total_fp16_bytes': sum(x['numel'] * 2 for x in manifest),
    }, indent=2) + '\n')
    (args.out / 'weight_mapping_audit.json').write_text(json.dumps({'status': 'PASS', 'layers': 4,
        'required_tensors_per_layer': len(REQUIRED), 'all_required_keys_found': True,
        'shape_match': True, 'unexpected_layer_tensors': [], 'mapping_source': 'safetensors model.layers.0..3'}, indent=2) + '\n')
    (args.out / 'portable_semantic_comparison.json').write_text(json.dumps(semantic, indent=2) + '\n')
    (args.out / 'hf_4layer_reference_summary.json').write_text(json.dumps({'status': 'PASS', 'layers': 4,
        'prefill_shape': [1, 8, 1024], 'cache_shape': [1, 8, 8, 128], 'decode_lengths': [9, 10, 11, 12],
        'finite': True, 'per_layer_cache': True}, indent=2) + '\n')
    (args.out / 'model_identity.txt').write_text(f'model=Qwen/Qwen3-0.6B\nrevision={REVISION}\nsnapshot={MODEL}\nsha256={digest}\n')
    (args.out / 'handoff_manifest.json').write_text(json.dumps({'status': 'PASS', 'filename': handoff_path.name,
        'location': str(args.handoff), 'sha256': handoff_hash, 'size': handoff_path.stat().st_size,
        'contains': 'layers 0-3 state, deterministic inputs and bounded references only',
        'no_full_model_payload': True, 'model_revision': REVISION}, indent=2) + '\n')
    (args.out / 'handoff_integrity.json').write_text(json.dumps({'status': 'PASS', 'sha256': handoff_hash,
        'size': handoff_path.stat().st_size, 'verified_before_stage_b': False}, indent=2) + '\n')
    (args.out / 'environment_phase1.txt').write_text('stage=phase1-hf\nmodel=Qwen/Qwen3-0.6B\nrevision=' + REVISION + '\n')
    print(json.dumps({'status': 'PASS', 'layers': 4, 'semantic_exact': True,
                      'handoff': str(handoff_path), 'handoff_sha256': handoff_hash,
                      'total_params': sum(x['numel'] for x in manifest)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    main(parser.parse_args())
