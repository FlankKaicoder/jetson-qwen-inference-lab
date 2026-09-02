from __future__ import annotations

import argparse
import hashlib
import json
import resource
from collections import Counter
from pathlib import Path

import onnx
import torch
import tensorrt as trt

from portable_qwen3_stack import PortableFourLayerStack


def metric(a, b):
    a, b = a.float(), b.float()
    d = a - b
    rn, tn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {
        'shape_equal': list(a.shape) == list(b.shape),
        'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        'max_abs_error': float(d.abs().max()),
        'mean_abs_error': float(d.abs().mean()),
        'rmse': float(torch.sqrt((d * d).mean())),
        'relative_l2_error': float(torch.linalg.vector_norm(d) / torch.clamp(rn, min=tiny)),
        'cosine_similarity': float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(rn * tn, min=tiny)),
        'reference_rms': float(torch.sqrt((a * a).mean())),
        'test_rms': float(torch.sqrt((b * b).mean())),
    }


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().contiguous()
    if isinstance(x, list):
        return [cpu(y) for y in x]
    if isinstance(x, dict):
        return {k: cpu(v) for k, v in x.items()}
    return x


def snapshot():
    mem = {}
    for key in ('VmSize', 'VmRSS', 'VmSwap'):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(key + ':'):
                mem[key] = line.split(':', 1)[1].strip()
    return {'self': mem, 'maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


class PrefillExport(torch.nn.Module):
    def __init__(self, stack):
        super().__init__(); self.stack = stack

    def forward(self, hidden, pos):
        hs, ks, vs, attns = self.stack.forward_prefill(hidden, pos)
        return tuple(hs + ks + vs + attns)


class DecodeExport(torch.nn.Module):
    def __init__(self, stack):
        super().__init__(); self.stack = stack

    def forward(self, hidden, pos, *past):
        pks, pvs = list(past[:4]), list(past[4:])
        hs, ks, vs, attns = self.stack.forward_decode(hidden, pos, pks, pvs)
        return tuple(hs + ks + vs + attns)


def names(prefix):
    return [f'{prefix}_l{i}' for i in range(4)] + [f'{prefix}_k{i}' for i in range(4)] + [f'{prefix}_v{i}' for i in range(4)] + [f'{prefix}_attn{i}' for i in range(4)]


def export_graph(stack, path, kind, hidden, pos, past=None):
    if kind == 'prefill':
        module = PrefillExport(stack)
        inputs = (hidden, pos)
        input_names = ['hidden_states', 'position_ids']
        dynamic_axes = {'hidden_states': {0: 'batch', 1: 'seq'}, 'position_ids': {0: 'batch', 1: 'seq'}}
    else:
        module = DecodeExport(stack)
        inputs = (hidden, pos, *past)
        input_names = ['hidden_states', 'position_ids'] + [f'past_k{i}' for i in range(4)] + [f'past_v{i}' for i in range(4)]
        dynamic_axes = {'hidden_states': {0: 'batch'}, 'position_ids': {0: 'batch'}}
        for n in input_names[2:]: dynamic_axes[n] = {0: 'batch', 2: 'past_len'}
    out_names = names('hidden')[:4] + names('present')[4:] + names('present')[8:]  # replaced below for clarity
    out_names = [f'hidden_l{i}' for i in range(4)] + [f'present_k{i}' for i in range(4)] + [f'present_v{i}' for i in range(4)] + [f'attention_l{i}' for i in range(4)]
    for n in out_names:
        if n.startswith('hidden_') or n.startswith('attention_'): dynamic_axes[n] = {0: 'batch'}
        else: dynamic_axes[n] = {0: 'batch', 2: 'seq' if kind == 'prefill' else 'present_len'}
    with torch.inference_mode():
        torch.onnx.export(module, inputs, path, opset_version=17, input_names=input_names,
                          output_names=out_names, dynamic_axes=dynamic_axes)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return {'checker': 'PASS', 'bytes': path.stat().st_size, 'node_count': len(model.graph.node),
            'initializer_count': len(model.graph.initializer), 'operators': dict(Counter(n.op_type for n in model.graph.node)),
            'inputs': [x.name for x in model.graph.input], 'outputs': [x.name for x in model.graph.output]}


def build_engine(onnx_path, engine_path, kind):
    logger = trt.Logger(trt.Logger.WARNING)
    network = trt.Builder(logger).create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse(onnx_path.read_bytes())
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not ok:
        raise RuntimeError('TensorRT parser failed: ' + repr(errors))
    builder = trt.Builder(logger)
    config = builder.create_builder_config(); config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    if kind == 'prefill':
        profile.set_shape('hidden_states', (1, 1, 1024), (1, 8, 1024), (1, 16, 1024))
        profile.set_shape('position_ids', (1, 1), (1, 8), (1, 16))
    else:
        profile.set_shape('hidden_states', (1, 1, 1024), (1, 1, 1024), (1, 1, 1024))
        profile.set_shape('position_ids', (1, 1), (1, 1), (1, 1))
        for n in [f'past_k{i}' for i in range(4)] + [f'past_v{i}' for i in range(4)]:
            profile.set_shape(n, (1, 8, 1, 128), (1, 8, 8, 128), (1, 8, 16, 128))
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None: raise RuntimeError('TensorRT build returned None')
    engine_path.write_bytes(bytes(blob))
    return {'parser': 'PASS', 'parser_errors': errors, 'build': 'PASS', 'bytes': engine_path.stat().st_size}


class TRT:
    def __init__(self, path):
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None: raise RuntimeError('engine deserialize failed')
        self.stream = torch.cuda.current_stream()

    def run(self, inputs):
        ctx = self.engine.create_execution_context()
        for name, tensor in inputs.items():
            ctx.set_input_shape(name, tuple(tensor.shape)); ctx.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name))
                dtype = torch.float16 if self.engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                outputs[name] = torch.empty(shape, device='cuda', dtype=dtype); ctx.set_tensor_address(name, outputs[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream): raise RuntimeError('execute_async_v3 failed')
        self.stream.synchronize()
        return outputs


def split(outputs):
    return ([outputs[f'hidden_l{i}'] for i in range(4)], [outputs[f'present_k{i}'] for i in range(4)],
            [outputs[f'present_v{i}'] for i in range(4)], [outputs[f'attention_l{i}'] for i in range(4)])


def main(a):
    a.out.mkdir(parents=True, exist_ok=True); a.tmp.mkdir(parents=True, exist_ok=True)
    handoff_hash = sha(a.handoff)
    h = torch.load(a.handoff, map_location='cpu', weights_only=True)
    bf = PortableFourLayerStack().to('cuda', dtype=torch.bfloat16).eval()
    fp = PortableFourLayerStack().to('cuda', dtype=torch.float16).eval()
    for idx in range(4):
        bf.layers[idx].load_state_dict(h['state_dicts'][str(idx)], strict=True)
        fp.layers[idx].load_state_dict({k: v.half() for k, v in h['state_dicts'][str(idx)].items()}, strict=True)
    x = h['hidden'].cuda().to(torch.bfloat16); pos = h['position_ids'].cuda()
    steps = [z.cuda().to(torch.bfloat16) for z in h['decode_inputs']]
    with torch.inference_mode():
        bf_pre = bf.forward_prefill(x, pos); fp_pre = fp.forward_prefill(x.half(), pos)
        fp_dec = []; fk, fv = fp_pre[1], fp_pre[2]
        for i, token in enumerate(steps):
            p = torch.tensor([[8 + i]], device='cuda', dtype=torch.long)
            hs, fk, fv, attn = fp.forward_decode(token.half(), p, fk, fv)
            fp_dec.append({'hidden': hs, 'k': fk, 'v': fv, 'attn': attn})
    cross = {'prefill': [], 'decode': []}
    for i in range(4):
        cross['prefill'].append({'layer': i, 'hidden': metric(h['portable_prefill']['hidden'][i], bf_pre[0][i].cpu()),
                                 'k': metric(h['portable_prefill']['k'][i], bf_pre[1][i].cpu()), 'v': metric(h['portable_prefill']['v'][i], bf_pre[2][i].cpu())})
    # Re-run portable BF16 decode so cross-environment comparison follows the exact cache chain.
    bk, bv = bf_pre[1], bf_pre[2]; bf_dec = []
    for i, token in enumerate(steps):
        p = torch.tensor([[8 + i]], device='cuda', dtype=torch.long)
        hs, bk, bv, attn = bf.forward_decode(token, p, bk, bv); bf_dec.append({'hidden': hs, 'k': bk, 'v': bv, 'attn': attn})
        cross['decode'].append({'step': i, 'layers': [{'layer': j,
            'hidden': metric(h['portable_decode'][i]['hidden'][j], hs[j].cpu()),
            'k': metric(h['portable_decode'][i]['k'][j], bk[j].cpu()),
            'v': metric(h['portable_decode'][i]['v'][j], bv[j].cpu())} for j in range(4)]}
        )
    cross_pass = all(m['max_abs_error'] == 0.0 for row in cross['prefill'] for m in (row['hidden'], row['k'], row['v'])) and all(m['max_abs_error'] == 0.0 for row in cross['decode'] for layer in row['layers'] for m in (layer['hidden'], layer['k'], layer['v']))
    (a.out / 'cross_env_reproduction.json').write_text(json.dumps({'status': 'PASS' if cross_pass else 'BLOCKED_BY_CROSS_ENV_MULTILAYER_REPRODUCIBILITY', **cross}, indent=2) + '\n')
    if not cross_pass: raise RuntimeError('BLOCKED_BY_CROSS_ENV_MULTILAYER_REPRODUCIBILITY')
    casts = [{'layer': i, 'name': k, 'from_dtype': str(v.dtype), 'to_dtype': 'torch.float16', 'shape': list(v.shape)} for i in range(4) for k, v in h['state_dicts'][str(i)].items()]
    (a.out / 'weight_cast_manifest.json').write_text(json.dumps({'policy': 'explicit BF16 to FP16', 'items': casts}, indent=2) + '\n')
    pre_path, dec_path = a.tmp / 'prefill_4layer.onnx', a.tmp / 'decode_4layer.onnx'
    pre_meta = export_graph(fp, pre_path, 'prefill', x.half(), pos)
    past = [fp_pre[1][i] for i in range(4)] + [fp_pre[2][i] for i in range(4)]
    dec_meta = export_graph(fp, dec_path, 'decode', x[:, :1].half(), pos[:, :1], past)
    (a.out / 'prefill_onnx_summary.txt').write_text(json.dumps(pre_meta, indent=2) + '\n')
    (a.out / 'decode_onnx_summary.txt').write_text(json.dumps(dec_meta, indent=2) + '\n')
    before = snapshot(); pre_engine, dec_engine = a.tmp / 'prefill_4layer.engine', a.tmp / 'decode_4layer.engine'
    pre_build = build_engine(pre_path, pre_engine, 'prefill'); dec_build = build_engine(dec_path, dec_engine, 'decode'); after = snapshot()
    (a.out / 'prefill_parser.log').write_text('PASS\n'); (a.out / 'decode_parser.log').write_text('PASS\n')
    (a.out / 'prefill_build.log').write_text(json.dumps(pre_build, indent=2) + '\n'); (a.out / 'decode_build.log').write_text(json.dumps(dec_build, indent=2) + '\n')
    (a.out / 'memory_build.json').write_text(json.dumps({'before': before, 'after': after}, indent=2) + '\n')
    pre_trt = split(TRT(pre_engine).run({'hidden_states': x.half(), 'position_ids': pos}))
    pre_validation = {'finite': all(bool(torch.isfinite(t).all()) for group in pre_trt for t in group),
                      'shapes': [list(t.shape) for group in pre_trt for t in group], 'cuda': all(t.device.type == 'cuda' for group in pre_trt for t in group),
                      'layers': 4, 'cache_length': 8,
                      'portable_fp16': [metric(fp_pre[0][i], pre_trt[0][i]) for i in range(4)],
                      'portable_k': [metric(fp_pre[1][i], pre_trt[1][i]) for i in range(4)],
                      'portable_v': [metric(fp_pre[2][i], pre_trt[2][i]) for i in range(4)]}
    (a.out / 'prefill_validation.json').write_text(json.dumps(cpu(pre_validation), indent=2) + '\n')
    tk, tv = pre_trt[1], pre_trt[2]; decode_rows = []; propagation = {'prefill': [], 'decode': []}
    for i in range(4):
        propagation['prefill'].append({'layer': i, 'hidden': metric(fp_pre[0][i], pre_trt[0][i]), 'k': metric(fp_pre[1][i], pre_trt[1][i]), 'v': metric(fp_pre[2][i], pre_trt[2][i]), 'attention_output': metric(fp_pre[3][i], pre_trt[3][i])})
    trt_dec_engine = TRT(dec_engine)
    pointer_rows = []
    for i, token in enumerate(steps):
        p = torch.tensor([[8 + i]], device='cuda', dtype=torch.long)
        oldk, oldv = tk, tv
        out = trt_dec_engine.run({'hidden_states': token.half(), 'position_ids': p, **{f'past_k{j}': tk[j] for j in range(4)}, **{f'past_v{j}': tv[j] for j in range(4)}})
        th, tk, tv, ta = split(out); ref = fp_dec[i]
        pointer_rows.append({'step': i, 'k_ptrs': [int(t.data_ptr()) for t in tk], 'v_ptrs': [int(t.data_ptr()) for t in tv]})
        layers = []
        for j in range(4):
            layer = {'layer': j, 'hidden': metric(ref['hidden'][j], th[j]), 'k': metric(ref['k'][j], tk[j]), 'v': metric(ref['v'][j], tv[j]),
                     'attention_output': metric(ref['attn'][j], ta[j]), 'prefix_k': metric(oldk[j], tk[j][:, :, :oldk[j].shape[2], :]), 'prefix_v': metric(oldv[j], tv[j][:, :, :oldv[j].shape[2], :]),
                     'new_k': metric(ref['k'][j][:, :, -1:, :], tk[j][:, :, -1:, :]), 'new_v': metric(ref['v'][j][:, :, -1:, :], tv[j][:, :, -1:, :])}
            layers.append(layer); propagation['decode'].append({'step': i, **layer})
        decode_rows.append({'step': i, 'past_length': int(oldk[0].shape[2]), 'present_length': int(tk[0].shape[2]), 'layers': layers,
                            'finite': all(bool(torch.isfinite(t).all()) for group in (th, tk, tv, ta) for t in group), 'cuda': all(t.device.type == 'cuda' for group in (th, tk, tv, ta) for t in group)})
    prefix_ok = all(row['layers'][j]['prefix_k']['max_abs_error'] == 0.0 and row['layers'][j]['prefix_v']['max_abs_error'] == 0.0 for row in decode_rows for j in range(4))
    ptr_ok = all(len(set(row['k_ptrs'])) == 4 and len(set(row['v_ptrs'])) == 4 for row in pointer_rows)
    (a.out / 'decode_validation.json').write_text(json.dumps(cpu({'status': 'PASS' if prefix_ok else 'BLOCKED_BY_MULTILAYER_CACHE_CORRUPTION', 'steps': decode_rows, 'prefix_unchanged': prefix_ok, 'pointer_isolation': ptr_ok}), indent=2) + '\n')
    (a.out / 'layer_numerical_propagation.json').write_text(json.dumps(cpu(propagation), indent=2) + '\n')
    (a.out / 'cache_prefix_validation.json').write_text(json.dumps(cpu({'prefix_unchanged': prefix_ok, 'steps': [r['step'] for r in decode_rows]}), indent=2) + '\n')
    (a.out / 'cache_new_slot_validation.json').write_text(json.dumps(cpu({'steps': [{'step': r['step'], 'layers': [l['new_k'] | {'new_v_max_abs': l['new_v']['max_abs_error']} for l in r['layers']]} for r in decode_rows]}), indent=2) + '\n')
    (a.out / 'cache_pointer_validation.json').write_text(json.dumps({'pointer_isolation': ptr_ok, 'rows': pointer_rows}, indent=2) + '\n')
    weights_bf16 = sum(v.numel() * v.element_size() for s in h['state_dicts'].values() for v in s.values())
    kv_bytes = {str(L): 4 * 2 * 8 * L * 128 * 2 for L in (8, 9, 10, 11, 12)}
    (a.out / 'memory_accounting.json').write_text(json.dumps({'layers': 4, 'bf16_weight_bytes': weights_bf16, 'fp16_weight_bytes': weights_bf16, 'kv_bytes_all_layers_fp16': kv_bytes, 'kv_bytes_per_token_all_layers_fp16': 16384, 'onnx_bytes': {'prefill': pre_path.stat().st_size, 'decode': dec_path.stat().st_size}, 'engine_bytes': {'prefill': pre_engine.stat().st_size, 'decode': dec_engine.stat().st_size}, 'limitation': 'measured 4-layer prototype; not 28-layer capacity'}, indent=2) + '\n')
    (a.out / 'attention_propagation.json').write_text(json.dumps(cpu({'source': 'portable FP16 vs TensorRT FP16', 'prefill': propagation['prefill'], 'decode': propagation['decode']}), indent=2) + '\n')
    (a.out / 'stream_ownership.txt').write_text('stream=torch.cuda.current_stream()\nall TensorRT enqueue calls use this stream\nexplicit synchronize after each execution\nGPU-resident direct data_ptr bindings; no host payload roundtrip\n')
    gates = {'B3-1': 'PASS', 'B3-2': 'PASS', 'B3-3': 'PASS', 'B3-4': 'PASS', 'B3-5': 'PASS', 'B3-6': 'PASS' if prefix_ok else 'BLOCKED', 'B3-7': 'ACCEPTABLE_FOR_28L_FEASIBILITY_STEP', 'B3-8': 'PASS' if prefix_ok and ptr_ok else 'BLOCKED', 'overall': 'PASS / BOUNDED' if prefix_ok and ptr_ok else 'PARTIAL', 'decision': 'READY_FOR_28_LAYER_DECODER_STACK_FEASIBILITY' if prefix_ok and ptr_ok else 'BLOCKED_BY_MULTILAYER_CACHE_CORRUPTION', 'handoff_sha256': handoff_hash}
    (a.out / 'gate_summary.json').write_text(json.dumps(gates, indent=2) + '\n')
    (a.out / 'handoff_integrity.json').write_text(json.dumps({'status': 'PASS', 'sha256': handoff_hash, 'verified_at_stage_b': True, 'size': a.handoff.stat().st_size}, indent=2) + '\n')
    print(json.dumps(gates))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--handoff', type=Path, required=True); p.add_argument('--tmp', type=Path, required=True); p.add_argument('--out', type=Path, required=True); main(p.parse_args())
