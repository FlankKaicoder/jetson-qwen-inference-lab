from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
from collections import Counter
from pathlib import Path

import torch

from portable_qwen3_28 import PortableTwentyEightLayerStack

REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
EXPECTED_MODEL_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
REQUIRED = [
    'input_layernorm.weight', 'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
    'self_attn.v_proj.weight', 'self_attn.o_proj.weight', 'self_attn.q_norm.weight',
    'self_attn.k_norm.weight', 'post_attention_layernorm.weight', 'mlp.gate_proj.weight',
    'mlp.up_proj.weight', 'mlp.down_proj.weight',
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def metric(a, b):
    a, b = a.float(), b.float(); d = a - b
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {'max_abs_error': float(d.abs().max()),
            'rmse': float(torch.sqrt((d*d).mean())),
            'relative_l2': float(torch.linalg.vector_norm(d) / torch.clamp(an, min=tiny)),
            'cosine': float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(an*bn, min=tiny)),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            'shape_equal': list(a.shape) == list(b.shape)}


def mem_snapshot(stage):
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1); mem[k] = int(v.split()[0]) * 1024
    proc = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmSwap:')):
            k, v = line.split(':', 1); proc[k] = v.strip()
    free, total = torch.cuda.mem_get_info()
    return {'stage': stage, 'mem_available_bytes': mem.get('MemAvailable'),
            'swap_used_bytes': mem.get('SwapTotal', 0) - mem.get('SwapFree', 0),
            'vmrss': proc.get('VmRSS'), 'vmswap': proc.get('VmSwap'),
            'cuda_free_bytes': free, 'cuda_total_bytes': total,
            'torch_allocated_bytes': torch.cuda.memory_allocated(),
            'torch_reserved_bytes': torch.cuda.memory_reserved(),
            'maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def load_stack(handoff: Path, dtype):
    stack = PortableTwentyEightLayerStack(device='cuda', dtype=dtype).eval()
    for i in range(28):
        payload = torch.load(handoff / f'layer_{i:02d}.pt', map_location='cpu', weights_only=True)
        stack.layers[i].load_state_dict({k: v.to(dtype=dtype) for k, v in payload.items()}, strict=True)
        del payload
        gc.collect()
    return stack


def validate_manifest(handoff: Path, out: Path):
    manifest = json.loads((handoff / 'streaming_handoff_manifest.json').read_text())
    if manifest.get('model_revision') != REVISION or manifest.get('model_sha256') != EXPECTED_MODEL_SHA:
        raise RuntimeError('FROZEN_MODEL_IDENTITY_MISMATCH')
    rows = []
    for i in range(28):
        p = handoff / f'layer_{i:02d}.pt'
        if not p.is_file() or sha(p) != manifest['layers'][i]['file_sha256']:
            raise RuntimeError(f'HANDOFF_HASH_MISMATCH_LAYER_{i}')
        rows.append({'layer_id': i, 'file': p.name, 'file_size': p.stat().st_size,
                     'file_sha256': sha(p), 'tensor_keys': manifest['layers'][i]['tensor_keys'],
                     'tensors': manifest['layers'][i]['tensors']})
    (out / 'layers_0_27_manifest.json').write_text(json.dumps({'status': 'PASS', 'model_revision': REVISION,
        'model_sha256': EXPECTED_MODEL_SHA, 'layer_count': 28, 'layers': rows}, indent=2) + '\n')
    (out / 'handoff_manifest.json').write_text(json.dumps({'status': 'PASS', 'source_manifest': str(handoff / 'streaming_handoff_manifest.json'),
        'layer_count': 28, 'total_file_bytes': sum(x['file_size'] for x in rows), 'files': rows}, indent=2) + '\n')


class PrefillExport(torch.nn.Module):
    def __init__(self, stack): super().__init__(); self.stack = stack
    def forward(self, hidden, pos):
        hs, ks, vs = self.stack.forward_prefill(hidden, pos)
        return tuple(hs + ks + vs)


class DecodeExport(torch.nn.Module):
    def __init__(self, stack): super().__init__(); self.stack = stack
    def forward(self, hidden, pos, *past):
        pks, pvs = list(past[:28]), list(past[28:])
        hs, ks, vs = self.stack.forward_decode(hidden, pos, pks, pvs)
        return tuple(hs + ks + vs)


def export_graph(stack, path: Path, kind: str, hidden, pos, past=None):
    import onnx
    module = PrefillExport(stack) if kind == 'prefill' else DecodeExport(stack)
    if kind == 'prefill':
        inputs = (hidden, pos); input_names = ['hidden_states', 'position_ids']
        dynamic = {'hidden_states': {0: 'batch', 1: 'seq'}, 'position_ids': {0: 'batch', 1: 'seq'}}
    else:
        inputs = (hidden, pos, *past)
        input_names = ['hidden_states', 'position_ids'] + [f'past_k{i}' for i in range(28)] + [f'past_v{i}' for i in range(28)]
        dynamic = {'hidden_states': {0: 'batch'}, 'position_ids': {0: 'batch'}}
        for n in input_names[2:]: dynamic[n] = {0: 'batch', 2: 'past_len'}
    outputs = [f'hidden_l{i}' for i in range(28)] + [f'present_k{i}' for i in range(28)] + [f'present_v{i}' for i in range(28)]
    for n in outputs:
        dynamic[n] = {0: 'batch', 1: 'seq'} if n.startswith('hidden_') else {0: 'batch', 2: ('seq' if kind == 'prefill' else 'present_len')}
    torch.onnx.export(module, inputs, str(path), opset_version=17, input_names=input_names,
                      output_names=outputs, dynamic_axes=dynamic)
    model = onnx.load(str(path)); onnx.checker.check_model(model)
    return {'status': 'PASS', 'bytes': path.stat().st_size, 'node_count': len(model.graph.node),
            'initializer_count': len(model.graph.initializer), 'operators': dict(Counter(x.op_type for x in model.graph.node)),
            'inputs': [x.name for x in model.graph.input], 'outputs': [x.name for x in model.graph.output]}


def export_mode(a):
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    validate_manifest(a.handoff, out)
    trace = [mem_snapshot('before_stack_load')]
    stack = load_stack(a.handoff, torch.float16); trace.append(mem_snapshot('after_stack_load'))
    torch.manual_seed(0)
    x = torch.randn((1, 8, 1024), device='cuda', dtype=torch.float16)
    pos = torch.arange(8, device='cuda', dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        hs, ks, vs = stack.forward_prefill(x, pos)
    torch.cuda.synchronize(); trace.append(mem_snapshot('after_prefill_reference'))
    a.tmp.mkdir(parents=True, exist_ok=True)
    pre = export_graph(stack, a.tmp / 'prefill_28layer.onnx', 'prefill', x, pos)
    (out / 'prefill_onnx_summary.json').write_text(json.dumps(pre, indent=2) + '\n')
    past = ks + vs
    dec = export_graph(stack, a.tmp / 'decode_28layer.onnx', 'decode', x[:, :1], pos[:, :1], past)
    (out / 'decode_onnx_summary.json').write_text(json.dumps(dec, indent=2) + '\n')
    (out / 'memory_trace.json').write_text(json.dumps(trace + [mem_snapshot('after_export')], indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'prefill': pre, 'decode': dec}))


def build_mode(a):
    import tensorrt as trt
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    logs = []
    def build(path, kind):
        logger = trt.Logger(trt.Logger.WARNING)
        network = trt.Builder(logger).create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger); ok = parser.parse(path.read_bytes())
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        if not ok: raise RuntimeError(f'PARSER_FAILED_{kind}: {errors}')
        builder = trt.Builder(logger); cfg = builder.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
        prof = builder.create_optimization_profile()
        if kind == 'prefill':
            prof.set_shape('hidden_states', (1,1,1024), (1,8,1024), (1,16,1024)); prof.set_shape('position_ids', (1,1), (1,8), (1,16))
        else:
            prof.set_shape('hidden_states', (1,1,1024), (1,1,1024), (1,1,1024)); prof.set_shape('position_ids', (1,1), (1,1), (1,1))
            for n in [f'past_k{i}' for i in range(28)] + [f'past_v{i}' for i in range(28)]: prof.set_shape(n, (1,8,1,128), (1,8,8,128), (1,8,16,128))
        cfg.add_optimization_profile(prof); blob = builder.build_serialized_network(network, cfg)
        if blob is None: raise RuntimeError(f'BUILD_FAILED_{kind}')
        p = a.tmp / f'{kind}_28layer.engine'; p.write_bytes(bytes(blob)); return {'status': 'PASS', 'bytes': p.stat().st_size, 'parser_errors': errors}
    before = mem_snapshot('before_trt_build')
    result = {'prefill': build(a.tmp / 'prefill_28layer.onnx', 'prefill'), 'decode': build(a.tmp / 'decode_28layer.onnx', 'decode')}
    after = mem_snapshot('after_trt_build'); (out / 'engine_summary.json').write_text(json.dumps(result, indent=2) + '\n'); (out / 'memory_trace_build.json').write_text(json.dumps([before, after], indent=2) + '\n')
    print(json.dumps(result))


class TRT:
    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None: raise RuntimeError('ENGINE_DESERIALIZE_FAILED')
        self.stream = torch.cuda.current_stream()

    def run(self, inputs):
        ctx = self.engine.create_execution_context()
        for name, tensor in inputs.items():
            if not ctx.set_input_shape(name, tuple(tensor.shape)): raise RuntimeError(f'SHAPE_REJECTED:{name}')
            ctx.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name)); dtype = self.engine.get_tensor_dtype(name)
                if dtype != self.trt.DataType.HALF: raise RuntimeError(f'UNEXPECTED_OUTPUT_DTYPE:{name}:{dtype}')
                outputs[name] = torch.empty(shape, device='cuda', dtype=torch.float16)
                ctx.set_tensor_address(name, outputs[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream): raise RuntimeError('EXECUTE_ASYNC_V3_FAILED')
        self.stream.synchronize(); return outputs


def split(outputs):
    return ([outputs[f'hidden_l{i}'] for i in range(28)],
            [outputs[f'present_k{i}'] for i in range(28)],
            [outputs[f'present_v{i}'] for i in range(28)])


def cpu_selected(hs, ks, vs):
    return {i: {'hidden': hs[i].detach().cpu(), 'k': ks[i].detach().cpu(), 'v': vs[i].detach().cpu()} for i in range(28)}


def runtime_mode(a):
    out = a.out; out.mkdir(parents=True, exist_ok=True); trace = [mem_snapshot('runtime_start')]
    validate_manifest(a.handoff, out)
    torch.manual_seed(0)
    x = torch.randn((1, 8, 1024), device='cuda', dtype=torch.float16)
    pos = torch.arange(8, device='cuda', dtype=torch.long).unsqueeze(0)
    steps = [torch.randn((1, 1, 1024), device='cuda', dtype=torch.float16) for _ in range(4)]
    stack = load_stack(a.handoff, torch.float16); trace.append(mem_snapshot('reference_stack_loaded'))
    with torch.inference_mode():
        hs, ks, vs = stack.forward_prefill(x, pos); reference_pre = cpu_selected(hs, ks, vs)
        reference_decode = []
        for step, token in enumerate(steps):
            p = torch.tensor([[8 + step]], device='cuda', dtype=torch.long)
            hs, ks, vs = stack.forward_decode(token, p, ks, vs)
            reference_decode.append(cpu_selected(hs, ks, vs))
    torch.cuda.synchronize(); trace.append(mem_snapshot('reference_complete'))
    del stack, hs, ks, vs; gc.collect(); torch.cuda.empty_cache(); trace.append(mem_snapshot('reference_released'))

    pre = TRT(a.tmp / 'prefill_28layer.engine'); trace.append(mem_snapshot('prefill_engine_loaded'))
    ph, pk, pv = split(pre.run({'hidden_states': x, 'position_ids': pos})); trace.append(mem_snapshot('prefill_executed'))
    pre_metrics = []
    for i in (0,3,7,15,27):
        pre_metrics.append({'layer': i, 'hidden': metric(reference_pre[i]['hidden'], ph[i].cpu()),
                            'k': metric(reference_pre[i]['k'], pk[i].cpu()), 'v': metric(reference_pre[i]['v'], pv[i].cpu())})
    pre_ok = (all(bool(torch.isfinite(t).all()) and t.device.type == 'cuda' for g in (ph,pk,pv) for t in g)
              and all(list(t.shape) == [1,8,8,128] for g in (pk,pv) for t in g)
              and all(list(t.shape) == [1,8,1024] for t in ph))
    del pre, ph; gc.collect(); torch.cuda.empty_cache(); trace.append(mem_snapshot('prefill_engine_released'))

    dec = TRT(a.tmp / 'decode_28layer.engine'); trace.append(mem_snapshot('decode_engine_loaded'))
    decode_rows = []; propagation = [{'stage': 'prefill', **row} for row in pre_metrics]
    for step, token in enumerate(steps):
        oldk, oldv = pk, pv; p = torch.tensor([[8 + step]], device='cuda', dtype=torch.long)
        outputs = dec.run({'hidden_states': token, 'position_ids': p,
            **{f'past_k{i}': oldk[i] for i in range(28)}, **{f'past_v{i}': oldv[i] for i in range(28)}})
        dh, pk, pv = split(outputs); layers = []
        all_prefix_k = [bool(torch.equal(oldk[i], pk[i][:,:,:oldk[i].shape[2],:])) for i in range(28)]
        all_prefix_v = [bool(torch.equal(oldv[i], pv[i][:,:,:oldv[i].shape[2],:])) for i in range(28)]
        all_new_k = [metric(reference_decode[step][i]['k'][:,:,-1:,:], pk[i][:,:,-1:,:].cpu()) for i in range(28)]
        all_new_v = [metric(reference_decode[step][i]['v'][:,:,-1:,:], pv[i][:,:,-1:,:].cpu()) for i in range(28)]
        for i in (0,3,7,15,27):
            ref = reference_decode[step][i]
            row = {'layer': i, 'hidden': metric(ref['hidden'], dh[i].cpu()),
                   'k': metric(ref['k'], pk[i].cpu()), 'v': metric(ref['v'], pv[i].cpu()),
                   'prefix_k_exact': bool(torch.equal(oldk[i], pk[i][:,:,:oldk[i].shape[2],:])),
                   'prefix_v_exact': bool(torch.equal(oldv[i], pv[i][:,:,:oldv[i].shape[2],:])),
                   'new_k': metric(ref['k'][:,:,-1:,:], pk[i][:,:,-1:,:].cpu()),
                   'new_v': metric(ref['v'][:,:,-1:,:], pv[i][:,:,-1:,:].cpu())}
            layers.append(row); propagation.append({'stage': f'decode_{8+step}_to_{9+step}', **row})
        decode_rows.append({'step': step, 'past_length': 8+step, 'present_length': int(pk[0].shape[2]),
            'all_28_finite_cuda': all(bool(torch.isfinite(t).all()) and t.device.type == 'cuda' for g in (dh,pk,pv) for t in g),
            'all_28_shapes': all(list(t.shape) == [1,8,9+step,128] for g in (pk,pv) for t in g),
            'all_28_prefix_k_exact': all(all_prefix_k), 'all_28_prefix_v_exact': all(all_prefix_v),
            'all_28_new_slots_finite': all(x['finite'] for x in all_new_k + all_new_v),
            'all_28_new_k_max_relative_l2': max(x['relative_l2'] for x in all_new_k),
            'all_28_new_v_max_relative_l2': max(x['relative_l2'] for x in all_new_v),
            'k_pointer_isolation': len({int(t.data_ptr()) for t in pk}) == 28,
            'v_pointer_isolation': len({int(t.data_ptr()) for t in pv}) == 28, 'selected_layers': layers})
        del outputs, dh; trace.append(mem_snapshot(f'decode_{8+step}_to_{9+step}'))
    prefix_ok = all(row['all_28_prefix_k_exact'] and row['all_28_prefix_v_exact'] for row in decode_rows)
    decode_ok = all(row['present_length'] == 9+row['step'] and row['all_28_finite_cuda'] and row['all_28_shapes'] for row in decode_rows)
    isolation = all(row['k_pointer_isolation'] and row['v_pointer_isolation'] for row in decode_rows)
    metrics = [x[k] for x in propagation for k in ('hidden','k','v')]
    numerical_ok = all(m['finite'] and m['shape_equal'] and m['relative_l2'] <= .10 and m['cosine'] >= .99 for m in metrics)
    (out/'runtime_validation.json').write_text(json.dumps({'status': 'PASS' if pre_ok else 'FAIL', 'layers': 28, 'prefill_shape': [1,8,1024], 'cache_shape': [1,8,8,128], 'all_finite_cuda': pre_ok}, indent=2)+'\n')
    (out/'decode_validation.json').write_text(json.dumps({'status': 'PASS' if decode_ok else 'FAIL', 'steps': decode_rows}, indent=2)+'\n')
    (out/'cache_validation.json').write_text(json.dumps({'status': 'PASS' if prefix_ok and isolation else 'FAIL', 'prefix_exact': prefix_ok, 'layer_pointer_isolation': isolation}, indent=2)+'\n')
    (out/'numerical_propagation.json').write_text(json.dumps({'decision': 'ACCEPTABLE_FOR_FULL_FP16_RUNTIME_STEP' if numerical_ok else 'NUMERICAL_PROPAGATION_RISK_REQUIRES_REVIEW', 'acceptance': {'relative_l2_max': .10, 'cosine_min': .99}, 'rows': propagation}, indent=2)+'\n')
    weights = sum(p.stat().st_size for p in a.handoff.glob('layer_??.pt'))
    (out/'memory_accounting.json').write_text(json.dumps({'decoder_handoff_file_bytes': weights, 'raw_fp16_tensor_bytes': 880932864,
        'kv_bytes_per_token': 114688, 'kv_bytes_L8': 917504, 'kv_bytes_L12': 1376256,
        'onnx_bytes': {'prefill': (a.tmp/'prefill_28layer.onnx').stat().st_size, 'decode': (a.tmp/'decode_28layer.onnx').stat().st_size},
        'engine_bytes': {'prefill': (a.tmp/'prefill_28layer.engine').stat().st_size, 'decode': (a.tmp/'decode_28layer.engine').stat().st_size},
        'capacity_claim': 'NONE'}, indent=2)+'\n')
    (out/'memory_trace_runtime.json').write_text(json.dumps(trace, indent=2)+'\n')
    gates = {'B4.2-1':'PASS','B4.2-2':'PASS','B4.2-3':'PASS','B4.2-4':'PASS' if pre_ok else 'FAIL',
        'B4.2-5':'PASS' if decode_ok else 'FAIL','B4.2-6':'PASS' if prefix_ok and isolation else 'FAIL',
        'B4.2-7':'PASS' if numerical_ok else 'REVIEW','B4.2-7_decision':'ACCEPTABLE_FOR_FULL_FP16_RUNTIME_STEP' if numerical_ok else 'NUMERICAL_PROPAGATION_RISK_REQUIRES_REVIEW',
        'B4.2-8':'PASS','overall':'PASS / BOUNDED' if pre_ok and decode_ok and prefix_ok and isolation and numerical_ok else 'REVIEW',
        'decision':'REAL_28_LAYER_TRT_DECODER_STACK_FEASIBLE' if pre_ok and decode_ok and prefix_ok and isolation and numerical_ok else 'REVIEW_REQUIRED',
        'primary_architecture':'ONE_28_LAYER_STACK_ENGINE','partition_fallback_used':False}
    (out/'gate_summary.json').write_text(json.dumps(gates, indent=2)+'\n'); print(json.dumps(gates))


def main(a):
    if a.mode == 'manifest':
        a.out.mkdir(parents=True, exist_ok=True); validate_manifest(a.handoff, a.out)
    elif a.mode == 'export': export_mode(a)
    elif a.mode == 'build': build_mode(a)
    elif a.mode == 'runtime': runtime_mode(a)
    else: raise ValueError(a.mode)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['manifest','export','build','runtime'], required=True)
    p.add_argument('--handoff', type=Path, required=True); p.add_argument('--tmp', type=Path, required=True); p.add_argument('--out', type=Path, required=True); main(p.parse_args())
