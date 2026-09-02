from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
B4 = HERE.parent / 'phase2_2b4_2'
if not (B4 / 'portable_qwen3_28.py').exists():
    B4 = HERE
sys.path.insert(0, str(B4))
from portable_qwen3_28 import PortableTwentyEightLayerStack


def metric(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    d = a - b
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {
        'shape_equal': list(a.shape) == list(b.shape),
        'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        'max_abs_error': float(d.abs().max()),
        'rmse': float(torch.sqrt((d * d).mean())),
        'relative_l2': float(torch.linalg.vector_norm(d) / torch.clamp(an, min=tiny)),
        'cosine': float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(an * bn, min=tiny)),
    }


def sha_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def describe(t):
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'device': str(t.device),
            'strides': list(t.stride()), 'numel': t.numel(), 'nbytes': t.numel() * t.element_size(),
            'contiguous': t.is_contiguous(), 'sha256_raw': sha_tensor(t), 'data_ptr': int(t.data_ptr())}


class TRT:
    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt
        self.path = Path(path)
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f'ENGINE_DESERIALIZE_FAILED:{self.path}')
        self.stream = torch.cuda.current_stream()

    def contract(self):
        rows = []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            rows.append({'index': i, 'name': n, 'mode': str(self.engine.get_tensor_mode(n)),
                         'dtype': str(self.engine.get_tensor_dtype(n)),
                         'shape': list(self.engine.get_tensor_shape(n))})
        return {'engine': str(self.path), 'num_io_tensors': self.engine.num_io_tensors,
                'stream_ptr': int(self.stream.cuda_stream), 'tensors': rows}

    def run(self, inputs):
        ctx = self.engine.create_execution_context()
        for name, t in inputs.items():
            if not ctx.set_input_shape(name, tuple(t.shape)):
                raise RuntimeError(f'SHAPE_REJECTED:{name}:{tuple(t.shape)}')
            ctx.set_tensor_address(name, t.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                td = torch.float16 if dtype == self.trt.DataType.HALF else torch.int64
                outputs[name] = torch.empty(shape, device='cuda', dtype=td)
                ctx.set_tensor_address(name, outputs[name].data_ptr())
        ok = bool(ctx.execute_async_v3(self.stream.cuda_stream))
        self.stream.synchronize()
        if not ok:
            raise RuntimeError('EXECUTE_ASYNC_V3_FAILED')
        return outputs, {'context_id': id(ctx), 'stream_ptr': int(self.stream.cuda_stream),
                         'input_ptrs': {n: int(t.data_ptr()) for n, t in inputs.items()},
                         'output_ptrs': {n: int(t.data_ptr()) for n, t in outputs.items()}}


def split(o):
    return ([o[f'hidden_l{i}'] for i in range(28)], [o[f'present_k{i}'] for i in range(28)],
            [o[f'present_v{i}'] for i in range(28)])


def load_stack(handoff):
    stack = PortableTwentyEightLayerStack(device='cuda', dtype=torch.float16).eval()
    for i in range(28):
        p = torch.load(Path(handoff) / f'layer_{i:02d}.pt', map_location='cpu', weights_only=True)
        stack.layers[i].load_state_dict({k: v.half() for k, v in p.items()}, strict=True)
        del p
        gc.collect()
    return stack


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pre = TRT(Path(a.decoder_dir) / 'prefill_28layer.engine')
    emb = TRT(Path(a.embedding_dir) / 'embedding_fp16.engine')
    ids = torch.arange(8, device='cuda', dtype=torch.long).reshape(1, 8)
    pos = torch.arange(8, device='cuda', dtype=torch.long).reshape(1, 8)
    torch.manual_seed(20260902)
    random_x = torch.randn((1, 8, 1024), device='cuda', dtype=torch.float16).contiguous()
    results = {'starting': {'input_ids': ids.cpu().tolist(), 'position_ids': pos.cpu().tolist()},
               'stream_pointer': int(torch.cuda.current_stream().cuda_stream),
               'engine_contract': {'embedding': emb.contract(), 'decoder_prefill': pre.contract()}}

    stack = load_stack(a.handoff)
    with torch.inference_mode():
        ref_random_h, _, _ = stack.forward_prefill(random_x, pos)
    trt_random, random_audit = pre.run({'hidden_states': random_x, 'position_ids': pos})
    random_h, _, _ = split(trt_random)
    results['D0_b4_2_control'] = {'input': describe(random_x), 'portable_layer27': describe(ref_random_h[-1].detach()),
                                  'trt_layer27': describe(random_h[-1]), 'metric': metric(ref_random_h[-1], random_h[-1].cpu()),
                                  'execution': random_audit,
                                  'status': 'PASS' if metric(ref_random_h[-1], random_h[-1].cpu())['relative_l2'] <= .10 and metric(ref_random_h[-1], random_h[-1].cpu())['cosine'] >= .99 else 'FAIL'}
    if results['D0_b4_2_control']['status'] != 'PASS':
        (out / 'c1d_diagnostic.json').write_text(json.dumps(results, indent=2) + '\n')
        raise RuntimeError('D0_B4_2_CURRENT_CONTROL_FAIL')
    del trt_random, random_h, ref_random_h, random_x
    gc.collect(); torch.cuda.empty_cache()

    # The serialized C1 reference contains the exact BF16 embedding output; convert once to canonical FP16.
    payload = torch.load(a.embedding_reference, map_location='cpu', weights_only=True)
    canonical = payload['fp16']['prefill'].contiguous().cuda()
    del payload
    trt_emb_out, emb_audit = emb.run({'input_ids': ids})
    trt_embedding = trt_emb_out['hidden_states']
    canonical_cpu = canonical.cpu()
    trt_embedding_cpu = trt_embedding.cpu()
    results['D1_boundary_fingerprint'] = {'A_portable_fp16': describe(canonical), 'B_trt_output': describe(trt_embedding),
        'metric_A_vs_B': metric(canonical_cpu, trt_embedding_cpu), 'byte_identical': bool(torch.equal(canonical_cpu.view(torch.uint8), trt_embedding_cpu.view(torch.uint8))),
        'first_difference': None, 'execution': emb_audit}
    if not results['D1_boundary_fingerprint']['byte_identical']:
        neq = torch.nonzero(canonical_cpu.view(torch.uint8) != trt_embedding_cpu.view(torch.uint8), as_tuple=False)
        if neq.numel():
            i = int(neq[0]); results['D1_boundary_fingerprint']['first_difference'] = {'byte_index': i, 'A': int(canonical_cpu.view(torch.uint8)[i]), 'B': int(trt_embedding_cpu.view(torch.uint8)[i])}

    with torch.inference_mode():
        ref_canonical_h, _, _ = stack.forward_prefill(canonical, pos)
    trt_canonical_o, canonical_audit = pre.run({'hidden_states': canonical, 'position_ids': pos})
    trt_canonical_h, _, _ = split(trt_canonical_o)
    results['D2_canonical_hidden'] = {'portable_layer27': describe(ref_canonical_h[-1]), 'trt_layer27': describe(trt_canonical_h[-1]),
        'metric': metric(ref_canonical_h[-1], trt_canonical_h[-1].cpu()),
        'layerwise': [{'layer': i, 'metric': metric(ref_canonical_h[i], trt_canonical_h[i].cpu()),
                       'trt_sha256_raw': sha_tensor(trt_canonical_h[i])} for i in range(28)],
        'execution': canonical_audit}

    staged = trt_embedding.detach().clone().cpu().contiguous().cuda()
    staged_o, staged_audit = pre.run({'hidden_states': staged, 'position_ids': pos})
    staged_h, _, _ = split(staged_o)
    results['D3_host_staged'] = {'staged_input': describe(staged), 'layer27': describe(staged_h[-1]),
        'metric_vs_portable_canonical': metric(ref_canonical_h[-1], staged_h[-1].cpu()), 'execution': staged_audit}
    results['D3_direct_device'] = {'direct_input': describe(trt_embedding), 'layer27': describe(trt_canonical_h[-1]),
        'metric_vs_portable_canonical': metric(ref_canonical_h[-1], trt_canonical_h[-1].cpu()), 'execution': canonical_audit}

    results['D4_stream_pointer_audit'] = {'embedding_stream': emb_audit['stream_ptr'], 'decoder_stream': canonical_audit['stream_ptr'],
        'same_stream': emb_audit['stream_ptr'] == canonical_audit['stream_ptr'], 'embedding_output_ptr': int(trt_embedding.data_ptr()),
        'decoder_input_ptr': canonical_audit['input_ptrs']['hidden_states'], 'explicit_sync_after_each_enqueue': True,
        'producer_consumer_ordering': 'same current stream plus synchronize after embedding and decoder'}
    results['D5_binding_contract'] = results['engine_contract']['decoder_prefill']['tensors']
    results['conclusion'] = 'CONFIRMED_INPUT_MISMATCH' if not results['D1_boundary_fingerprint']['byte_identical'] else 'NOT_CONFIRMED_BY_BOUNDARY_OR_STREAM_TEST'
    (out / 'c1d_diagnostic.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'output': str(out / 'c1d_diagnostic.json'), 'conclusion': results['conclusion']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--decoder-dir', type=Path, required=True); p.add_argument('--embedding-dir', type=Path, required=True)
    p.add_argument('--embedding-reference', type=Path, required=True); p.add_argument('--handoff', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); main(p.parse_args())
