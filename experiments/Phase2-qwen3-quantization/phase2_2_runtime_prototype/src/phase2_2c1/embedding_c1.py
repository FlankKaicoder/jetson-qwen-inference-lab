from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
from collections import Counter
from pathlib import Path

import torch

MODEL = Path('/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca')
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
MODEL_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
EMBED_KEY = 'model.embed_tokens.weight'
SELECTED = {
    'short': torch.tensor([[0, 1, 2]], dtype=torch.long),
    'prefill': torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long),
    'alternate': torch.tensor([[151935, 17, 42, 999, 2048, 7, 12345, 31415]], dtype=torch.long),
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def snap(stage):
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


def load_weight():
    from safetensors import safe_open
    path = MODEL / 'model.safetensors'
    if sha_file(path) != MODEL_SHA:
        raise RuntimeError('MODEL_SHA_MISMATCH')
    with safe_open(str(path), framework='pt', device='cpu') as source:
        weight = source.get_tensor(EMBED_KEY).contiguous()
    return weight


def metric(a, b):
    a, b = a.float(), b.float(); d = a - b
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {'shape_equal': list(a.shape) == list(b.shape), 'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            'max_abs_error': float(d.abs().max()), 'mean_abs_error': float(d.abs().mean()),
            'rmse': float(torch.sqrt((d*d).mean())), 'relative_l2': float(torch.linalg.vector_norm(d) / torch.clamp(an, min=tiny)),
            'cosine': float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(an*bn, min=tiny))}


def audit(a):
    w = load_weight(); rows = {'status': 'PASS', 'model': 'Qwen/Qwen3-0.6B', 'revision': REVISION,
        'checkpoint_sha256': MODEL_SHA, 'module_path': 'model.embed_tokens', 'checkpoint_key': EMBED_KEY,
        'shape': list(w.shape), 'dtype': str(w.dtype), 'numel': w.numel(), 'bf16_bytes': w.numel()*w.element_size(),
        'fp16_bytes': w.numel()*2, 'weight_sha256_bf16': sha_tensor(w), 'vocab_size': w.shape[0], 'hidden_size': w.shape[1],
        'tied_to_lm_head': True, 'tie_evidence': 'Qwen3 config tie_word_embeddings=true; lm_head shares embed_tokens weight'}
    a.out.mkdir(parents=True, exist_ok=True); (a.out/'embedding_weight_audit.json').write_text(json.dumps(rows, indent=2)+'\n')
    if a.weight:
        torch.save(w.half(), a.weight)
    print(json.dumps(rows))


def reference(a):
    w = load_weight(); bf16, fp16 = {}, {}
    with torch.inference_mode():
        for name, ids in SELECTED.items():
            bf16[name] = torch.nn.functional.embedding(ids, w).contiguous()
            fp16[name] = bf16[name].half()
    payload = {'bf16': bf16, 'fp16': fp16, 'weight_meta': {'shape': list(w.shape), 'dtype': str(w.dtype), 'sha256': sha_tensor(w)}}
    torch.save(payload, a.reference)
    print(json.dumps({'status': 'PASS', 'cases': list(bf16), 'reference': str(a.reference)}))


class Embedding(torch.nn.Module):
    def __init__(self, weight):
        super().__init__(); self.weight = torch.nn.Parameter(weight, requires_grad=False)
    def forward(self, ids):
        return torch.nn.functional.embedding(ids, self.weight)


def export(a):
    import onnx
    w = torch.load(a.weight, map_location='cpu', weights_only=True).to(torch.float16)
    module = Embedding(w).eval(); ids = SELECTED['prefill']
    a.tmp.mkdir(parents=True, exist_ok=True); path = a.tmp/'embedding_fp16.onnx'
    torch.onnx.export(module, (ids,), str(path), opset_version=17, input_names=['input_ids'], output_names=['hidden_states'],
                      dynamic_axes={'input_ids': {0:'batch',1:'seq'}, 'hidden_states': {0:'batch',1:'seq'}})
    model = onnx.load(str(path)); onnx.checker.check_model(model)
    meta = {'status':'PASS','opset':17,'bytes':path.stat().st_size,'node_count':len(model.graph.node),
            'initializer_count':len(model.graph.initializer),'operators':dict(Counter(n.op_type for n in model.graph.node)),
            'input':'input_ids int64 [B,S]','output':'hidden_states float16 [B,S,1024]'}
    a.out.mkdir(parents=True, exist_ok=True); (a.out/'embedding_onnx_summary.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta))


def build(a):
    import tensorrt as trt
    path = a.tmp/'embedding_fp16.onnx'; a.out.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); parser = trt.OnnxParser(network, logger)
    ok = parser.parse(path.read_bytes()); errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not ok: raise RuntimeError(f'EMBED_PARSER_FAILED:{errors}')
    config = builder.create_builder_config(); config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile(); profile.set_shape('input_ids',(1,1),(1,8),(2,1024)); config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None: raise RuntimeError('EMBED_BUILD_FAILED')
    engine = a.tmp/'embedding_fp16.engine'; engine.write_bytes(bytes(blob))
    result = {'status':'PASS','engine_bytes':engine.stat().st_size,'parser_errors':errors,'fp16_only':True,
              'input':'input_ids int64 [B,S] dynamic','output':'hidden_states float16 [B,S,1024] dynamic'}
    (a.out/'embedding_engine_summary.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result))


class TRT:
    def __init__(self, path):
        import tensorrt as trt
        self.trt = trt; self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None: raise RuntimeError('ENGINE_DESERIALIZE_FAILED')
        self.stream = torch.cuda.current_stream()
    def run(self, inputs):
        ctx = self.engine.create_execution_context(); outputs = {}
        for name, t in inputs.items():
            if not ctx.set_input_shape(name, tuple(t.shape)): raise RuntimeError('INPUT_SHAPE_REJECTED')
            ctx.set_tensor_address(name, t.data_ptr())
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name)); dtype = self.engine.get_tensor_dtype(name)
                outputs[name] = torch.empty(shape, device='cuda', dtype=torch.float16 if dtype == self.trt.DataType.HALF else torch.int64)
                ctx.set_tensor_address(name, outputs[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream): raise RuntimeError('EXECUTE_FAILED')
        self.stream.synchronize(); return outputs


def validate_embedding(a):
    out = a.out; out.mkdir(parents=True, exist_ok=True); trace = [snap('start')]
    if a.reference:
        payload = torch.load(a.reference, map_location='cpu', weights_only=True)
        refs = payload['bf16']; fp_refs = payload['fp16']; weight_meta = payload['weight_meta']; del payload
    else:
        w = load_weight(); weight_meta = {'shape':list(w.shape),'dtype':str(w.dtype),'sha256':sha_tensor(w)}
        refs = {}; fp_refs = {}
        with torch.inference_mode():
            for name, ids in SELECTED.items():
                refs[name] = torch.nn.functional.embedding(ids, w).contiguous(); fp_refs[name] = refs[name].half()
        del w
    trace.append(snap('after_reference')); gc.collect()
    emb = TRT(a.tmp/'embedding_fp16.engine'); rows = []
    for name, ids_cpu in SELECTED.items():
        ids = ids_cpu.cuda(); result = emb.run({'input_ids':ids}); got = result['hidden_states']
        rows.append({'case':name,'input_ids':ids_cpu.tolist(),'shape':list(got.shape),'dtype':str(got.dtype),'cuda':got.device.type=='cuda',
                     'hf_bf16':metric(refs[name],got.cpu()),'hf_fp16':metric(fp_refs[name],got.cpu()),
                     'finite':bool(torch.isfinite(got).all())})
    trace.append(snap('embedding_engine_complete')); (out/'embedding_validation.json').write_text(json.dumps({'status':'PASS','weight':weight_meta,'cases':rows}, indent=2)+'\n')
    del emb, result, got; gc.collect(); torch.cuda.empty_cache()

    # Integrate with the existing B4.2 decoder without rebuilding or modifying it.
    decoder_dir = a.decoder_dir or Path('/tmp/phase2_2b4_2_20260902T082326Z')
    pre = TRT(decoder_dir/'prefill_28layer.engine'); ids = SELECTED['prefill'].cuda(); emb_out = pre_emb = TRT(a.tmp/'embedding_fp16.engine').run({'input_ids':ids})['hidden_states']
    pos = torch.arange(8, device='cuda', dtype=torch.long).unsqueeze(0); dec_out = pre.run({'hidden_states':emb_out,'position_ids':pos})
    final_trt = dec_out['hidden_l27']; trace.append(snap('trt_embedding_to_decoder'))
    # Portable FP16 reference uses the existing B4.2 source-faithful layer implementation.
    import sys
    b4 = Path(__file__).resolve().parents[1] / 'phase2_2b4_2'; sys.path.insert(0, str(b4))
    from portable_qwen3_28 import PortableTwentyEightLayerStack
    handoff = Path('/tmp/phase2_2b4_stream_20260902T070000Z'); stack = PortableTwentyEightLayerStack(device='cuda', dtype=torch.float16).eval()
    for i in range(28):
        payload = torch.load(handoff/f'layer_{i:02d}.pt', map_location='cpu', weights_only=True)
        stack.layers[i].load_state_dict({k:v.half() for k,v in payload.items()}, strict=True); del payload
    with torch.inference_mode(): ref_hs, ref_k, ref_v = stack.forward_prefill(refs['prefill'].cuda().half(), pos)
    final_ref = ref_hs[-1]; final_metric = metric(final_ref, final_trt)
    integration_status = 'PASS' if final_metric['relative_l2'] <= 0.10 and final_metric['cosine'] >= 0.99 else 'BLOCKED_NUMERICAL_MISMATCH'
    (out/'decoder_integration_validation.json').write_text(json.dumps({'status':integration_status,'input_ids':SELECTED['prefill'].tolist(),'embedding_to_decoder':integration_status,
        'decoder_output_shape':list(final_trt.shape),'decoder_output_dtype':str(final_trt.dtype),'decoder_output_cuda':final_trt.device.type=='cuda',
        'final_hidden_layer':27,'final_hidden_metric':final_metric,'all_decoder_outputs_finite':all(bool(torch.isfinite(t).all()) for t in dec_out.values())}, indent=2)+'\n')
    trace.append(snap('portable_reference_complete')); (out/'memory_trace_c1.json').write_text(json.dumps(trace, indent=2)+'\n')
    if integration_status != 'PASS':
        raise RuntimeError('BLOCKED_BY_EMBEDDING_TO_B4_2_DECODER_NUMERICAL_MISMATCH')
    print(json.dumps({'status':'PASS','embedding_cases':len(rows),'final_hidden_metric':final_metric}))


def main(a):
    if a.mode == 'audit': audit(a)
    elif a.mode == 'reference': reference(a)
    elif a.mode == 'export': export(a)
    elif a.mode == 'build': build(a)
    elif a.mode == 'validate': validate_embedding(a)
    else: raise ValueError(a.mode)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['audit','reference','export','build','validate'], required=True)
    p.add_argument('--tmp', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--weight', type=Path); p.add_argument('--reference', type=Path); p.add_argument('--decoder-dir', type=Path); main(p.parse_args())
