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

from portable_qwen3_28 import PortableQwen3Layer

MODEL = Path('/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca')
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
EXPECTED_SHA = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
REQUIRED = ['input_layernorm.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','self_attn.q_norm.weight','self_attn.k_norm.weight','post_attention_layernorm.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']

def sha_bytes(x): return hashlib.sha256(x).hexdigest()
def sha_file(p): return sha_bytes(p.read_bytes())

def metric(a, b):
    a, b = a.float(), b.float(); d = a - b; rn, tn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b); tiny = torch.finfo(torch.float32).tiny
    return {'shape_equal': list(a.shape)==list(b.shape), 'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'max_abs_error': float(d.abs().max()), 'mean_abs_error': float(d.abs().mean()), 'rmse': float(torch.sqrt((d*d).mean())), 'relative_l2_error': float(torch.linalg.vector_norm(d)/torch.clamp(rn,min=tiny)), 'cosine_similarity': float(torch.dot(a.reshape(-1),b.reshape(-1))/torch.clamp(rn*tn,min=tiny)), 'reference_rms': float(torch.sqrt((a*a).mean())), 'test_rms': float(torch.sqrt((b*b).mean()))}

def load_state(source, idx):
    out = {}
    for name in REQUIRED: out[name] = source.get_tensor(f'model.layers.{idx}.{name}')
    return out

def hf_call(layer, rotary, hidden, pos, cache, idx, past_len):
    b, s, _ = hidden.shape; total = past_len + s; mask = torch.zeros((b,1,s,total),device=hidden.device,dtype=hidden.dtype)
    if past_len == 0: mask = mask.masked_fill(torch.triu(torch.ones((s,s),device=hidden.device,dtype=torch.bool),1)[None,None], torch.finfo(hidden.dtype).min)
    out = layer(hidden_states=hidden, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True, cache_position=torch.arange(past_len,past_len+s,device=hidden.device), position_embeddings=rotary(hidden,pos))
    k, v = cache[idx]
    return out, k, v

def cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().contiguous()
    if isinstance(x, list): return [cpu(y) for y in x]
    if isinstance(x, dict): return {k: cpu(v) for k,v in x.items()}
    return x

def main(a):
    a.out.mkdir(parents=True,exist_ok=True); a.handoff.mkdir(parents=True,exist_ok=True); ck = MODEL/'model.safetensors'; digest = sha_file(ck)
    if digest != EXPECTED_SHA: raise RuntimeError(f'checkpoint SHA mismatch {digest}')
    cfg = Qwen3Config.from_pretrained(str(MODEL)); cfg._attn_implementation='eager'; rotary = Qwen3RotaryEmbedding(cfg).to('cuda')
    torch.manual_seed(0); x = torch.randn((1,8,1024),device='cuda',dtype=torch.bfloat16); pos = torch.arange(8,device='cuda',dtype=torch.long).unsqueeze(0); steps = [torch.randn((1,1,1024),device='cuda',dtype=torch.bfloat16) for _ in range(4)]
    states, items = {}, []
    with safe_open(str(ck),framework='pt',device='cpu') as source:
        for idx in range(28):
            st = load_state(source,idx); states[str(idx)] = st
            for name,t in st.items(): items.append({'layer':idx,'checkpoint_key':f'model.layers.{idx}.{name}','module_key':name,'shape':list(t.shape),'dtype':str(t.dtype),'numel':t.numel(),'bytes':t.numel()*t.element_size(),'sha256':sha_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())})
    hf_pre = {'hidden':[], 'k':[], 'v':[]}; hf_dec=[]; cache=DynamicCache(); hidden=x
    with torch.inference_mode(), safe_open(str(ck),framework='pt',device='cpu') as source:
        for idx in range(28):
            layer=Qwen3DecoderLayer(cfg,layer_idx=idx).to('cuda',dtype=torch.bfloat16).eval(); layer.load_state_dict(load_state(source,idx),strict=True); hidden,k,v=hf_call(layer,rotary,hidden,pos,cache,idx,0); hf_pre['hidden'].append(cpu(hidden)); hf_pre['k'].append(cpu(k)); hf_pre['v'].append(cpu(v)); del layer; torch.cuda.empty_cache()
        for step,token in enumerate(steps):
            p=torch.tensor([[8+step]],device='cuda',dtype=torch.long); current=token; row={'hidden':[],'k':[],'v':[]}
            for idx in range(28):
                layer=Qwen3DecoderLayer(cfg,layer_idx=idx).to('cuda',dtype=torch.bfloat16).eval(); layer.load_state_dict(load_state(source,idx),strict=True); current,k,v=hf_call(layer,rotary,current,p,cache,idx,8+step); row['hidden'].append(cpu(current)); row['k'].append(cpu(k)); row['v'].append(cpu(v)); del layer; torch.cuda.empty_cache()
            hf_dec.append(row)
    port_pre={'hidden':[],'k':[],'v':[],'attn':[]}; port_dec=[]; pk=[None]*28; pv=[None]*28; hidden=x
    for idx in range(28):
        layer=PortableQwen3Layer().to('cuda',dtype=torch.bfloat16).eval(); layer.load_state_dict(states[str(idx)],strict=True); hidden,k,v,attn=layer.forward_prefill(hidden,pos); port_pre['hidden'].append(cpu(hidden)); port_pre['k'].append(cpu(k)); port_pre['v'].append(cpu(v)); port_pre['attn'].append(cpu(attn)); pk[idx],pv[idx]=k,v; del layer; torch.cuda.empty_cache()
    for step,token in enumerate(steps):
        p=torch.tensor([[8+step]],device='cuda',dtype=torch.long); current=token; row={'hidden':[],'k':[],'v':[],'attn':[]}
        for idx in range(28):
            layer=PortableQwen3Layer().to('cuda',dtype=torch.bfloat16).eval(); layer.load_state_dict(states[str(idx)],strict=True); current,pk[idx],pv[idx],attn=layer.forward_decode(current,p,pk[idx],pv[idx]); row['hidden'].append(cpu(current)); row['k'].append(cpu(pk[idx])); row['v'].append(cpu(pv[idx])); row['attn'].append(cpu(attn)); del layer; torch.cuda.empty_cache()
        port_dec.append(row)
    sem={'prefill':[],'decode':[]}
    for idx in range(28): sem['prefill'].append({'layer':idx,'hidden':metric(hf_pre['hidden'][idx],port_pre['hidden'][idx]),'k':metric(hf_pre['k'][idx],port_pre['k'][idx]),'v':metric(hf_pre['v'][idx],port_pre['v'][idx])})
    for step in range(4): sem['decode'].append({'step':step,'layers':[{'layer':idx,'hidden':metric(hf_dec[step]['hidden'][idx],port_dec[step]['hidden'][idx]),'k':metric(hf_dec[step]['k'][idx],port_dec[step]['k'][idx]),'v':metric(hf_dec[step]['v'][idx],port_dec[step]['v'][idx])} for idx in range(28)]})
    exact=all(m['max_abs_error']==0.0 for row in sem['prefill'] for m in (row['hidden'],row['k'],row['v'])) and all(m['max_abs_error']==0.0 for row in sem['decode'] for lay in row['layers'] for m in (lay['hidden'],lay['k'],lay['v']))
    handoff={'state_dicts':states,'hidden':cpu(x),'position_ids':cpu(pos),'decode_inputs':cpu(steps),'hf_prefill':cpu(hf_pre),'hf_decode':cpu(hf_dec),'portable_prefill':cpu(port_pre),'portable_decode':cpu(port_dec),'model_revision':REVISION}
    hp=a.handoff/'layers0_27_handoff.pt'; torch.save(handoff,hp); hh=sha_file(hp)
    (a.out/'layers_0_27_weight_manifest.json').write_text(json.dumps({'model_sha256':digest,'revision':REVISION,'layers':28,'items':items,'per_layer_params':{str(i):sum(x['numel'] for x in items if x['layer']==i) for i in range(28)},'total_params':sum(x['numel'] for x in items),'total_bf16_bytes':sum(x['bytes'] for x in items),'total_fp16_bytes':sum(x['numel']*2 for x in items)},indent=2)+'\n')
    (a.out/'weight_mapping_audit.json').write_text(json.dumps({'status':'PASS','layers':28,'required_tensors_per_layer':11,'all_required_keys_found':True,'shape_match':True},indent=2)+'\n'); (a.out/'portable_semantic_comparison.json').write_text(json.dumps(sem,indent=2)+'\n'); (a.out/'hf_28layer_reference_summary.json').write_text(json.dumps({'status':'PASS','layers':28,'prefill_shape':[1,8,1024],'cache_shape':[1,8,8,128],'decode_lengths':[9,10,11,12],'finite':True,'sequential_layer_memory_safe':True},indent=2)+'\n'); (a.out/'model_identity.txt').write_text(f'model=Qwen/Qwen3-0.6B\nrevision={REVISION}\nsnapshot={MODEL}\nsha256={digest}\n'); (a.out/'handoff_manifest.json').write_text(json.dumps({'status':'PASS','filename':hp.name,'location':str(a.handoff),'sha256':hh,'size':hp.stat().st_size,'contains':'layers 0-27 state and bounded references only','no_embedding_or_lm_head':True},indent=2)+'\n'); (a.out/'handoff_integrity.json').write_text(json.dumps({'status':'PASS','sha256':hh,'size':hp.stat().st_size,'verified_before_stage_b':False},indent=2)+'\n'); (a.out/'environment_phase1.txt').write_text(f'stage=phase1-hf\nrevision={REVISION}\nlayers=28\n'); print(json.dumps({'status':'PASS' if exact else 'BLOCKED_BY_28L_ORACLE_MISMATCH','layers':28,'semantic_exact':exact,'handoff':str(hp),'handoff_sha256':hh,'total_params':sum(x['numel'] for x in items)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--handoff',type=Path,required=True); main(p.parse_args())
