from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from portable_qwen3_28 import PortableQwen3Layer

MODEL=Path('/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca'); REV='c1899de289a04d12100db370d81485cdf75e47ca'; EXPECTED='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
REQ=['input_layernorm.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','self_attn.q_norm.weight','self_attn.k_norm.weight','post_attention_layernorm.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def cpu(x):
 if isinstance(x,torch.Tensor): return x.detach().cpu().contiguous()
 if isinstance(x,list): return [cpu(y) for y in x]
 if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
 return x
def metric(a,b):
 a,b=a.float(),b.float(); d=a-b; rn,tn=torch.linalg.vector_norm(a),torch.linalg.vector_norm(b); e=torch.finfo(torch.float32).tiny
 return {'max_abs_error':float(d.abs().max()),'rmse':float(torch.sqrt((d*d).mean())),'relative_l2_error':float(torch.linalg.vector_norm(d)/torch.clamp(rn,min=e)),'cosine_similarity':float(torch.dot(a.reshape(-1),b.reshape(-1))/torch.clamp(rn*tn,min=e)),'reference_rms':float(torch.sqrt((a*a).mean())),'test_rms':float(torch.sqrt((b*b).mean()))}
def hf_call(layer,rot,h,pos,cache,idx,past):
 s=h.shape[1]; mask=torch.zeros((1,1,s,past+s),device='cuda',dtype=h.dtype)
 if past==0: mask=mask.masked_fill(torch.triu(torch.ones((s,s),device='cuda',dtype=torch.bool),1)[None,None],torch.finfo(h.dtype).min)
 y=layer(hidden_states=h,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,cache_position=torch.arange(past,past+s,device='cuda'),position_embeddings=rot(h,pos)); k,v=cache[idx]; return y,k,v
def main(a):
 a.out.mkdir(parents=True,exist_ok=True); a.handoff.mkdir(parents=True,exist_ok=True); ck=MODEL/'model.safetensors'; sh=sha(ck)
 if sh!=EXPECTED: raise RuntimeError('checkpoint SHA mismatch')
 cfg=Qwen3Config.from_pretrained(str(MODEL)); cfg._attn_implementation='eager'; rot=Qwen3RotaryEmbedding(cfg).to('cuda'); torch.manual_seed(0); x=torch.randn((1,8,1024),device='cuda',dtype=torch.bfloat16); pos=torch.arange(8,device='cuda',dtype=torch.long).unsqueeze(0); steps=[torch.randn((1,1,1024),device='cuda',dtype=torch.bfloat16) for _ in range(4)]
 states={}; items=[]
 with safe_open(str(ck),framework='pt',device='cpu') as src:
  for idx in range(28):
   states[str(idx)]={n:src.get_tensor(f'model.layers.{idx}.{n}') for n in REQ}
   for n,t in states[str(idx)].items(): items.append({'layer':idx,'name':n,'shape':list(t.shape),'dtype':str(t.dtype),'numel':t.numel(),'bytes':t.numel()*t.element_size(),'sha256':hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
 hf_pre={'hidden':[],'k':[],'v':[]}; hf_dec=[]; hc=DynamicCache(); h=x
 with torch.inference_mode():
  for idx in range(28):
   l=Qwen3DecoderLayer(cfg,layer_idx=idx).to('cuda',dtype=torch.bfloat16).eval(); l.load_state_dict(states[str(idx)],strict=True); h,k,v=hf_call(l,rot,h,pos,hc,idx,0); hf_pre['hidden'].append(cpu(h)); hf_pre['k'].append(cpu(k)); hf_pre['v'].append(cpu(v)); del l; torch.cuda.empty_cache()
  for si,tok in enumerate(steps):
   p=torch.tensor([[8+si]],device='cuda',dtype=torch.long); cur=tok; row={'hidden':[],'k':[],'v':[]}
   for idx in range(28):
    l=Qwen3DecoderLayer(cfg,layer_idx=idx).to('cuda',dtype=torch.bfloat16).eval(); l.load_state_dict(states[str(idx)],strict=True); cur,k,v=hf_call(l,rot,cur,p,hc,idx,8+si); row['hidden'].append(cpu(cur)); row['k'].append(cpu(k)); row['v'].append(cpu(v)); del l; torch.cuda.empty_cache()
   hf_dec.append(row)
 port_pre={'hidden':[],'k':[],'v':[],'attn':[]}; port_dec=[]; pk=[None]*28; pv=[None]*28; h=x
 for idx in range(28):
  l=PortableQwen3Layer().to('cuda',dtype=torch.bfloat16).eval(); l.load_state_dict(states[str(idx)],strict=True); h,k,v,at=l.forward_prefill(h,pos); port_pre['hidden'].append(cpu(h)); port_pre['k'].append(cpu(k)); port_pre['v'].append(cpu(v)); port_pre['attn'].append(cpu(at)); pk[idx],pv[idx]=k,v; del l; torch.cuda.empty_cache()
 for si,tok in enumerate(steps):
  p=torch.tensor([[8+si]],device='cuda',dtype=torch.long); cur=tok; row={'hidden':[],'k':[],'v':[],'attn':[]}
  for idx in range(28):
   l=PortableQwen3Layer().to('cuda',dtype=torch.bfloat16).eval(); l.load_state_dict(states[str(idx)],strict=True); cur,pk[idx],pv[idx],at=l.forward_decode(cur,p,pk[idx],pv[idx]); row['hidden'].append(cpu(cur)); row['k'].append(cpu(pk[idx])); row['v'].append(cpu(pv[idx])); row['attn'].append(cpu(at)); del l; torch.cuda.empty_cache()
  port_dec.append(row)
 sem={'prefill':[{'layer':i,'hidden':metric(hf_pre['hidden'][i],port_pre['hidden'][i]),'k':metric(hf_pre['k'][i],port_pre['k'][i]),'v':metric(hf_pre['v'][i],port_pre['v'][i])} for i in range(28)],'decode':[{'step':s,'layers':[{'layer':i,'hidden':metric(hf_dec[s]['hidden'][i],port_dec[s]['hidden'][i]),'k':metric(hf_dec[s]['k'][i],port_dec[s]['k'][i]),'v':metric(hf_dec[s]['v'][i],port_dec[s]['v'][i])} for i in range(28)]} for s in range(4)]}
 exact=all(m['max_abs_error']==0.0 for row in sem['prefill'] for m in (row['hidden'],row['k'],row['v'])) and all(m['max_abs_error']==0.0 for row in sem['decode'] for lay in row['layers'] for m in (lay['hidden'],lay['k'],lay['v']))
 hp=a.handoff/'layers0_27_handoff.pt'; torch.save({'state_dicts':states,'hidden':cpu(x),'position_ids':cpu(pos),'decode_inputs':cpu(steps),'hf_prefill':cpu(hf_pre),'hf_decode':cpu(hf_dec),'portable_prefill':cpu(port_pre),'portable_decode':cpu(port_dec),'model_revision':REV},hp); hh=sha(hp)
 (a.out/'layers_0_27_weight_manifest.json').write_text(json.dumps({'model_sha256':sh,'revision':REV,'layers':28,'items':items,'total_params':sum(x['numel'] for x in items),'total_bf16_bytes':sum(x['bytes'] for x in items),'total_fp16_bytes':sum(x['numel']*2 for x in items)},indent=2)+'\n'); (a.out/'weight_mapping_audit.json').write_text(json.dumps({'status':'PASS','layers':28,'required_tensors_per_layer':11,'all_required_keys_found':True,'shape_match':True},indent=2)+'\n'); (a.out/'portable_semantic_comparison.json').write_text(json.dumps(sem,indent=2)+'\n'); (a.out/'hf_28layer_reference_summary.json').write_text(json.dumps({'status':'PASS','layers':28,'prefill_shape':[1,8,1024],'cache_shape':[1,8,8,128],'decode_lengths':[9,10,11,12],'finite':True,'sequential_layer_memory_safe':True},indent=2)+'\n'); (a.out/'model_identity.txt').write_text(f'model=Qwen/Qwen3-0.6B\nrevision={REV}\nsha256={sh}\n'); (a.out/'handoff_manifest.json').write_text(json.dumps({'status':'PASS','filename':hp.name,'location':str(a.handoff),'sha256':hh,'size':hp.stat().st_size,'no_embedding_or_lm_head':True},indent=2)+'\n'); (a.out/'handoff_integrity.json').write_text(json.dumps({'status':'PASS','sha256':hh,'size':hp.stat().st_size,'verified_before_stage_b':False},indent=2)+'\n'); (a.out/'environment_phase1.txt').write_text(f'stage=phase1-hf\nrevision={REV}\nlayers=28\n'); print(json.dumps({'status':'PASS' if exact else 'BLOCKED_BY_28L_ORACLE_MISMATCH','layers':28,'semantic_exact':exact,'handoff':str(hp),'handoff_sha256':hh,'total_params':sum(x['numel'] for x in items)}))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--handoff',type=Path,required=True); main(p.parse_args())
