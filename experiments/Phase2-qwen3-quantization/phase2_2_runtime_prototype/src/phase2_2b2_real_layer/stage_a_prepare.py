from __future__ import annotations
import argparse, hashlib, json, inspect
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
from transformers.cache_utils import DynamicCache
from portable_qwen3_layer import PortableQwen3Layer0

MODEL=Path('/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca'); EXPECTED='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
REQ=['input_layernorm.weight','self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight','self_attn.q_norm.weight','self_attn.k_norm.weight','post_attention_layernorm.weight','mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']
def sha(b): return hashlib.sha256(b).hexdigest()
def metric(a,b):
 d=a.float()-b.float(); return {'shape_equal':list(a.shape)==list(b.shape),'finite':bool(torch.isfinite(a).all().item()),'max_abs':float(d.abs().max().item()),'mean_abs':float(d.abs().mean().item()),'rmse':float(torch.sqrt((d*d).mean()).item())}
def hf_run(layer,rot,x,pos,past=None):
 b,s,_=x.shape; cache=DynamicCache();
 if past is not None: cache.update(past[0],past[1],0,{'cache_position':torch.arange(past[0].shape[2],device=x.device)})
 mask=torch.zeros((b,1,s,(past[0].shape[2]+s if past else s)),device=x.device,dtype=x.dtype)
 if past is None: mask=mask.masked_fill(torch.triu(torch.ones((s,s),device=x.device,dtype=torch.bool),1)[None,None],torch.finfo(x.dtype).min)
 pe=rot(x,pos)
 out=layer(x,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,cache_position=torch.arange((past[0].shape[2] if past else 0),(past[0].shape[2] if past else 0)+s,device=x.device),position_embeddings=pe)
 k,v=cache[0]; return out,k,v
def main(out):
 out.mkdir(parents=True,exist_ok=True); ck=MODEL/'model.safetensors'; digest=sha(ck.read_bytes()); assert digest==EXPECTED, (digest,EXPECTED)
 cfg=Qwen3Config.from_pretrained(str(MODEL)); cfg._attn_implementation='eager'; layer=Qwen3DecoderLayer(cfg,layer_idx=0).to(device='cuda',dtype=torch.bfloat16).eval(); rot=Qwen3RotaryEmbedding(cfg).to('cuda')
 state={}; manifest=[]
 with safe_open(str(ck),framework='pt',device='cpu') as f:
  for mod in REQ:
   key='model.layers.0.'+mod; t=f.get_tensor(key); state[mod]=t; manifest.append({'checkpoint_key':key,'module_key':mod,'shape':list(t.shape),'dtype':str(t.dtype),'numel':t.numel(),'bytes':t.numel()*t.element_size(),'sha256':sha(t.contiguous().view(torch.uint8).numpy().tobytes())})
 layer.load_state_dict(state,strict=True); portable=PortableQwen3Layer0().to('cuda',dtype=torch.bfloat16).eval(); portable.load_state_dict(state,strict=True)
 torch.manual_seed(0); x=torch.randn((1,8,1024),device='cuda',dtype=torch.bfloat16); pos=torch.arange(8,device='cuda',dtype=torch.long).unsqueeze(0); steps=[torch.randn((1,1,1024),device='cuda',dtype=torch.bfloat16) for _ in range(4)]
 with torch.inference_mode():
  ah,ak,av=hf_run(layer,rot,x,pos); hf_prefill=(ah,ak,av); bh,bk,bv=portable.forward_prefill(x,pos); portable_prefill=(bh,bk,bv)
  rows=[{'phase':'prefill','hidden':metric(ah,bh),'k':metric(ak,bk),'v':metric(av,bv)}]; a_k,a_v=ak,av; b_k,b_v=bk,bv; dec=[]
  hf_decode=[]; portable_decode=[]
  for i,hx in enumerate(steps):
   pp=torch.tensor([[8+i]],device='cuda',dtype=torch.long); ah,a_k,a_v=hf_run(layer,rot,hx,pp,(a_k,a_v)); bh,b_k,b_v=portable.forward_decode(hx,pp,b_k,b_v); hf_decode.append((ah.cpu(),a_k.cpu(),a_v.cpu())); portable_decode.append((bh.cpu(),b_k.cpu(),b_v.cpu())); dec.append({'step':i,'past_length':8+i,'present_length':9+i,'hidden':metric(ah,bh),'k':metric(a_k,b_k),'v':metric(a_v,b_v),'new_k':metric(a_k[:,:,-1:],b_k[:,:,-1:]),'new_v':metric(a_v[:,:,-1:],b_v[:,:,-1:])})
 torch.save({'state_dict':{k:v.cpu().contiguous() for k,v in state.items()},'x':x.cpu(),'pos':pos.cpu(),'decode_inputs':[z.cpu() for z in steps],'hf_prefill':{'hidden':hf_prefill[0].cpu(),'k':hf_prefill[1].cpu(),'v':hf_prefill[2].cpu()},'hf_decode':hf_decode,'portable_prefill':{'hidden':portable_prefill[0].cpu(),'k':portable_prefill[1].cpu(),'v':portable_prefill[2].cpu()},'portable_decode':portable_decode,'model_revision':'c1899de289a04d12100db370d81485cdf75e47ca'},out/'layer0_handoff.pt')
 (out/'layer0_weight_manifest.json').write_text(json.dumps({'model_sha256':digest,'items':manifest,'total_params':sum(t.numel() for t in state.values()),'total_bf16_bytes':sum(t.numel()*t.element_size() for t in state.values())},indent=2)+'\n'); (out/'portable_semantic_comparison.json').write_text(json.dumps({'prefill':rows[0],'decode':dec},indent=2)+'\n'); (out/'model_identity.txt').write_text(f'model=Qwen/Qwen3-0.6B\nrevision=c1899de289a04d12100db370d81485cdf75e47ca\nsnapshot={MODEL}\nsha256={digest}\n'); (out/'transformers_source_identity.txt').write_text('module=transformers.models.qwen3.modeling_qwen3\nclasses=Qwen3DecoderLayer,Qwen3Attention,Qwen3RMSNorm,Qwen3MLP,Qwen3RotaryEmbedding\nfunctions=apply_rotary_pos_emb,repeat_kv\n'); print(json.dumps({'status':'PASS','manifest':len(manifest),'semantic':rows[0],'decode':dec},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); main(a.out)
