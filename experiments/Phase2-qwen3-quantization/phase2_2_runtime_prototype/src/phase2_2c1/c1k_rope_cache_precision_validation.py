"""C1K: validate an explicit portable FP16 RoPE-cache correction.

The corrected graph uses fixed FP16 cos/sin initializers generated once from
the source-faithful Qwen3 formula for positions 0..7.  This makes the cache
cast boundary observable to TensorRT and changes no attention/MLP operation.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, resource, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; B3 = HERE.parent / "phase2_2b3_real_stack"; sys.path.insert(0, str(B3))
from portable_qwen3_stack import PortableQwen3Layer

def sha(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def metric(a,b):
    x,y=a.detach().float().cpu(),b.detach().float().cpu(); d=x-y; xn,yn=torch.linalg.vector_norm(x),torch.linalg.vector_norm(y); t=torch.finfo(torch.float32).tiny
    return {"shape":list(x.shape),"candidate_shape":list(y.shape),"dtype":str(a.dtype),"candidate_dtype":str(b.dtype),"finite":bool(torch.isfinite(x).all() and torch.isfinite(y).all()),"max_abs":float(d.abs().max()),"mean_abs":float(d.abs().mean()),"rmse":float(torch.sqrt((d*d).mean())),"relative_l2":float(torch.linalg.vector_norm(d)/torch.clamp(xn,min=t)),"cosine":float(torch.dot(x.reshape(-1),y.reshape(-1))/torch.clamp(xn*yn,min=t)),"sha256":sha(a),"candidate_sha256":sha(b)}
def snap(stage):
    avail=next((int(v.split(':',1)[1].split()[0])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')),None); free,total=torch.cuda.mem_get_info()
    return {"stage":stage,"mem_available_bytes":avail,"cuda_free_bytes":free,"cuda_total_bytes":total,"torch_allocated_bytes":torch.cuda.memory_allocated(),"torch_reserved_bytes":torch.cuda.memory_reserved(),"maxrss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
def cache(pos,dim=128,theta=1_000_000.0,dtype=torch.float16):
    inv=torch.pow(torch.tensor(theta,device=pos.device,dtype=torch.float32),-torch.arange(dim//2,device=pos.device,dtype=torch.float32)/(dim//2)); f=pos.float().unsqueeze(-1)*inv; e=torch.cat((f,f),-1); return e.cos().unsqueeze(1).to(dtype),e.sin().unsqueeze(1).to(dtype),inv
def rope(x,pos):
    c,s,_=cache(pos,x.shape[-1],dtype=x.dtype); h=x.shape[-1]//2; r=torch.cat((-x[...,h:],x[...,:h]),-1); return x*c+r*s
class NativeRope(torch.nn.Module):
    def forward(self,x,pos): return rope(x,pos)
class FixedRope(torch.nn.Module):
    def __init__(self,c,s): super().__init__(); self.register_buffer('cos',c); self.register_buffer('sin',s)
    def forward(self,x):
        h=x.shape[-1]//2; r=torch.cat((-x[...,h:],x[...,:h]),-1); return x*self.cos+r*self.sin
class NativeLayer(torch.nn.Module):
    def __init__(self,l): super().__init__(); self.l=l
    def forward(self,h,pos): return run_layer(self.l,h,pos,None)
class FixedLayer(torch.nn.Module):
    def __init__(self,l,cq,sq,ck,sk): super().__init__(); self.l=l; [self.register_buffer(n,v) for n,v in [('cq',cq),('sq',sq),('ck',ck),('sk',sk)]]
    def forward(self,h):
        # The corrected graph intentionally fixes the audited positions 0..7.
        # Keeping position_ids out of this export avoids an unused ONNX input.
        pos=torch.arange(h.shape[1],device=h.device,dtype=torch.long).reshape(1,-1)
        return run_layer(self.l,h,pos,(self.cq,self.sq,self.ck,self.sk))
def run_layer(l,h,pos,fixed):
    residual=h; x=l.input_layernorm(h); b,s,_=x.shape; q=l.self_attn.q_norm(l.self_attn.q_proj(x).view(b,s,l.q_heads,l.head_dim)).transpose(1,2); k=l.self_attn.k_norm(l.self_attn.k_proj(x).view(b,s,l.kv_heads,l.head_dim)).transpose(1,2); v=l.self_attn.v_proj(x).view(b,s,l.kv_heads,l.head_dim).transpose(1,2)
    if fixed:
        cq,sq,ck,sk=fixed; hq=q.shape[-1]//2; hk=k.shape[-1]//2; q=q*cq+torch.cat((-q[...,hq:],q[...,:hq]),-1)*sq; k=k*ck+torch.cat((-k[...,hk:],k[...,:hk]),-1)*sk
    else: q,k=rope(q,pos),rope(k,pos)
    ka=k[:, :, None].expand(b,l.kv_heads,l.groups,s,l.head_dim).reshape(b,l.q_heads,s,l.head_dim); va=v[:, :, None].expand(b,l.kv_heads,l.groups,s,l.head_dim).reshape(b,l.q_heads,s,l.head_dim); z=torch.matmul(q,ka.transpose(-2,-1))*(l.head_dim**-0.5); mask=torch.triu(torch.ones((s,s),device=h.device,dtype=torch.bool),1); p=torch.softmax(z.masked_fill(mask,torch.finfo(z.dtype).min),-1,dtype=torch.float32).to(q.dtype); ctx=torch.matmul(p,va).transpose(1,2).contiguous().reshape(b,s,-1); ar=residual+l.self_attn.o_proj(ctx); post=l.post_attention_layernorm(ar); down=l.mlp.down_proj(torch.nn.functional.silu(l.mlp.gate_proj(post))*l.mlp.up_proj(post)); return ar+down
class TRT:
    def __init__(self,path):
        import tensorrt as trt; self.trt=trt; self.e=trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes()); self.stream=torch.cuda.current_stream()
        if self.e is None: raise RuntimeError('ENGINE_DESERIALIZE_FAILED')
    def run(self,ins):
        c=self.e.create_execution_context(); out={}
        for n,t in ins.items(): c.set_input_shape(n,tuple(t.shape)); c.set_tensor_address(n,t.data_ptr())
        for i in range(self.e.num_io_tensors):
            n=self.e.get_tensor_name(i)
            if self.e.get_tensor_mode(n)!=self.trt.TensorIOMode.OUTPUT: continue
            sh=tuple(c.get_tensor_shape(n)); dt=self.e.get_tensor_dtype(n); td=torch.float16 if dt==self.trt.DataType.HALF else torch.float32; out[n]=torch.empty(sh,device='cuda',dtype=td); c.set_tensor_address(n,out[n].data_ptr())
        if not c.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('EXECUTE_FAILED')
        self.stream.synchronize()
        return out
def export(mod,path,args,ins,outs,axes):
    import onnx; torch.onnx.export(mod,args,str(path),opset_version=17,input_names=ins,output_names=outs,dynamic_axes=axes); m=onnx.load(str(path)); onnx.checker.check_model(m); return {'status':'PASS','bytes':path.stat().st_size,'nodes':len(m.graph.node)}
def build(op,ep,profiles):
    import tensorrt as trt; b=trt.Builder(trt.Logger(trt.Logger.WARNING)); n=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); p=trt.OnnxParser(n,trt.Logger(trt.Logger.WARNING));
    if not p.parse(Path(op).read_bytes()): raise RuntimeError('PARSER_FAILED')
    cfg=b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); pr=b.create_optimization_profile()
    for k,v in profiles.items(): pr.set_shape(k,*v)
    cfg.add_optimization_profile(pr); blob=b.build_serialized_network(n,cfg)
    if blob is None: raise RuntimeError('BUILD_FAILED')
    Path(ep).write_bytes(bytes(blob)); return {'status':'PASS','bytes':Path(ep).stat().st_size}
def load(path):
    l=PortableQwen3Layer().to(device='cuda',dtype=torch.float16).eval(); st=torch.load(path,map_location='cpu',weights_only=True); l.load_state_dict({k:v.half() for k,v in st.items()},strict=True); return l
def main(a):
    a.tmp.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True); trace=[snap('start')]; payload=torch.load(a.embedding_reference,map_location='cpu',weights_only=True); h=payload['fp16']['prefill'].contiguous().cuda(); del payload; pos=torch.arange(8,device='cuda',dtype=torch.long).reshape(1,8); l=load(a.layer_file); x=l.input_layernorm(h); b,s,_=x.shape; q=l.self_attn.q_norm(l.self_attn.q_proj(x).view(b,s,l.q_heads,l.head_dim)).transpose(1,2).contiguous(); k=l.self_attn.k_norm(l.self_attn.k_proj(x).view(b,s,l.kv_heads,l.head_dim)).transpose(1,2).contiguous(); cq,sq,inv=cache(pos,l.head_dim,dtype=torch.float16); ck,sk,_=cache(pos,l.head_dim,dtype=torch.float16); refq,refk=rope(q,pos),rope(k,pos); trace.append(snap('reference_complete'))
    b4=TRT(a.b4_engine).run({'hidden_states':h,'position_ids':pos})['hidden_l0']; ref_layer=run_layer(l,h,pos,None); b4m=metric(ref_layer,b4)
    micro={}; axes={'x':{0:'batch'},'position_ids':{0:'batch',1:'seq'},'y':{0:'batch',1:'heads',2:'seq'}}
    for name,z,c,si,ref in [('q',q,cq,sq,refq),('k',k,ck,sk,refk)]:
        rows={}
        for mode,mod,args,ins in [('baseline',NativeRope(),(z,pos),['x','position_ids']),('corrected',FixedRope(c,si),(z,),['x'])]:
            op=a.tmp/f'c1k_{name}_{mode}.onnx'; ep=a.tmp/f'c1k_{name}_{mode}.engine'
            export_axes={'x':{0:'batch'},'y':{0:'batch',1:'heads',2:'seq'}}
            if 'position_ids' in ins: export_axes['position_ids']={0:'batch',1:'seq'}
            ex=export(mod,op,args,ins,['y'],export_axes); prof={'x':(tuple(z.shape),tuple(z.shape),tuple(z.shape))};
            if 'position_ids' in ins: prof['position_ids']=((1,1),(1,8),(1,16))
            eb=build(op,ep,prof); got=TRT(ep).run({'x':z,**({'position_ids':pos} if 'position_ids' in ins else {})})['y']; rows[mode]={'onnx':ex,'engine':eb,'vs_portable_fp16':metric(ref,got)}; del got,mod; gc.collect(); torch.cuda.empty_cache()
        micro[name]=rows
    layer_rows={}
    layer_specs=[
        ('baseline',NativeLayer(l),(h,pos),['hidden_states','position_ids'],
         {'hidden_states':{0:'batch',1:'seq'},'position_ids':{0:'batch',1:'seq'},'final_hidden':{0:'batch',1:'seq'}},
         {'hidden_states':((1,1,1024),(1,8,1024),(1,16,1024)),'position_ids':((1,1),(1,8),(1,16))},
         {'hidden_states':h,'position_ids':pos}),
        ('corrected',FixedLayer(l,cq,sq,ck,sk),(h,),['hidden_states'],
         {'hidden_states':{0:'batch',1:'seq'},'final_hidden':{0:'batch',1:'seq'}},
         {'hidden_states':((1,1,1024),(1,8,1024),(1,16,1024))},
         {'hidden_states':h}),
    ]
    for mode,mod,args,ins,layer_axes,layer_profiles,layer_inputs in layer_specs:
        ex=export(mod,op,args,ins,['final_hidden'],layer_axes)
        eb=build(op,ep,layer_profiles)
        got=TRT(ep).run(layer_inputs)['final_hidden']
        layer_rows[mode]={'onnx':ex,'engine':eb,'inputs':ins,
                          'fixed_positions':list(range(8)) if mode=='corrected' else None,
                          'vs_portable_fp16':metric(ref_layer,got),'vs_b4_control':metric(b4,got)}
        del got,mod; gc.collect(); torch.cuda.empty_cache()
    trace.append(snap('complete')); base=layer_rows['baseline']['vs_portable_fp16']['relative_l2']; corr=layer_rows['corrected']['vs_portable_fp16']['relative_l2']; result={'experiment':'Phase 2.2-C1K','input_contract':{'shape':list(h.shape),'dtype':str(h.dtype),'position_ids':list(range(8)),'hidden_sha256':sha(h)},'correction':{'design':'fixed portable cos/sin cache as FP16 graph initializers; FP16 multiply/add retained','only_changed':'RoPE cache precision/cast boundary','rotary_dim':128,'theta':1000000.0,'layout':'half_split_rotate_half'},'b4_2_control':b4m,'micro':micro,'layer0':layer_rows,'improvement':{'baseline_relative_l2':base,'corrected_relative_l2':corr,'absolute_reduction':base-corr,'factor':base/corr if corr else None},'memory':{'trace':trace,'oom':False,'exit137':False},'result':'ROPE_CACHE_FIX_VALIDATED' if corr < base*0.5 else ('ROPE_CACHE_FIX_PARTIAL' if corr < base else 'ROPE_CACHE_NOT_SUFFICIENT'),'c1_status':'BLOCKED'}; out=a.out/f'c1k_rope_cache_precision_{a.timestamp}.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'artifact':str(out),'b4':b4m['relative_l2'],'micro_q_baseline':micro['q']['baseline']['vs_portable_fp16']['relative_l2'],'micro_q_corrected':micro['q']['corrected']['vs_portable_fp16']['relative_l2'],'micro_k_baseline':micro['k']['baseline']['vs_portable_fp16']['relative_l2'],'micro_k_corrected':micro['k']['corrected']['vs_portable_fp16']['relative_l2'],'layer_baseline':base,'layer_corrected':corr,'result':result['result']}))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--b4-engine',type=Path,required=True); p.add_argument('--embedding-reference',type=Path,required=True); p.add_argument('--layer-file',type=Path,required=True); p.add_argument('--tmp',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--timestamp',required=True); main(p.parse_args())
