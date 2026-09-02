"""C1I upstream Q/K, per-head norm and RoPE numerical isolation."""
from __future__ import annotations
import argparse, gc, hashlib, json, resource, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
B3 = HERE.parent / "phase2_2b3_real_stack"
sys.path.insert(0, str(B3))
from portable_qwen3_stack import PortableQwen3Layer

def sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def metric(a, b):
    x, y = a.detach().float().cpu(), b.detach().float().cpu(); d = x-y
    xn, yn = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y); tiny = torch.finfo(torch.float32).tiny
    return {"shape":list(x.shape),"candidate_shape":list(y.shape),"shape_equal":list(x.shape)==list(y.shape),"dtype":str(a.dtype),"candidate_dtype":str(b.dtype),"finite":bool(torch.isfinite(x).all() and torch.isfinite(y).all()),"max_abs":float(d.abs().max()),"mean_abs":float(d.abs().mean()),"rmse":float(torch.sqrt((d*d).mean())),"relative_l2":float(torch.linalg.vector_norm(d)/torch.clamp(xn,min=tiny)),"cosine":float(torch.dot(x.reshape(-1),y.reshape(-1))/torch.clamp(xn*yn,min=tiny)),"sha256":sha(a),"candidate_sha256":sha(b)}
def snap(stage):
    info={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1); info[k]=int(v.split()[0])*1024
    free,total=torch.cuda.mem_get_info()
    rss=next((x.split(':',1)[1].strip() for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')),None)
    return {"stage":stage,"mem_available_bytes":info.get('MemAvailable'),"vmrss":rss,"cuda_free_bytes":free,"cuda_total_bytes":total,"torch_allocated_bytes":torch.cuda.memory_allocated(),"torch_reserved_bytes":torch.cuda.memory_reserved(),"maxrss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

def rms(x, w, eps=1e-6):
    y=x.float(); return w * (y*torch.rsqrt(y.pow(2).mean(-1,keepdim=True)+eps)).to(x.dtype)
def rope(x, pos, theta=1_000_000.0, fp32=False):
    d=x.shape[-1]; inv=torch.pow(torch.tensor(theta,device=x.device,dtype=torch.float32),-torch.arange(d//2,device=x.device,dtype=torch.float32)/(d//2)); freq=pos.float().unsqueeze(-1)*inv; emb=torch.cat((freq,freq),dim=-1); c=emb.cos().unsqueeze(1); s=emb.sin().unsqueeze(1); half=d//2; rot=torch.cat((-x[...,half:],x[...,:half]),dim=-1)
    if fp32: return (x.float()*c + rot.float()*s).to(x.dtype)
    return x*c.to(x.dtype)+rot*s.to(x.dtype)

def upstream(layer, hidden, pos, mode='native'):
    rms_out=rms(hidden,layer.input_layernorm.weight,layer.input_layernorm.eps)
    b,s,_=rms_out.shape; q=layer.self_attn.q_proj(rms_out).view(b,s,layer.q_heads,layer.head_dim).transpose(1,2); k=layer.self_attn.k_proj(rms_out).view(b,s,layer.kv_heads,layer.head_dim).transpose(1,2)
    qn=rms(q,layer.self_attn.q_norm.weight,layer.self_attn.q_norm.eps); kn=rms(k,layer.self_attn.k_norm.weight,layer.self_attn.k_norm.eps)
    if mode=='rms_fp32':
        rms_out=rms(hidden.float(),layer.input_layernorm.weight.float(),layer.input_layernorm.eps).to(hidden.dtype)
        q=layer.self_attn.q_proj(rms_out).view(b,s,layer.q_heads,layer.head_dim).transpose(1,2)
        k=layer.self_attn.k_proj(rms_out).view(b,s,layer.kv_heads,layer.head_dim).transpose(1,2)
        qn=rms(q,layer.self_attn.q_norm.weight,layer.self_attn.q_norm.eps); kn=rms(k,layer.self_attn.k_norm.weight,layer.self_attn.k_norm.eps)
    if mode=='qnorm_fp32': qn=rms(q.float(),layer.self_attn.q_norm.weight.float(),layer.self_attn.q_norm.eps).to(q.dtype)
    if mode=='knorm_fp32': kn=rms(k.float(),layer.self_attn.k_norm.weight.float(),layer.self_attn.k_norm.eps).to(k.dtype)
    qr=rope(qn,pos,fp32=mode in ('qrope_fp32','rope_fp32')); kr=rope(kn,pos,fp32=mode in ('krope_fp32','rope_fp32'))
    kr_rep=kr[:, :, None].expand(b,layer.kv_heads,layer.groups,s,layer.head_dim).reshape(b,layer.q_heads,s,layer.head_dim)
    raw=torch.matmul(qr,kr_rep.transpose(-2,-1)); scaled=raw*(layer.head_dim**-0.5)
    mask=torch.triu(torch.ones((s,s),device=hidden.device,dtype=torch.bool),diagonal=1); scores=scaled.masked_fill(mask,torch.finfo(scaled.dtype).min); probs=torch.softmax(scores,dim=-1,dtype=torch.float32).to(qr.dtype); vh=layer.self_attn.v_proj(rms_out).view(b,s,layer.kv_heads,layer.head_dim).transpose(1,2); vr=vh[:, :, None].expand(b,layer.kv_heads,layer.groups,s,layer.head_dim).reshape(b,layer.q_heads,s,layer.head_dim); ctx=torch.matmul(probs,vr).transpose(1,2).contiguous().reshape(b,s,-1); op=layer.self_attn.o_proj(ctx); ar=hidden+op; post=layer.post_attention_layernorm(ar); down=layer.mlp.down_proj(torch.nn.functional.silu(layer.mlp.gate_proj(post))*layer.mlp.up_proj(post))
    return {'input_rmsnorm':rms_out,'q_projection':q,'k_projection':k,'q_pre_norm':q.clone(),'k_pre_norm':k.clone(),'q_norm':qn,'k_norm':kn,'q_rope':qr,'k_rope':kr,'k_repeat':kr_rep,'qk_raw':raw,'qk_scaled':scaled,'final_hidden':ar+down}

class LayerProbe(torch.nn.Module):
    names=['input_rmsnorm','q_projection','k_projection','q_pre_norm','k_pre_norm','q_norm','k_norm','q_rope','k_rope','k_repeat','qk_raw','qk_scaled','final_hidden']
    def __init__(self,layer,mode='native'): super().__init__(); self.layer=layer; self.mode=mode
    def forward(self,hidden,pos):
        v=upstream(self.layer,hidden,pos,self.mode); return tuple(v[n] for n in self.names)
class LayerFinalProbe(torch.nn.Module):
    def __init__(self,layer,mode='native'): super().__init__(); self.layer=layer; self.mode=mode
    def forward(self,hidden,pos): return upstream(self.layer,hidden,pos,self.mode)['final_hidden']
class RMSMicro(torch.nn.Module):
    def __init__(self,w,eps): super().__init__(); self.register_buffer('w',w); self.eps=eps
    def forward(self,x): return rms(x,self.w,self.eps)
class NormMicro(RMSMicro): pass
class RopeMicro(torch.nn.Module):
    def __init__(self,theta=1_000_000.0,fp32=False): super().__init__(); self.theta=theta; self.fp32=fp32
    def forward(self,x,pos): return rope(x,pos,self.theta,self.fp32)

class TRT:
    def __init__(self,path):
        import tensorrt as trt
        self.trt=trt; self.engine=trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(Path(path).read_bytes()); self.stream=torch.cuda.current_stream()
        if self.engine is None: raise RuntimeError('ENGINE_DESERIALIZE_FAILED')
    def run(self,inputs):
        c=self.engine.create_execution_context(); out={}
        for n,t in inputs.items():
            if not c.set_input_shape(n,tuple(t.shape)): raise RuntimeError('INPUT_SHAPE_REJECTED:'+n)
            c.set_tensor_address(n,t.data_ptr())
        for i in range(self.engine.num_io_tensors):
            n=self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n)!=self.trt.TensorIOMode.OUTPUT: continue
            sh=tuple(c.get_tensor_shape(n)); dt=self.engine.get_tensor_dtype(n); td=torch.float16 if dt==self.trt.DataType.HALF else torch.float32; out[n]=torch.empty(sh,device='cuda',dtype=td); c.set_tensor_address(n,out[n].data_ptr())
        if not c.execute_async_v3(self.stream.cuda_stream): raise RuntimeError('EXECUTE_FAILED')
        self.stream.synchronize(); return out
    def inspector(self):
        try:
            return self.engine.create_engine_inspector().get_engine_information(self.trt.LayerInformationFormat.JSON)
        except Exception as e: return 'INSPECTOR_UNAVAILABLE:'+str(e)

def export_graph(module,path,args,inputs,outputs,dyn):
    import onnx
    torch.onnx.export(module,args,str(path),opset_version=17,input_names=inputs,output_names=list(outputs),dynamic_axes=dyn)
    m=onnx.load(str(path)); onnx.checker.check_model(m); return {'status':'PASS','bytes':path.stat().st_size,'nodes':len(m.graph.node)}
def build_graph(onnx_path,engine_path,profiles):
    import tensorrt as trt
    b=trt.Builder(trt.Logger(trt.Logger.WARNING)); net=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); p=trt.OnnxParser(net,trt.Logger(trt.Logger.WARNING))
    if not p.parse(Path(onnx_path).read_bytes()): raise RuntimeError('PARSER_FAILED:'+str([str(p.get_error(i)) for i in range(p.num_errors)]))
    cfg=b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); prof=b.create_optimization_profile()
    for n,(mn,opt,mx) in profiles.items(): prof.set_shape(n,mn,opt,mx)
    cfg.add_optimization_profile(prof); blob=b.build_serialized_network(net,cfg)
    if blob is None: raise RuntimeError('BUILD_FAILED')
    Path(engine_path).write_bytes(bytes(blob)); return {'status':'PASS','bytes':Path(engine_path).stat().st_size}
def load_layer(path):
    l=PortableQwen3Layer().to(device='cuda',dtype=torch.float16).eval(); st=torch.load(path,map_location='cpu',weights_only=True); l.load_state_dict({k:v.half() for k,v in st.items()},strict=True); return l
def heads(a,b):
    vals=[]
    for i in range(a.shape[1]): vals.append(float(torch.linalg.vector_norm((a[:,i]-b[:,i]).float())/torch.clamp(torch.linalg.vector_norm(a[:,i].float()),min=torch.finfo(torch.float32).tiny)))
    ordered=sorted(enumerate(vals), key=lambda item: item[1]); only=[v for _,v in ordered]
    return {'min':only[0],'median':only[len(only)//2],'max':only[-1],'worst_head':int(ordered[-1][0])}

def main(a):
    a.tmp.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True); trace=[snap('start')]
    payload=torch.load(a.embedding_reference,map_location='cpu',weights_only=True); hidden=payload['fp16']['prefill'].contiguous().cuda(); del payload; pos=torch.arange(8,device='cuda',dtype=torch.long).reshape(1,8); layer=load_layer(a.layer_file); ref=upstream(layer,hidden,pos); trace.append(snap('reference_complete'))
    control=TRT(a.b4_engine).run({'hidden_states':hidden,'position_ids':pos})['hidden_l0']; b4=metric(ref['final_hidden'],control); trace.append(snap('b4_control_complete'))
    probe_path=a.tmp/'c1i_upstream_probe.onnx'; probe_eng=a.tmp/'c1i_upstream_probe.engine'; names=[(n,4 if n in ('q_projection','k_projection','q_pre_norm','k_pre_norm','q_norm','k_norm','q_rope','k_rope','k_repeat','qk_raw','qk_scaled') else 3) for n in LayerProbe.names]; dyn={'hidden_states':{0:'batch',1:'seq'},'position_ids':{0:'batch',1:'seq'}}; dyn.update({n:{0:'batch',1:'heads_or_seq'} for n,r in names}); ps=export_graph(LayerProbe(layer).eval(),probe_path,(hidden,pos),['hidden_states','position_ids'],[n for n,r in names],dyn); bs=build_graph(probe_path,probe_eng,{'hidden_states':((1,1,1024),(1,8,1024),(1,16,1024)),'position_ids':(tuple(pos.shape),tuple(pos.shape),tuple(pos.shape))}); probe=TRT(probe_eng); got=probe.run({'hidden_states':hidden,'position_ids':pos}); probe_metrics={n:metric(ref[n],got[n]) for n,r in names}; per_head={n:heads(ref[n],got[n]) for n in ('q_projection','k_projection','q_norm','k_norm','q_rope','k_rope')}; trace.append(snap('probe_complete'))
    # Same-input micro engines for RMSNorm, Q/K norm and RoPE.
    micro={}
    micro_specs=[('input_rmsnorm',RMSMicro(layer.input_layernorm.weight,layer.input_layernorm.eps),hidden,pos),('q_norm',NormMicro(layer.self_attn.q_norm.weight,layer.self_attn.q_norm.eps),ref['q_pre_norm'].contiguous(),None),('k_norm',NormMicro(layer.self_attn.k_norm.weight,layer.self_attn.k_norm.eps),ref['k_pre_norm'].contiguous(),None),('q_rope',RopeMicro(fp32=False),ref['q_norm'].contiguous(),pos),('k_rope',RopeMicro(fp32=False),ref['k_norm'].contiguous(),pos)]
    for name,mod,x,p0 in micro_specs:
        op=a.tmp/f'micro_{name}.onnx'; ep=a.tmp/f'micro_{name}.engine'; is_rope=p0 is not None and name.endswith('rope'); ins=['x','position_ids'] if is_rope else ['x']; args=(x,p0) if is_rope else (x,); dynm={'x':{0:'batch'}}; dynm['x'].update({1:'heads_or_seq'} if x.ndim>2 else {1:'seq'}); outs=['y']; dynm['y']={0:'batch'}; dynm['y'].update({1:'heads_or_seq'} if x.ndim>2 else {1:'seq'}); 
        if is_rope: dynm['position_ids']={0:'batch',1:'seq'}
        ex=export_graph(mod.eval(),op,args,ins,outs,dynm); prof={'x':(tuple(x.shape),tuple(x.shape),tuple(x.shape))};
        if is_rope: prof['position_ids']=(tuple(p0.shape),tuple(p0.shape),tuple(p0.shape))
        eb=build_graph(op,ep,prof); out=TRT(ep).run({'x':x,**({'position_ids':p0} if is_rope else {})})['y']
        # Oracle uses the same formulas in FP32; portable is the reference checkpoint.
        if name=='input_rmsnorm': oracle=rms(x.float(),layer.input_layernorm.weight.float(),layer.input_layernorm.eps)
        elif name=='q_norm': oracle=rms(x.float(),layer.self_attn.q_norm.weight.float(),layer.self_attn.q_norm.eps)
        elif name=='k_norm': oracle=rms(x.float(),layer.self_attn.k_norm.weight.float(),layer.self_attn.k_norm.eps)
        else: oracle=rope(x.float(),p0,fp32=True)
        micro[name]={'onnx':ex,'engine':eb,'portable_vs_trt':metric(ref[name],out),'portable_vs_oracle':metric(ref[name],oracle.to(ref[name].dtype)),'trt_vs_oracle':metric(oracle.to(out.dtype),out)}
        del out,mod; gc.collect(); torch.cuda.empty_cache()
    # Build an independent A/B only for components whose same-input TRT mismatch is material.
    ab={}; threshold=1e-4; variant_modes={'input_rmsnorm':'rms_fp32','q_norm':'qnorm_fp32','k_norm':'knorm_fp32','q_rope':'qrope_fp32','k_rope':'krope_fp32'}
    for name,mode in variant_modes.items():
        if micro[name]['portable_vs_trt']['relative_l2']<=threshold: ab[name]={'status':'NOT_REQUIRED','reason':'same_input_micro_not_material'}; continue
        op=a.tmp/f'layer0_{name}_fp32.onnx'; ep=a.tmp/f'layer0_{name}_fp32.engine'; ex=export_graph(LayerFinalProbe(layer,mode).eval(),op,(hidden,pos),['hidden_states','position_ids'],['final_hidden'],{'hidden_states':{0:'batch',1:'seq'},'position_ids':{0:'batch',1:'seq'},'final_hidden':{0:'batch',1:'seq'}}); eb=build_graph(op,ep,{'hidden_states':((1,1,1024),(1,8,1024),(1,16,1024)),'position_ids':(tuple(pos.shape),tuple(pos.shape),tuple(pos.shape))}); out=TRT(ep).run({'hidden_states':hidden,'position_ids':pos})['final_hidden']; ab[name]={'status':'RUN','onnx':ex,'engine':eb,'vs_reference':metric(ref['final_hidden'],out),'vs_b4_control':metric(control,out)}; del out; gc.collect(); torch.cuda.empty_cache()
    inspector=probe.inspector(); trace.append(snap('complete'))
    growth=[]; prev=0.0
    for n in ('input_rmsnorm','q_projection','q_norm','q_rope','k_projection','k_norm','k_rope','qk_raw'):
        v=probe_metrics[n]['relative_l2']; growth.append({'stage':n,'relative_l2':v,'growth_vs_previous':v-prev}); prev=v
    result={'experiment':'Phase 2.2-C1I','input_contract':{'shape':list(hidden.shape),'dtype':str(hidden.dtype),'position_ids':list(range(8)),'hidden_sha256':sha(hidden)},'pipeline':{'P00':'hidden FP16','P01':'RMSNorm FP32 reduction then FP16','P02/P03':'Linear Q/K FP16 output','P04/P05':'reshape + transpose to [B,heads,S,D]','P06/P07':'per-head RMSNorm FP32 reduction then FP16','P08/P09':'RoPE FP32 frequency/cos/sin, cos/sin FP16 multiply','P10/P11':'RoPE Q and repeated GQA K FP16','P12':'Q@K^T FP16 raw score'},'layer_structure':{'hidden':1024,'q_heads':16,'kv_heads':8,'head_dim':128,'gqa_repeat':2,'rotary_dim':128,'rope_theta':1000000.0,'rmsnorm_eps':1e-6},'b4_2_control':b4,'probe_build':{'onnx':ps,'engine':bs,'inspector':inspector},'probe_metrics':probe_metrics,'per_head_metrics':per_head,'micro_isolation':micro,'precision_ab':ab,'error_growth':growth,'warning_mapping':'WARNING_MAPPING_NOT_CONFIRMED: existing TensorRT logs identify FP16 Reduce/Pow normalization warnings but do not map them to a specific Layer 0 reduction','memory':{'trace':trace,'oom':False,'exit137':False},'c1_status':'BLOCKED'}
    out=a.out/f'c1i_qk_rope_numerics_{a.timestamp}.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'artifact':str(out),'b4_relative_l2':b4['relative_l2'],'probe_qk_raw_relative_l2':probe_metrics['qk_raw']['relative_l2'],'micro':{k:v['portable_vs_trt']['relative_l2'] for k,v in micro.items()},'ab':{k:v.get('vs_reference',{}).get('relative_l2',v.get('status')) for k,v in ab.items()}}))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--b4-engine',type=Path,required=True); p.add_argument('--embedding-reference',type=Path,required=True); p.add_argument('--layer-file',type=Path,required=True); p.add_argument('--tmp',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--timestamp',required=True); main(p.parse_args())
