from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
import onnx, tensorrt as trt
from portable_qwen3_layer import PortableQwen3Layer0

def metric(a,b):
 d=a.float()-b.float(); return {'shape_equal':list(a.shape)==list(b.shape),'finite':bool(torch.isfinite(a).all().item()),'max_abs':float(d.abs().max().item()),'mean_abs':float(d.abs().mean().item()),'rmse':float(torch.sqrt((d*d).mean()).item())}
class Pre(torch.nn.Module):
 def __init__(self,m): super().__init__(); self.m=m
 def forward(self,x,p): return self.m.forward_prefill(x,p)
class Dec(torch.nn.Module):
 def __init__(self,m): super().__init__(); self.m=m
 def forward(self,x,p,k,v): return self.m.forward_decode(x,p,k,v)
class RT:
 def __init__(self,path):
  self.engine=trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes()); self.ctx=self.engine.create_execution_context(); self.stream=torch.cuda.current_stream()
 def run(self,inputs):
  for n,t in inputs.items(): self.ctx.set_input_shape(n,tuple(t.shape)); self.ctx.set_tensor_address(n,t.data_ptr())
  out={}
  for i in range(self.engine.num_io_tensors):
   n=self.engine.get_tensor_name(i)
   if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT:
    sh=tuple(self.ctx.get_tensor_shape(n)); dt=torch.float16 if self.engine.get_tensor_dtype(n)==trt.DataType.HALF else torch.float32; out[n]=torch.empty(sh,device='cuda',dtype=dt); self.ctx.set_tensor_address(n,out[n].data_ptr())
  assert self.ctx.execute_async_v3(self.stream.cuda_stream); self.stream.synchronize(); return out
def build(path,out,kind):
 net=trt.Builder(trt.Logger(trt.Logger.WARNING)).create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); parser=trt.OnnxParser(net,trt.Logger(trt.Logger.WARNING)); model=onnx.load(str(path)); ok=parser.parse(model.SerializeToString()); errs=[str(parser.get_error(i)) for i in range(parser.num_errors)]; assert ok,errs; b=trt.Builder(trt.Logger(trt.Logger.WARNING)); cfg=b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); prof=b.create_optimization_profile();
 if kind=='pre': prof.set_shape('hidden_states',(1,1,1024),(1,8,1024),(1,16,1024)); prof.set_shape('position_ids',(1,1),(1,8),(1,16))
 else: prof.set_shape('hidden_states',(1,1,1024),(1,1,1024),(1,1,1024)); prof.set_shape('position_ids',(1,1),(1,1),(1,1)); prof.set_shape('past_k',(1,8,1,128),(1,8,8,128),(1,8,16,128)); prof.set_shape('past_v',(1,8,1,128),(1,8,8,128),(1,8,16,128))
 cfg.add_optimization_profile(prof); blob=b.build_serialized_network(net,cfg); assert blob is not None; out.write_bytes(bytes(blob)); return {'parse':'PASS','build':'PASS','bytes':out.stat().st_size,'parser_errors':errs}
def main(a):
 h=torch.load(a.handoff,map_location='cpu',weights_only=True); state=h['state_dict']; bf=PortableQwen3Layer0().to('cuda',dtype=torch.bfloat16).eval(); bf.load_state_dict(state); fp=PortableQwen3Layer0().to('cuda',dtype=torch.float16).eval(); fp.load_state_dict({k:v.half() for k,v in state.items()}); x=h['x'].cuda().to(torch.bfloat16); pos=h['pos'].cuda(); steps=[z.cuda().to(torch.bfloat16) for z in h['decode_inputs']];
 with torch.inference_mode():
  bh,bk,bv=bf.forward_prefill(x,pos); fh,fk,fv=fp.forward_prefill(x.half(),pos); aout= h['hf_prefill']; three={'prefill':{'hf_bf16_vs_portable_bf16':{'hidden':metric(aout['hidden'],bh.cpu()),'k':metric(aout['k'],bk.cpu()),'v':metric(aout['v'],bv.cpu())},'portable_bf16_vs_fp16':{'hidden':metric(bh,fh),'k':metric(bk,fk),'v':metric(bv,fv)}}};
  bf_k,bf_v=bk,bv; fp_k,fp_v=fk,fv; dec=[]
 for i,z in enumerate(steps):
   pp=torch.tensor([[8+i]],device='cuda',dtype=torch.long); bh,bf_k,bf_v=bf.forward_decode(z,pp,bf_k,bf_v); fh,fp_k,fp_v=fp.forward_decode(z.half(),pp,fp_k,fp_v); dec.append({'step':i,'portable_bf16_vs_fp16':{'hidden':metric(bh,fh),'k':metric(bf_k,fp_k),'v':metric(bf_v,fp_v)},'portable_fp16':{'hidden':fh.cpu(),'k':fp_k.cpu(),'v':fp_v.cpu()},'portable_bf16':{'hidden':bh.cpu(),'k':bf_k.cpu(),'v':bf_v.cpu()}})
 pre=Pre(fp).cuda().eval(); pe=a.tmp/'prefill.onnx'; torch.onnx.export(pre,(x.half(),pos),pe,opset_version=17,input_names=['hidden_states','position_ids'],output_names=['hidden_out','present_k','present_v'],dynamic_axes={'hidden_states':{0:'batch',1:'seq'},'position_ids':{0:'batch',1:'seq'},'hidden_out':{0:'batch',1:'seq'},'present_k':{0:'batch',2:'seq'},'present_v':{0:'batch',2:'seq'}}); de=Dec(fp).cuda().eval(); de_path=a.tmp/'decode.onnx'; pk=torch.randn((1,8,8,128),device='cuda',dtype=torch.float16); torch.onnx.export(de,(x[:,:1].half(),pos[:,:1],pk,pk),de_path,opset_version=17,input_names=['hidden_states','position_ids','past_k','past_v'],output_names=['hidden_out','present_k','present_v'],dynamic_axes={'hidden_states':{0:'batch'},'position_ids':{0:'batch'},'past_k':{0:'batch',2:'past_len'},'past_v':{0:'batch',2:'past_len'},'hidden_out':{0:'batch'},'present_k':{0:'batch',2:'present_len'},'present_v':{0:'batch',2:'present_len'}})
 ps=build(pe,a.tmp/'prefill.engine','pre'); ds=build(de_path,a.tmp/'decode.engine','dec'); pro=RT(a.tmp/'prefill.engine').run({'hidden_states':x.half(),'position_ids':pos}); trt_pre={'hidden':metric(fh,pro['hidden_out']),'k':metric(fk,pro['present_k']),'v':metric(fv,pro['present_v'])}; tk,tv=pro['present_k'],pro['present_v']; trt_dec=[]
 for i,z in enumerate(steps):
   pp=torch.tensor([[8+i]],device='cuda',dtype=torch.long); o=RT(a.tmp/'decode.engine').run({'hidden_states':z.half(),'position_ids':pp,'past_k':tk,'past_v':tv}); oldk,oldv=tk,tv; tk,tv=o['present_k'],o['present_v']; ref=dec[i]['portable_fp16']; trt_dec.append({'step':i,'past_length':oldk.shape[2],'present_length':tk.shape[2],'hidden':metric(ref['hidden'],o['hidden_out'].cpu()),'k':metric(ref['k'],tk.cpu()),'v':metric(ref['v'],tv.cpu()),'prefix_k':metric(tk[:,:,:oldk.shape[2],:],oldk),'prefix_v':metric(tv[:,:,:oldv.shape[2],:],oldv),'new_k':metric(tk[:,:,-1:,:],ref['k'][:,:,-1:]),'new_v':metric(tv[:,:,-1:,:],ref['v'][:,:,-1:]),'finite':bool(torch.isfinite(tk).all() and torch.isfinite(tv).all() and torch.isfinite(o['hidden_out']).all())})
 a.out.mkdir(parents=True,exist_ok=True); (a.out/'portable_bf16_semantic_comparison.json').write_text(json.dumps(three,indent=2)); (a.out/'trt_build_result.json').write_text(json.dumps({'prefill':ps,'decode':ds},indent=2)); (a.out/'prefill_three_way_accuracy.json').write_text(json.dumps({'portable_fp16_vs_trt_fp16':trt_pre,'hf_bf16_vs_trt_fp16':{'hidden':metric(aout['hidden'],pro['hidden_out'].cpu()),'k':metric(aout['k'],pro['present_k'].cpu()),'v':metric(aout['v'],pro['present_v'].cpu())}},indent=2)); (a.out/'decode_three_way_accuracy.json').write_text(json.dumps(trt_dec,indent=2)); print(json.dumps({'status':'PASS','prefill':trt_pre,'decode':trt_dec},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--handoff',type=Path,required=True); p.add_argument('--tmp',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.tmp.mkdir(parents=True,exist_ok=True); main(a)
