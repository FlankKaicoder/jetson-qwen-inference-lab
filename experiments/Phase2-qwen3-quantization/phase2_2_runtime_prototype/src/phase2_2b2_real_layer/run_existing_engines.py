from __future__ import annotations
import argparse,json
from pathlib import Path
import torch,tensorrt as trt
class RT:
 def __init__(self,p): self.e=trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(p.read_bytes()); self.c=self.e.create_execution_context(); self.s=torch.cuda.current_stream()
 def run(self,d):
  for n,t in d.items(): self.c.set_input_shape(n,tuple(t.shape)); self.c.set_tensor_address(n,t.data_ptr())
  o={}
  for i in range(self.e.num_io_tensors):
   n=self.e.get_tensor_name(i)
   if self.e.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT:
    sh=tuple(self.c.get_tensor_shape(n)); o[n]=torch.empty(sh,device='cuda',dtype=torch.float16); self.c.set_tensor_address(n,o[n].data_ptr())
  assert self.c.execute_async_v3(self.s.cuda_stream); self.s.synchronize(); return o
def m(a,b):
 d=a.float()-b.float(); return {'max_abs':float(d.abs().max()),'rmse':float(torch.sqrt((d*d).mean())),'finite':bool(torch.isfinite(a).all()),'shape_equal':list(a.shape)==list(b.shape)}
def main(a):
 h=torch.load(a.handoff,map_location='cpu',weights_only=True); mod=__import__('portable_qwen3_layer'); fp=mod.PortableQwen3Layer0().cuda().half().eval(); fp.load_state_dict({k:v.half() for k,v in h['state_dict'].items()}); x=h['x'].cuda().half(); pos=h['pos'].cuda(); steps=[z.cuda().half() for z in h['decode_inputs']]
 with torch.inference_mode():
  fh,fk,fv=fp.forward_prefill(x,pos); prefill_ref=(fh,fk,fv); pre=RT(a.prefill).run({'hidden_states':x,'position_ids':pos}); rows=[]; tk,tv=pre['present_k'],pre['present_v']; pk,pv=fk,fv; hf_pre=h['hf_prefill']
  for i,z in enumerate(steps):
   pp=torch.tensor([[8+i]],device='cuda',dtype=torch.long); fh,pk,pv=fp.forward_decode(z,pp,pk,pv); oldk,oldv=tk,tv; o=RT(a.decode).run({'hidden_states':z,'position_ids':pp,'past_k':tk,'past_v':tv}); tk,tv=o['present_k'],o['present_v']; hf=h['hf_decode'][i]; rows.append({'step':i,'past_length':oldk.shape[2],'present_length':tk.shape[2],'hidden':m(o['hidden_out'],fh),'k':m(tk,pk),'v':m(tv,pv),'hf_bf16_vs_trt_fp16':{'hidden':m(o['hidden_out'].cpu(),hf[0]),'k':m(tk.cpu(),hf[1]),'v':m(tv.cpu(),hf[2])},'prefix_k':m(tk[:,:,:oldk.shape[2]],oldk),'prefix_v':m(tv[:,:,:oldv.shape[2]],oldv),'new_k':m(tk[:,:,-1:],pk[:,:,-1:]),'new_v':m(tv[:,:,-1:],pv[:,:,-1:]),'devices':{n:str(t.device) for n,t in o.items()}})
 result={'status':'PASS','prefill':{'hidden':m(pre['hidden_out'],prefill_ref[0]),'k':m(pre['present_k'],prefill_ref[1]),'v':m(pre['present_v'],prefill_ref[2]),'shapes':{n:list(t.shape) for n,t in pre.items()},'devices':{n:str(t.device) for n,t in pre.items()}},'decode':rows,'prefix_unchanged':all(r['prefix_k']['max_abs']==0 and r['prefix_v']['max_abs']==0 for r in rows),'all_finite':all(r['hidden']['finite'] and r['k']['finite'] and r['v']['finite'] for r in rows),'host_payload_roundtrip':False,'stream':'torch.cuda.current_stream with explicit synchronize'}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--handoff',type=Path,required=True); p.add_argument('--prefill',type=Path,required=True); p.add_argument('--decode',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); main(a)
