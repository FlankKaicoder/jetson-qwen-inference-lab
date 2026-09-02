import argparse,json
from pathlib import Path
import torch
from qwen3_layer_semantics import CONFIG,make_layer
from trt_runtime import TRTRuntime

def metrics(a,b):
    d=(a.float()-b.float()); return {'shape_equal':list(a.shape)==list(b.shape),'finite':bool(torch.isfinite(a).all().item()),'max_abs_error':float(d.abs().max().item()),'mean_abs_error':float(d.abs().mean().item()),'rmse':float(torch.sqrt(torch.mean(d*d)).item())}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--prefill-engine',type=Path,required=True); p.add_argument('--decode-engine',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True); a=p.parse_args(); torch.manual_seed(77); m=make_layer(device='cuda'); x=torch.randn((1,8,1024),device='cuda',dtype=torch.float16); pos=torch.arange(8,device='cuda',dtype=torch.long).unsqueeze(0)
    with torch.no_grad(): ref_h,past_k,past_v=m.forward_prefill(x,pos)
    pre=TRTRuntime(a.prefill_engine).execute({'hidden_states':x,'position_ids':pos}); result={'prefill':{'hidden':metrics(pre['hidden_out'],ref_h),'k':metrics(pre['present_k'],past_k),'v':metrics(pre['present_v'],past_v),'shapes':{k:list(v.shape) for k,v in pre.items()},'cuda_devices':{k:str(v.device) for k,v in pre.items()}}}; cur_k,cur_v=past_k,past_v; trt_k,trt_v=pre['present_k'],pre['present_v']; steps=[]
    for step in range(4):
        hx=torch.randn((1,1,1024),device='cuda',dtype=torch.float16); pp=torch.tensor([[8+step]],device='cuda',dtype=torch.long)
        with torch.no_grad(): rh,rk,rv=m.forward_decode(hx,pp,cur_k,cur_v)
        ref_prefix_k,ref_prefix_v=cur_k.clone(),cur_v.clone()
        out=TRTRuntime(a.decode_engine).execute({'hidden_states':hx,'position_ids':pp,'past_k':trt_k,'past_v':trt_v}); pk,pv=out['present_k'],out['present_v'];
        prefix_k=metrics(pk[:,:,:trt_k.shape[2],:],trt_k); prefix_v=metrics(pv[:,:,:trt_v.shape[2],:],trt_v)
        ref_prefix_k_check=metrics(rk[:,:,:cur_k.shape[2],:],ref_prefix_k); ref_prefix_v_check=metrics(rv[:,:,:cur_v.shape[2],:],ref_prefix_v)
        new_k=metrics(pk[:,:,trt_k.shape[2]:,:],rk[:,:,cur_k.shape[2]:,:]); new_v=metrics(pv[:,:,trt_v.shape[2]:,:],rv[:,:,cur_v.shape[2]:,:])
        steps.append({'step':step,'past_length':cur_k.shape[2],'present_length':pk.shape[2],'hidden':metrics(out['hidden_out'],rh),'k':metrics(pk,rk),'v':metrics(pv,rv),'prefix_k':prefix_k,'prefix_v':prefix_v,'reference_prefix_k':ref_prefix_k_check,'reference_prefix_v':ref_prefix_v_check,'new_slot_k':new_k,'new_slot_v':new_v,'devices':{k:str(v.device) for k,v in out.items()}}); cur_k,cur_v=rk,rv; trt_k,trt_v=pk,pv
    result['decode_steps']=steps; result['stream']='torch.cuda.current_stream() owned by runtime; explicit synchronize after execute'; result['host_payload_roundtrip']=False; result['cache_prefix_input_immutability']=all(s['prefix_k']['max_abs_error']==0.0 and s['prefix_v']['max_abs_error']==0.0 for s in steps); result['all_outputs_finite']=all(x['finite'] for x in [result['prefill']['hidden'],result['prefill']['k'],result['prefill']['v']] + [q for s in steps for q in (s['hidden'],s['k'],s['v'],s['prefix_k'],s['prefix_v'],s['new_slot_k'],s['new_slot_v'])]); a.out_dir.mkdir(parents=True,exist_ok=True); (a.out_dir/'runtime_validation.json').write_text(json.dumps(result,indent=2)+'\n'); (a.out_dir/'cache_growth.json').write_text(json.dumps([{'past_length':s['past_length'],'present_length':s['present_length']} for s in steps],indent=2)+'\n'); (a.out_dir/'cache_accuracy.json').write_text(json.dumps({'prefill':result['prefill'],'decode_steps':steps,'cache_prefix_input_immutability':result['cache_prefix_input_immutability'],'all_outputs_finite':result['all_outputs_finite']},indent=2)+'\n'); print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
