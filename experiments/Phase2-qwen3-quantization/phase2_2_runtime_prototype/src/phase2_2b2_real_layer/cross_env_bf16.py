import argparse,json
from pathlib import Path
import torch
from portable_qwen3_layer import PortableQwen3Layer0
def m(a,b):
 d=a.float()-b.float(); return {'max_abs':float(d.abs().max()),'rmse':float(torch.sqrt((d*d).mean())),'finite':bool(torch.isfinite(a).all()),'shape_equal':list(a.shape)==list(b.shape)}
def main(a):
 h=torch.load(a.handoff,map_location='cpu',weights_only=True); mod=PortableQwen3Layer0().eval().to('cuda',dtype=torch.bfloat16); mod.load_state_dict(h['state_dict']); x=h['x'].cuda(); pos=h['pos'].cuda(); rows=[]
 with torch.inference_mode():
  oh,ok,ov=mod.forward_prefill(x,pos); ref=h['hf_prefill']; rows.append({'phase':'prefill','hidden':m(oh.cpu(),ref['hidden']),'k':m(ok.cpu(),ref['k']),'v':m(ov.cpu(),ref['v'])}); ck,cv=ok,ov
  for i,z in enumerate(h['decode_inputs']):
   pp=torch.tensor([[8+i]],device='cuda'); oh,ck,cv=mod.forward_decode(z.cuda(),pp,ck,cv); ref=h['hf_decode'][i]; rows.append({'phase':'decode','step':i,'hidden':m(oh.cpu(),ref[0]),'k':m(ck.cpu(),ref[1]),'v':m(cv.cpu(),ref[2])})
 out={'status':'PASS','environment':'phase2-trt-tools','comparisons':rows,'all_zero':all(v['max_abs']==0 for r in rows for k in ('hidden','k','v') for v in (r[k],))}; a.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--handoff',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); main(a)
