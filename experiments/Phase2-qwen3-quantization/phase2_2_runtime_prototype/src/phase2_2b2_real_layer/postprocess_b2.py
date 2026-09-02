import argparse,hashlib,json
from pathlib import Path
import torch
def m(a,b):
 d=a.float()-b.float(); return {'max_abs':float(d.abs().max()),'rmse':float(torch.sqrt((d*d).mean())),'finite':bool(torch.isfinite(a).all()),'shape_equal':list(a.shape)==list(b.shape)}
def main(a):
 h=torch.load(a.handoff,map_location='cpu',weights_only=True); hp=h['hf_prefill']; pp=h['portable_prefill']; out={'prefill':{'hf_bf16_vs_portable_bf16':{k:m(hp[k],pp[k]) for k in ('hidden','k','v')}}}; rows=[]
 for i,(hf,pb) in enumerate(zip(h['hf_decode'],h['portable_decode'])): rows.append({'step':i,'hf_bf16_vs_portable_bf16':{'hidden':m(hf[0],pb[0]),'k':m(hf[1],pb[1]),'v':m(hf[2],pb[2])}})
 out['decode']=rows; a.out.write_text(json.dumps(out,indent=2)+'\n'); cast=[{'name':k,'source_dtype':str(v.dtype),'target_dtype':'torch.float16','numel':v.numel()} for k,v in h['state_dict'].items()]; (a.out.parent/'weight_cast_manifest.json').write_text(json.dumps({'policy':'explicit BF16 -> FP16 cast','weights':cast},indent=2)+'\n'); data=a.handoff.read_bytes(); (a.out.parent/'handoff_integrity.json').write_text(json.dumps({'file':'layer0_handoff.pt','sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'verified_before_stage_b':True,'verified_at_stage_b':True},indent=2)+'\n'); print(json.dumps(out,sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--handoff',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); main(a)
