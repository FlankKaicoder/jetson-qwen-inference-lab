from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL='/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca'
def snap(label):
    m={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1); m[k]=int(v.strip().split()[0])*1024
    free,total=torch.cuda.mem_get_info()
    return {'label':label,'MemAvailable':m['MemAvailable'],'SwapUsed':m['SwapTotal']-m['SwapFree'],'cuda_free':free,'cuda_total':total,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'max_allocated':torch.cuda.max_memory_allocated(),'max_reserved':torch.cuda.max_memory_reserved()}
torch.cuda.reset_peak_memory_stats(); rows=[snap('before_load')]
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='eager',local_files_only=True).eval(); rows.append(snap('after_load'))
tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True); inputs=tokenizer('memory diagnostic',return_tensors='pt').to('cuda:0')
with torch.inference_mode(): out=model(**inputs,use_cache=True)
rows.append(snap('after_forward'))
result={'status':'PASS','rows':rows,'finite_logits':bool(torch.isfinite(out.logits).all()),'logits_shape':list(out.logits.shape),'device':str(out.logits.device)}
print(json.dumps(result))
