import argparse
from pathlib import Path
import torch
from qwen3_layer_semantics import CONFIG, make_layer
class DecodeWrapper(torch.nn.Module):
    def __init__(self, layer): super().__init__(); self.layer=layer
    def forward(self, hidden_states, position_ids, past_k, past_v): return self.layer.forward_decode(hidden_states, position_ids, past_k, past_v)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); args=p.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    m=DecodeWrapper(make_layer(device='cuda')); x=torch.randn((1,1,CONFIG['hidden_size']),device='cuda',dtype=torch.float16); pos=torch.tensor([[8]],device='cuda',dtype=torch.long); pk=torch.randn((1,8,8,128),device='cuda',dtype=torch.float16); pv=torch.randn_like(pk)
    torch.onnx.export(m,(x,pos,pk,pv),args.out,opset_version=17,input_names=['hidden_states','position_ids','past_k','past_v'],output_names=['hidden_out','present_k','present_v'],dynamic_axes={'hidden_states':{0:'batch'},'position_ids':{0:'batch'},'past_k':{0:'batch',2:'past_len'},'past_v':{0:'batch',2:'past_len'},'hidden_out':{0:'batch'},'present_k':{0:'batch',2:'present_len'},'present_v':{0:'batch',2:'present_len'}})
    print({'status':'EXPORT_PASS','shape':[1,1,1024],'past_len':8,'onnx':str(args.out)})
if __name__=='__main__': main()
