import argparse
from pathlib import Path
import torch
from qwen3_layer_semantics import CONFIG, make_layer
class PrefillWrapper(torch.nn.Module):
    def __init__(self, layer): super().__init__(); self.layer=layer
    def forward(self, hidden_states, position_ids): return self.layer.forward_prefill(hidden_states, position_ids)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); args=p.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    m=PrefillWrapper(make_layer(device='cuda')); x=torch.randn((1,8,CONFIG['hidden_size']),device='cuda',dtype=torch.float16); pos=torch.arange(8,device='cuda',dtype=torch.long).unsqueeze(0)
    torch.onnx.export(m,(x,pos),args.out,opset_version=17,input_names=['hidden_states','position_ids'],output_names=['hidden_out','present_k','present_v'],dynamic_axes={'hidden_states':{0:'batch',1:'seq'},'position_ids':{0:'batch',1:'seq'},'hidden_out':{0:'batch',1:'seq'},'present_k':{0:'batch',2:'seq'},'present_v':{0:'batch',2:'seq'}})
    print({'status':'EXPORT_PASS','shape':[1,8,1024],'onnx':str(args.out)})
if __name__=='__main__': main()
