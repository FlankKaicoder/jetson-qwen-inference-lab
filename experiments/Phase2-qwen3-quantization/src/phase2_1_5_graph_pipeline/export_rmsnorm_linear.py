import argparse
from pathlib import Path
import torch

class RMSNormLinear(torch.nn.Module):
    def __init__(self):
        super().__init__(); torch.manual_seed(0); self.weight=torch.nn.Parameter(torch.ones(1024,device='cuda',dtype=torch.float16)); self.linear=torch.nn.Linear(1024,3072).half().cuda()
    def forward(self,x):
        y=x * torch.rsqrt(torch.mean(x*x,dim=-1,keepdim=True) + 1e-6) * self.weight
        return self.linear(y)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(RMSNormLinear().eval(), torch.randn(1,1024,device='cuda',dtype=torch.float16), a.out, opset_version=17, input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'M'},'output':{0:'M'}})
    print({'status':'EXPORT_PASS','path':str(a.out),'opset':17})
if __name__=='__main__': main()
