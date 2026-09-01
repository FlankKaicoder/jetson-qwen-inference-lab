import argparse
from pathlib import Path
import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__(); torch.manual_seed(0)
        self.fc1 = torch.nn.Linear(1024, 3072).half().cuda(); self.fc2 = torch.nn.Linear(3072, 1024).half().cuda()
    def forward(self, x): return self.fc2(torch.nn.functional.gelu(self.fc1(x), approximate='tanh'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(MLP().eval(), torch.randn(1,1024,device='cuda',dtype=torch.float16), a.out, opset_version=17, input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'M'},'output':{0:'M'}})
    print({'status':'EXPORT_PASS','path':str(a.out),'opset':17})
if __name__=='__main__': main()
