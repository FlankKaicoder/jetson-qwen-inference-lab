import argparse
from pathlib import Path
import torch

class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(1024, 3072).half().cuda()
    def forward(self, x):
        return self.linear(x)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); args=p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = Linear().eval(); x = torch.randn(1, 1024, device='cuda', dtype=torch.float16)
    torch.onnx.export(model, x, args.out, opset_version=17, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0:'M'}, 'output': {0:'M'}})
    print({'status':'EXPORT_PASS','path':str(args.out),'opset':17})
if __name__ == '__main__': main()
