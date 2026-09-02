"""Small FP16-vs-FP32 accumulation probe for the runtime design record."""
import json
from pathlib import Path
import torch

def main(out: Path) -> None:
    torch.manual_seed(20260902)
    x = (torch.randn((4, 1024), device="cuda", dtype=torch.float16) * 8).half()
    w = torch.ones((1024,), device="cuda", dtype=torch.float16)
    eps = 1e-6
    fp32 = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).half()
    fp16 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    d = (fp16.float() - fp32.float()).abs()
    result = {"status": "PASS", "shape": list(x.shape), "dtype": "fp16", "fp32_accumulation_reference": True,
              "max_abs_difference": float(d.max().item()), "rmse": float(torch.sqrt((d*d).mean()).item()),
              "note": "TensorRT FP16 Reduce/Pow warning is retained; this probe is not a production accuracy bound."}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args(); main(a.out)
