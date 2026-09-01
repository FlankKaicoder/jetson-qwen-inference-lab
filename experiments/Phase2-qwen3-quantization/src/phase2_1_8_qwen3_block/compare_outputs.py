import argparse
import json
from pathlib import Path
import tensorrt as trt
import torch
from qwen3_block import CONFIG, make_block


def main():
    p = argparse.ArgumentParser(); p.add_argument("--engine", type=Path, required=True); p.add_argument("--result", type=Path, required=True); p.add_argument("--batch", type=int, default=1); args = p.parse_args()
    torch.manual_seed(123); block = make_block(); x = torch.randn((args.batch, CONFIG["sequence_length"], CONFIG["hidden_size"]), device="cuda", dtype=torch.float16)
    with torch.no_grad(): ref = block(x)
    logger = trt.Logger(trt.Logger.WARNING); runtime = trt.Runtime(logger); engine = runtime.deserialize_cuda_engine(args.engine.read_bytes()); context = engine.create_execution_context(); context.set_input_shape("hidden_states", tuple(x.shape)); out_shape = tuple(context.get_tensor_shape("output")); y = torch.empty(out_shape, device="cuda", dtype=torch.float16)
    context.set_tensor_address("hidden_states", x.data_ptr()); context.set_tensor_address("output", y.data_ptr()); executed = bool(context.execute_async_v3(torch.cuda.current_stream().cuda_stream)); torch.cuda.synchronize()
    delta = (y.float() - ref.float()).abs(); denom = ref.float().abs().clamp_min(1e-6); result = {"executed": executed, "batch": args.batch, "shape_equal": list(y.shape) == list(ref.shape), "finite": bool(torch.isfinite(y).all().item()), "max_abs_error": float(delta.max().item()), "max_rel_error": float((delta / denom).max().item()), "rmse": float(torch.sqrt(torch.mean(delta * delta)).item()), "tolerance": "informational only; FP16 graph comparison, no Gate accuracy threshold"}
    args.result.parent.mkdir(parents=True, exist_ok=True); args.result.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
