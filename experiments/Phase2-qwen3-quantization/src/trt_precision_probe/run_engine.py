"""Run one synthetic TensorRT Linear engine with existing PyTorch CUDA allocations."""
import argparse
import json

import numpy as np
import tensorrt as trt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--out-features", required=True, type=int)
    parser.add_argument("--m", required=True, type=int, choices=[1, 32])
    parser.add_argument("--qdq", action="store_true")
    args = parser.parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(open(args.engine, "rb").read())
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize engine")
    context = engine.create_execution_context()
    context.set_input_shape("input", (args.m, 1024))
    torch.manual_seed(0)
    x = torch.randn((args.m, 1024), device="cuda:0", dtype=torch.float16)
    output_dtype = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
    }.get(engine.get_tensor_dtype("output"))
    if output_dtype is None:
        raise RuntimeError(f"unsupported TensorRT output dtype: {engine.get_tensor_dtype('output')}")
    y = torch.empty((args.m, args.out_features), device="cuda:0", dtype=output_dtype)
    context.set_tensor_address("input", x.data_ptr())
    context.set_tensor_address("output", y.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v3(stream)
    torch.cuda.synchronize()
    result = {
        "execute_async_v3": bool(ok),
        "input_device": str(x.device),
        "output_device": str(y.device),
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "output_finite": bool(torch.isfinite(y).all().item()),
    }
    rng = np.random.default_rng(0)
    weight = torch.from_numpy(
        (rng.standard_normal((1024, args.out_features)).astype(np.float16) * np.float16(0.02))
    ).to(device="cuda:0")
    reference = x @ weight
    finite_pair = bool(torch.isfinite(y).all().item() and torch.isfinite(reference).all().item())
    if finite_pair:
        abs_err = (y - reference).abs()
        result.update({
            "max_abs_error": float(abs_err.max().item()),
            "mean_abs_error": float(abs_err.mean().item()),
            "relative_error": float((abs_err / reference.abs().clamp_min(1e-3)).max().item()),
        })
    else:
        result.update({"max_abs_error": None, "mean_abs_error": None, "relative_error": None})
    result["numerical_status"] = "PASS" if finite_pair else "FAIL_NONFINITE"
    print(json.dumps(result, sort_keys=True))
    if not ok or not result["output_finite"]:
        raise RuntimeError("TensorRT GPU execution did not complete with finite output")


if __name__ == "__main__":
    main()
