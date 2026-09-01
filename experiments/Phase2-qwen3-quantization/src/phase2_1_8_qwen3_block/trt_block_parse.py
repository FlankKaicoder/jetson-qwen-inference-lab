import argparse
import json
from pathlib import Path
import onnx
import tensorrt as trt
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()
    result = {"onnx_checker": "FAIL", "trt_parse": "FAIL", "engine_build": "FAIL", "cuda_execute": "FAIL"}
    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)
    result.update({"onnx_checker": "PASS", "node_count": len(model.graph.node), "operators": [n.op_type for n in model.graph.node]})
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse(model.SerializeToString())
    result["parser_errors"] = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parsed:
        args.result.parent.mkdir(parents=True, exist_ok=True); args.result.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result)); return
    result["trt_parse"] = "PASS"
    config = builder.create_builder_config(); config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile(); profile.set_shape("hidden_states", (1, 8, 1024), (1, 8, 1024), (2, 8, 1024)); config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        result["error"] = "build_serialized_network returned None"
    else:
        args.engine.parent.mkdir(parents=True, exist_ok=True); args.engine.write_bytes(bytes(blob)); result.update({"engine_build": "PASS", "engine_bytes": args.engine.stat().st_size})
        runtime = trt.Runtime(logger); engine = runtime.deserialize_cuda_engine(bytes(blob)); context = engine.create_execution_context(); context.set_input_shape("hidden_states", (args.batch, 8, 1024))
        x = torch.randn((args.batch, 8, 1024), device="cuda", dtype=torch.float16); out_shape = tuple(context.get_tensor_shape("output")); out_dtype = engine.get_tensor_dtype("output")
        dtype_map = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.BF16: torch.bfloat16}
        if out_dtype not in dtype_map: raise RuntimeError(f"unsupported output dtype {out_dtype}")
        y = torch.empty(out_shape, device="cuda", dtype=dtype_map[out_dtype]); context.set_tensor_address("hidden_states", x.data_ptr()); context.set_tensor_address("output", y.data_ptr()); ok = bool(context.execute_async_v3(torch.cuda.current_stream().cuda_stream)); torch.cuda.synchronize()
        result.update({"cuda_execute": "PASS" if ok else "FAIL", "batch": args.batch, "input_shape": list(x.shape), "output_shape": list(out_shape), "output_dtype": str(out_dtype), "output_finite": bool(torch.isfinite(y).all().item())})
    args.result.parent.mkdir(parents=True, exist_ok=True); args.result.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
