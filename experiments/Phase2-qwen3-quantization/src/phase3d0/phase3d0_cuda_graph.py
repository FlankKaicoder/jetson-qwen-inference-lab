from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt


def load_phase3b_module():
    source = (Path(__file__).resolve().parents[1] / "phase3b"
              / "phase3b_runtime_context.py")
    spec = importlib.util.spec_from_file_location("phase3b_runtime_context", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = load_phase3b_module()


def tensor_dtype(engine, name: str) -> torch.dtype:
    dtype = engine.get_tensor_dtype(name)
    mapping = {
        trt.DataType.HALF: torch.float16,
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    if dtype not in mapping:
        raise RuntimeError(f"UNSUPPORTED_TENSOR_DTYPE:{name}:{dtype}")
    return mapping[dtype]


def execute_no_sync(rt, inputs: dict, outputs: dict | None = None) -> dict:
    ctx = rt.context
    for name, value in inputs.items():
        if not tuple(value.shape) == tuple(ctx.get_tensor_shape(name)):
            if not ctx.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"INPUT_SHAPE_REJECTED:{name}:{tuple(value.shape)}")
        ctx.set_tensor_address(name, value.data_ptr())
    if outputs is None:
        outputs = {}
        for i in range(rt.engine.num_io_tensors):
            name = rt.engine.get_tensor_name(i)
            if rt.engine.get_tensor_mode(name) == rt.trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name))
                outputs[name] = torch.empty(
                    shape, device="cuda", dtype=tensor_dtype(rt.engine, name))
    for name, value in outputs.items():
        ctx.set_tensor_address(name, value.data_ptr())
    if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError(f"GRAPH_EXECUTE_FAILED:{rt.engine_path}")
    return outputs


def run_stream_window(args, objects, token_ids: list[int],
                      initial_ks: list, initial_vs: list) -> dict:
    embed, dec, norm, lm = objects
    hidden_rows = []
    logits_rows = []
    cache_rows = []
    for step in range(args.decode_steps):
        token = token_ids[step % len(token_ids)]
        th = embed.run({"input_ids": torch.tensor(
            [[token]], device="cuda", dtype=torch.int64)})["hidden_states"]
        position = 8 + step
        inp = {"hidden_states": th,
               "position_ids": torch.tensor([[position]], device="cuda",
                                             dtype=torch.int64)}
        inp.update({f"past_k{i}": cache_rows[-1]["ks"][i] if cache_rows
                    else initial_ks[i] for i in range(28)})
        inp.update({f"past_v{i}": cache_rows[-1]["vs"][i] if cache_rows
                    else initial_vs[i] for i in range(28)})
        out = dec.run(inp)
        hidden = out["hidden_l27"]
        ks = [out[f"present_k{i}"] for i in range(28)]
        vs = [out[f"present_v{i}"] for i in range(28)]
        normed = norm.run({"hidden_states": hidden})["normalized_hidden_states"]
        logits = lm.run({"hidden_states": normed})["logits"]
        hidden_rows.append(hidden)
        logits_rows.append(logits)
        cache_rows.append({"position": position, "ks": ks, "vs": vs})
    return {"hidden": hidden_rows, "logits": logits_rows, "caches": cache_rows}


class GraphDecodeWindow:
    def __init__(self, objects, initial_ks: list, initial_vs: list,
                 token_ids: list[int], decode_steps: int, graph_mode: str):
        self.embed, self.dec, self.norm, self.lm = objects
        self.decode_steps = decode_steps
        self.token_ids = token_ids
        self.tokens = [torch.tensor([[token]], device="cuda", dtype=torch.int64)
                       for token in token_ids]
        self.positions = [torch.tensor([[8 + step]], device="cuda",
                                       dtype=torch.int64)
                          for step in range(decode_steps)]
        self.initial_ks = [k.detach().clone() for k in initial_ks]
        self.initial_vs = [v.detach().clone() for v in initial_vs]
        self.step_inputs = []
        self.step_outputs = []
        self.embedding_outputs = []
        self.embedding_hidden = []
        self.norm_outputs = []
        self.logits_outputs = []
        self.hidden_outputs = []
        self.graph = None
        self.step_graphs = []
        self.graph_mode = graph_mode

    def _prepare_step(self, step: int) -> None:
        length = 8 + step
        out_length = length + 1
        hidden_out = self.hidden_outputs[step]
        norm_out = self.norm_outputs[step]
        logits_out = self.logits_outputs[step]
        inputs = {"hidden_states": self.embedding_hidden[step],
                  "position_ids": self.positions[step]}
        outputs = {"hidden_l27": hidden_out}
        for layer in range(27):
            outputs[f"hidden_l{layer}"] = torch.empty(
                (1, 1, 1024), device="cuda", dtype=torch.float16)
        for i in range(28):
            inputs[f"past_k{i}"] = (self.initial_ks[i] if step == 0
                                    else self.step_outputs[step - 1][f"present_k{i}"])
            inputs[f"past_v{i}"] = (self.initial_vs[i] if step == 0
                                    else self.step_outputs[step - 1][f"present_v{i}"])
            outputs[f"present_k{i}"] = torch.empty(
                (1, 8, out_length, 128), device="cuda", dtype=torch.float16)
            outputs[f"present_v{i}"] = torch.empty(
                (1, 8, out_length, 128), device="cuda", dtype=torch.float16)
        self.step_inputs.append(inputs)
        self.step_outputs.append(outputs)

    def capture(self) -> dict:
        for step in range(self.decode_steps):
            self.embedding_outputs.append(execute_no_sync(self.embed, {
                "input_ids": self.tokens[step]}))
            self.embedding_hidden.append(
                self.embedding_outputs[step]["hidden_states"])
            shape = tuple(self.embedding_hidden[step].shape)
            self.hidden_outputs.append(torch.empty(
                shape, device="cuda", dtype=torch.float16))
            self.norm_outputs.append(torch.empty(
                shape, device="cuda", dtype=torch.float16))
            if not self.lm.context.set_input_shape("hidden_states", shape):
                raise RuntimeError(f"LM_HEAD_SHAPE_REJECTED:{shape}")
            logits_shape = self.lm.engine.get_tensor_shape("logits")
            self.logits_outputs.append(torch.empty(
                tuple(self.lm.context.get_tensor_shape("logits")),
                device="cuda", dtype=torch.float16))
            self._prepare_step(step)
        torch.cuda.synchronize()

        for step in range(self.decode_steps):
            execute_no_sync(self.embed,
                            {"input_ids": self.tokens[step]},
                            self.embedding_outputs[step])
            execute_no_sync(self.dec, self.step_inputs[step],
                            self.step_outputs[step])
            execute_no_sync(self.norm,
                            {"hidden_states": self.hidden_outputs[step]},
                            {"normalized_hidden_states": self.norm_outputs[step]})
            execute_no_sync(self.lm,
                            {"hidden_states": self.norm_outputs[step]},
                            {"logits": self.logits_outputs[step]})
        torch.cuda.synchronize()

        if self.graph_mode == "full_window":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for step in range(self.decode_steps):
                    execute_no_sync(self.embed,
                                    {"input_ids": self.tokens[step]},
                                    self.embedding_outputs[step])
                    execute_no_sync(self.dec, self.step_inputs[step],
                                    self.step_outputs[step])
                    execute_no_sync(self.norm,
                                    {"hidden_states": self.hidden_outputs[step]},
                                    {"normalized_hidden_states": self.norm_outputs[step]})
                    execute_no_sync(self.lm,
                                    {"hidden_states": self.norm_outputs[step]},
                                    {"logits": self.logits_outputs[step]})
            self.graph = graph
        else:
            self.graph = None
            self.step_graphs = []
            for step in range(self.decode_steps):
                calls = [
                    (self.embed, {"input_ids": self.tokens[step]},
                     self.embedding_outputs[step]),
                    (self.dec, self.step_inputs[step], self.step_outputs[step]),
                    (self.norm, {"hidden_states": self.hidden_outputs[step]},
                     {"normalized_hidden_states": self.norm_outputs[step]}),
                    (self.lm, {"hidden_states": self.norm_outputs[step]},
                     {"logits": self.logits_outputs[step]}),
                ]
                for engine, inputs, outputs in calls:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        execute_no_sync(engine, inputs, outputs)
                    self.step_graphs.append(graph)
        torch.cuda.synchronize()
        return self.topology_report()

    def replay(self) -> None:
        if self.graph is None and not getattr(self, "step_graphs", None):
            raise RuntimeError("GRAPH_NOT_CAPTURED")
        if self.graph is not None:
            self.graph.replay()
        else:
            for graph in self.step_graphs:
                graph.replay()

    def outputs(self) -> dict:
        return {"hidden": list(self.hidden_outputs),
                "logits": list(self.logits_outputs),
                "caches": [{"position": 8 + step,
                            "ks": [self.step_outputs[step][f"present_k{i}"]
                                   for i in range(28)],
                            "vs": [self.step_outputs[step][f"present_v{i}"]
                                   for i in range(28)]}
                           for step in range(self.decode_steps)]}

    def topology_report(self) -> dict:
        cache_chain = []
        for step in range(self.decode_steps):
            previous = (self.initial_ks if step == 0
                        else [self.step_outputs[step - 1][f"present_k{i}"]
                              for i in range(28)])
            inputs = [self.step_inputs[step][f"past_k{i}"] for i in range(28)]
            outputs = [self.step_outputs[step][f"present_k{i}"] for i in range(28)]
            cache_chain.append({
                "step": step,
                "input_length": 8 + step,
                "output_length": 9 + step,
                "input_is_previous_output": all(a is b or a.data_ptr() == b.data_ptr()
                                                for a, b in zip(previous, inputs)),
                "input_output_pointer_isolation": len(
                    {t.data_ptr() for t in inputs + outputs}) == 56,
            })
        return {
            "graph_api": ("torch.cuda.CUDAGraph full_window" if self.graph_mode == "full_window"
                          else "torch.cuda.CUDAGraph per_engine_enqueue"),
            "graph_mode": self.graph_mode,
            "captured_engines": ["embedding", "decode_28layer",
                                 "final_rmsnorm", "lm_head"],
            "captured_steps": self.decode_steps,
            "fixed_prefill_length": 8,
            "context_length": 8 + self.decode_steps,
            "capture_once": True,
            "per_step_capture": False,
            "static_tokens": True,
            "static_positions": True,
            "static_hidden": True,
            "static_logits": True,
            "static_kv_inputs": True,
            "static_kv_outputs": True,
            "cpu_sampling_inside_graph": False,
            "dynamic_allocation_inside_graph": False,
            "synchronization_inside_graph": False,
            "captured_graph_count": 1 if self.graph is not None else len(self.step_graphs),
            "cache_chain": cache_chain,
        }


def load_objects(args, mixed: bool) -> tuple[dict, list, object]:
    prefix = "mixed_" if mixed else ""
    engine_dir = args.mixed_dir if mixed else args.fp16_dir
    paths = [args.embedding_engine,
             engine_dir / f"{prefix}prefill_28layer.engine",
             engine_dir / f"{prefix}decode_28layer.engine",
             args.norm_engine, args.lm_engine]
    t0 = time.perf_counter()
    objects = [F.make_trt(path, True) for path in paths]
    embed_fn, _logits_fn = F.F.make_pipeline(objects[0], objects[3], objects[4])
    return {
        "engine_paths": [str(path) for path in paths],
        "initialization_s": time.perf_counter() - t0,
        "context_creations": sum(obj.context_creations for obj in objects),
    }, objects, embed_fn


def metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    delta = a - b
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta)
                             / torch.clamp(norm_a, min=tiny)),
        "cosine": float(torch.dot(a.reshape(-1), b.reshape(-1))
                         / torch.clamp(norm_a * norm_b, min=tiny)),
    }


def validate(args, stream_result: dict, graph: GraphDecodeWindow,
             graph_topology: dict) -> dict:
    graph.replay()
    torch.cuda.synchronize()
    graph_result = graph.outputs()
    comparisons = []
    for step in range(args.decode_steps):
        comparisons.append({
            "step": step,
            "hidden": metric(stream_result["hidden"][step],
                             graph_result["hidden"][step]),
            "logits": metric(stream_result["logits"][step],
                             graph_result["logits"][step]),
            "kv_exact": all(torch.equal(a, b)
                            for a, b in zip(stream_result["caches"][step]["ks"],
                                            graph_result["caches"][step]["ks"]))
            and all(torch.equal(a, b)
                    for a, b in zip(stream_result["caches"][step]["vs"],
                                    graph_result["caches"][step]["vs"])),
            "kv_shapes_pass": all(
                tuple(k.shape) == (1, 8, 9 + step, 128)
                for k in graph_result["caches"][step]["ks"]) and all(
                tuple(v.shape) == (1, 8, 9 + step, 128)
                for v in graph_result["caches"][step]["vs"]),
        })
    topology_pass = all(row["input_is_previous_output"]
                        and row["input_output_pointer_isolation"]
                        for row in graph_topology["cache_chain"])
    validation_pass = all(row["hidden"]["finite"] and row["logits"]["finite"]
                          and row["kv_exact"] and row["kv_shapes_pass"]
                          for row in comparisons) and topology_pass
    return {"phase": "Phase 3-D0-B", "gate": "PASS" if validation_pass else "BLOCKED",
            "pass": validation_pass, "topology_pass": topology_pass,
            "topology": graph_topology, "steps": comparisons,
            "address_stability_after_replay": graph.topology_report()
            == graph_topology}


def benchmark(args, objects, stream_result: dict, graph: GraphDecodeWindow,
              initial_ks: list, initial_vs: list, capture_s: float) -> dict:
    token_ids = args.force_cont[:args.decode_steps]
    memory = [F.memory_snapshot("benchmark_start")]
    for _ in range(args.warmup):
        run_stream_window(args, objects, token_ids, initial_ks, initial_vs)
    torch.cuda.synchronize()
    memory.append(F.memory_snapshot("stream_warmup_complete"))
    stream_raw = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_stream_window(args, objects, token_ids, initial_ks, initial_vs)
        torch.cuda.synchronize()
        stream_raw.append(time.perf_counter() - t0)
    memory.append(F.memory_snapshot("stream_benchmark_complete"))

    for _ in range(args.warmup):
        graph.replay()
        torch.cuda.synchronize()
    memory.append(F.memory_snapshot("graph_warmup_complete"))
    graph_raw = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        graph_raw.append(time.perf_counter() - t0)
    memory.append(F.memory_snapshot("graph_benchmark_complete"))

    def summary(raw: list[float]) -> dict:
        return F.F.timing_summary(raw)

    stream_summary = summary(stream_raw)
    graph_summary = summary(graph_raw)
    comparison = {
        "stream_window_ms": stream_summary,
        "graph_window_ms": graph_summary,
        "stream_tpot_ms": {key: value / args.decode_steps
                           for key, value in stream_summary.items()
                           if isinstance(value, (int, float))},
        "graph_tpot_ms": {key: value / args.decode_steps
                          for key, value in graph_summary.items()
                          if isinstance(value, (int, float))},
        "stream_tokens_per_sec": float(args.decode_steps / np.mean(stream_raw)),
        "graph_tokens_per_sec": float(args.decode_steps / np.mean(graph_raw)),
        "latency_reduction_pct": float(
            100.0 * (1.0 - np.mean(graph_raw) / np.mean(stream_raw))),
        "throughput_ratio": float(np.mean(stream_raw) / np.mean(graph_raw)),
        "capture_s": capture_s,
    }
    return {"phase": "Phase 3-D0-C", "comparison": comparison,
            "stream_raw_s": stream_raw, "graph_raw_s": graph_raw,
            "memory": memory}


def audit_from_objects(objects: list, runtime: str, boundary: str = "UNKNOWN") -> dict:
    engines = {
        "embedding": objects[0], "prefill_28layer": objects[1],
        "decode_28layer": objects[2], "final_rmsnorm": objects[3],
        "lm_head": objects[4],
    }
    tensors = {}
    for name, obj in engines.items():
        rows = []
        for i in range(obj.engine.num_io_tensors):
            tensor_name = obj.engine.get_tensor_name(i)
            rows.append({
                "name": tensor_name,
                "mode": str(obj.engine.get_tensor_mode(tensor_name)),
                "shape": list(obj.engine.get_tensor_shape(tensor_name)),
                "dtype": str(obj.engine.get_tensor_dtype(tensor_name)),
                "format": str(obj.engine.get_tensor_format(tensor_name)),
                "location": str(obj.engine.get_tensor_location(tensor_name)),
            })
        tensors[name] = rows
    matrix = [
        {"component": "Execution context", "capture_possible": "PARTIAL",
         "evidence": "One persistent context per engine is compatible, but current PersistentContext.run sets shapes/addresses and synchronizes per call."},
        {"component": "Tensor addresses", "capture_possible": "NO_IN_CURRENT_RUN",
         "evidence": "Fresh output tensors are allocated for every run; graph replay requires fixed output addresses."},
        {"component": "Dynamic decode shape", "capture_possible": "NO_PER_STEP_GRAPH",
         "evidence": "KV cache length advances each step; a bounded multi-step graph can encode fixed lengths 8..15."},
        {"component": "KV cache update", "capture_possible": "BOUNDED",
         "evidence": "Fixed-address step outputs can become next-step inputs inside one captured window; arbitrary continuation requires recapture or memory redesign."},
        {"component": "Embedding", "capture_possible": "BOUNDED",
         "evidence": "Static token-id and output addresses support capture for forced tokens."},
        {"component": "Decoder layer", "capture_possible": "BOUNDED",
         "evidence": "TensorRT execute_async_v3 on the capture stream can be captured after warmup."},
        {"component": "RMSNorm", "capture_possible": "BOUNDED",
         "evidence": "Static hidden/normed addresses support capture."},
        {"component": "LM Head", "capture_possible": "BOUNDED",
         "evidence": "Static normed/logits addresses support capture."},
        {"component": "CPU sampling", "capture_possible": "NO",
         "evidence": "CPU NumPy argmax is outside CUDA Graph; benchmark uses forced tokens."},
    ]
    return {"phase": "Phase 3-D0-A", "runtime": runtime,
            "engine_tensor_contract": tensors, "compatibility_matrix": matrix,
            "prototype_boundary": boundary,
            "graph_requirements": ["fixed topology", "fixed shapes",
                                   "fixed tensor addresses", "capture stream",
            "no per-step allocation or synchronize"]}


def audit_mode(args) -> dict:
    _info, objects, _embed_fn = load_objects(args, args.runtime == "mixed")
    return audit_from_objects(
        objects, args.runtime,
        "ONE_FIXED_S8_PREFILL_TO_8_STEP_DECODE_WINDOW")


def profile_mode(args) -> dict:
    info, objects, embed_fn = load_objects(args, args.runtime == "mixed")
    token_ids = args.force_cont[:args.decode_steps]
    with F.nvtx_range("PHASE3D0_INIT", True):
        pass
    hidden, ks, vs = F.F.run_prefill(objects[1], embed_fn, args.bench_ids[:8])
    graph = GraphDecodeWindow(
        [objects[0], objects[2], objects[3], objects[4]], ks, vs,
        token_ids, args.decode_steps, args.graph_mode)
    with F.nvtx_range("PHASE3D0_WARMUP", True):
        for _ in range(args.warmup):
            run_stream_window(
                args, [objects[0], objects[2], objects[3], objects[4]],
                token_ids, ks, vs)
        torch.cuda.synchronize()
    graph.capture()
    if args.graph:
        with F.nvtx_range("PHASE3D0_STEADY_CUDA_GRAPH_WINDOW", True):
            graph.replay()
        torch.cuda.synchronize()
    else:
        with F.nvtx_range("PHASE3D0_STEADY_STREAM_WINDOW", True):
            run_stream_window(
                args, [objects[0], objects[2], objects[3], objects[4]],
                token_ids, ks, vs)
        torch.cuda.synchronize()
    return {"phase": "Phase 3-D0-D", "path": "cuda_graph" if args.graph
            else "persistent_stream", "runtime": args.runtime,
            "decode_steps": args.decode_steps, "initialization_s": info["initialization_s"]}


def bench_mode(args) -> dict:
    token_ids = args.force_cont[:args.decode_steps]
    info, objects, embed_fn = load_objects(args, args.runtime == "mixed")
    hidden, ks, vs = F.F.run_prefill(objects[1], embed_fn, args.bench_ids[:8])
    graph_objects = [objects[0], objects[2], objects[3], objects[4]]
    stream_result = run_stream_window(args, graph_objects, token_ids, ks, vs)
    graph = GraphDecodeWindow(
        graph_objects, ks, vs, token_ids, args.decode_steps, args.graph_mode)
    capture_start = time.perf_counter()
    graph_topology = graph.capture()
    capture_s = time.perf_counter() - capture_start
    validation = validate(args, stream_result, graph, graph_topology)
    benchmark_result = benchmark(
        args, graph_objects, stream_result, graph, ks, vs, capture_s)
    audit = audit_from_objects(
        objects, args.runtime,
        "ONE_FIXED_S8_PREFILL_TO_8_STEP_DECODE_WINDOW")
    return {"phase": "Phase 3-D0-B/C", "runtime": args.runtime,
            "runtime_info": info, "validation": validation,
            "benchmark": benchmark_result, "audit": audit,
            "minimum_memory": min(row.get("mem_available_bytes", 0)
                                  for row in benchmark_result["memory"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "bench", "profile"],
                        default="audit")
    parser.add_argument("--runtime", choices=["fp16", "mixed"], default="fp16")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--graph-mode", choices=["full_window", "per_engine"],
                        default="full_window")
    parser.add_argument("--fp16-dir", type=Path,
                        default=Path("/tmp/phase2_2b4_2_20260902T082326Z"))
    parser.add_argument("--mixed-dir", type=Path,
                        default=Path("/tmp/phase2_3e_20260904T020000Z/work"))
    parser.add_argument("--embedding-engine", type=Path,
                        default=Path("/tmp/phase2_2c1_20260902T090000Z/embedding_fp16.engine"))
    parser.add_argument("--norm-engine", type=Path,
                        default=Path("/tmp/phase2_2c2_20260903T/final_rmsnorm_fp32_reduce.engine"))
    parser.add_argument("--lm-engine", type=Path,
                        default=Path("/tmp/phase2_2c3_20260903T024500Z/lm_head_fp16.engine"))
    parser.add_argument("--eval-manifest", type=Path,
                        default=Path("experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T205610Z/evaluation_manifest.json"))
    parser.add_argument("--force-cont", type=Path,
                        default=Path("experiments/Phase2-qwen3-quantization/src/phase2_3f/force_cont.json"))
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    manifest = json.loads(args.eval_manifest.read_text())
    evaluation = [row for row in manifest["rows"] if row.get("split") == "evaluation"]
    if len(evaluation) != 12:
        raise RuntimeError(f"EVAL_PROMPT_COUNT:{len(evaluation)}")
    args.bench_ids = evaluation[0]["token_ids"]
    args.force_cont = json.loads(args.force_cont.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    F.F.dump(args.out / "environment.json", F.environment_snapshot())
    F.F.dump(args.out / "run_config.json", {
        "mode": args.mode, "runtime": args.runtime, "graph": args.graph,
        "graph_mode": args.graph_mode,
        "decode_steps": args.decode_steps, "warmup": args.warmup,
        "repeats": args.repeats, "batch": 1, "prefill_length": 8,
        "sampling": "forced deterministic continuation tokens outside graph",
        "timing": "time.perf_counter with torch.cuda.synchronize before/after",
        "capture_policy": "capture once after stream warmup; no per-step capture",
    })
    if args.mode == "audit":
        result = audit_mode(args)
    elif args.mode == "bench":
        result = bench_mode(args)
    else:
        result = profile_mode(args)
    F.F.dump(args.out / f"{args.mode}_result.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
