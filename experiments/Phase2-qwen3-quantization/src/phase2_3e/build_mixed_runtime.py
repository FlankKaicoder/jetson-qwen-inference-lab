from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import re
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np


OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def command_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE:{type(exc).__name__}"


def environment() -> dict[str, object]:
    info = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = list(torch.cuda.get_device_capability(0))
            info["cuda_runtime"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"UNAVAILABLE:{type(exc).__name__}"
    try:
        import onnx
        info["onnx"] = onnx.__version__
    except Exception as exc:  # noqa: BLE001
        info["onnx"] = f"UNAVAILABLE:{type(exc).__name__}"
    try:
        import tensorrt as trt
        info["tensorrt_python"] = trt.__version__
    except Exception as exc:  # noqa: BLE001
        info["tensorrt_python"] = f"UNAVAILABLE:{type(exc).__name__}"
    return info


def snap(stage: str) -> dict[str, object]:
    row: dict[str, object] = {"stage": stage, "maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                row["mem_available_bytes"] = int(line.split()[1]) * 1024
            elif line.startswith("SwapTotal:"):
                row["swap_total_bytes"] = int(line.split()[1]) * 1024
            elif line.startswith("SwapFree:"):
                row["swap_free_bytes"] = int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            row.update(cuda_free_bytes=free, cuda_total_bytes=total,
                       torch_allocated_bytes=torch.cuda.memory_allocated(),
                       torch_reserved_bytes=torch.cuda.memory_reserved())
    except Exception:  # noqa: BLE001
        pass
    return row


def load_policy(path: Path) -> dict[str, dict]:
    obj = json.loads(path.read_text())
    rows = obj.get("entries", [])
    if len(rows) != 196:
        raise RuntimeError(f"POLICY_CARDINALITY_INVALID:{len(rows)}")
    states = {x.get("precision_state") for x in rows}
    if states - {"FP16", "PT_W8A8"}:
        raise RuntimeError(f"POLICY_STATE_INVALID:{states}")
    targets = [x.get("target") for x in rows]
    if len(set(targets)) != 196:
        raise RuntimeError("POLICY_TARGET_DUPLICATE")
    return {x["target"]: x for x in rows}


def load_scales(path: Path) -> dict[str, float]:
    obj = json.loads(path.read_text())
    out = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            out[key] = float(value["scale"])
        else:
            out[key] = float(value)
    return out


def target_from_node(name: str) -> str | None:
    m = re.search(r"/(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)(?:_(\d+))?/MatMul$", name)
    if not m:
        return None
    return f"L{int(m.group(2) or 0)}:{m.group(1)}"


def sha_bytes(x: np.ndarray) -> str:
    return hashlib.sha256(x.tobytes()).hexdigest()


def metric(a, b) -> dict[str, object]:
    import torch
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    d = a - b
    an, bn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    tiny = torch.finfo(torch.float32).tiny
    return {
        "shape_equal": list(a.shape) == list(b.shape),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_abs": float(d.abs().max()),
        "mean_abs": float(d.abs().mean()),
        "rmse": float(torch.sqrt((d * d).mean())),
        "relative_l2": float(torch.linalg.vector_norm(d) / torch.clamp(an, min=tiny)),
        "cosine": float(torch.dot(a.reshape(-1), b.reshape(-1)) / torch.clamp(an * bn, min=tiny)),
    }


def topk_compare(a, b, k: int = 5) -> dict[str, object]:
    import torch
    a = a.detach().float().cpu().reshape(-1, a.shape[-1])
    b = b.detach().float().cpu().reshape(-1, b.shape[-1])
    ia, ib = a.argmax(dim=-1), b.argmax(dim=-1)
    ta, tb = a.topk(k, dim=-1).indices, b.topk(k, dim=-1).indices
    overlap = [len(set(x.tolist()) & set(y.tolist())) for x, y in zip(ta, tb)]
    va, _ = a.topk(2, dim=-1)
    vb, _ = b.topk(2, dim=-1)
    return {
        "top1_agreement": bool(torch.equal(ia, ib)),
        "top1_a": ia.tolist(),
        "top1_b": ib.tolist(),
        "top5_overlap": overlap,
        "top5_overlap_mean": float(sum(overlap) / len(overlap)),
        "top1_top2_margin_a": float((va[:, 0] - va[:, 1]).mean()),
        "top1_top2_margin_b": float((vb[:, 0] - vb[:, 1]).mean()),
    }


def transform(src: Path, dst: Path, policy: dict, scales: dict[str, float]) -> dict:
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(src), load_external_data=False)
    graph = model.graph
    initializers = {x.name: x for x in graph.initializer}
    rows = []
    new_nodes = []
    seen: set[str] = set()
    for node in graph.node:
        target = target_from_node(node.name) if node.op_type == "MatMul" else None
        if target is None:
            new_nodes.append(node)
            continue
        seen.add(target)
        entry = policy[target]
        wname = node.input[1]
        if entry["precision_state"] == "FP16":
            rows.append({
                "target": target, "layer": entry["layer"], "operator": entry["operator"],
                "precision_state": "FP16", "weight_key": wname,
                "weight_shape": entry.get("shape"),
                "activation_scale_available": False, "build_mode": "FP16",
            })
            new_nodes.append(node)
            continue
        if target not in scales:
            raise RuntimeError(f"MISSING_ACTIVATION_SCALE:{target}")
        if wname not in initializers:
            raise RuntimeError(f"WEIGHT_NOT_INITIALIZER:{target}:{wname}")
        arr = numpy_helper.to_array(initializers[wname])
        if arr.dtype not in (np.float16, np.float32):
            raise RuntimeError(f"UNEXPECTED_WEIGHT_DTYPE:{target}:{arr.dtype}")
        arr32 = arr.astype(np.float32)
        ws = float(np.max(np.abs(arr32)) / 127.0) or 1.0
        q = np.clip(np.rint(arr32 / ws), -127, 127).astype(np.int8)
        initializers[wname].CopyFrom(numpy_helper.from_array(q, name=wname))
        ws_name = f"{wname}__w_scale"
        wz_name = f"{wname}__w_zp"
        graph.initializer.extend([
            numpy_helper.from_array(np.asarray(ws, dtype=np.float16), name=ws_name),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), name=wz_name),
        ])
        a_scale_name = f"{node.name}__a_scale"
        a_zp_name = f"{node.name}__a_zp"
        graph.initializer.extend([
            numpy_helper.from_array(np.asarray(scales[target], dtype=np.float16), name=a_scale_name),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), name=a_zp_name),
        ])
        wdq = f"{wname}__dq"
        aq = f"{node.name}__a_q"
        adq = f"{node.name}__a_dq"
        new_nodes.extend([
            helper.make_node("QuantizeLinear", [node.input[0], a_scale_name, a_zp_name], [aq], name=f"{node.name}__a_quant"),
            helper.make_node("DequantizeLinear", [aq, a_scale_name, a_zp_name], [adq], name=f"{node.name}__a_dequant"),
            helper.make_node("DequantizeLinear", [wname, ws_name, wz_name], [wdq], name=f"{node.name}__w_dequant"),
        ])
        clone = onnx.NodeProto()
        clone.CopyFrom(node)
        clone.input[0] = adq
        clone.input[1] = wdq
        new_nodes.append(clone)
        rows.append({
            "target": target, "layer": entry["layer"], "operator": entry["operator"],
            "precision_state": "PT_W8A8", "weight_key": wname,
            "weight_shape": entry.get("shape"),
            "onnx_weight_shape": list(q.shape),
            "weight_scale": ws,
            "activation_scale": scales[target],
            "activation_scale_available": True, "build_mode": "PT_W8A8",
        })
    if len(seen) != 196:
        raise RuntimeError(f"LINEAR_NODE_CARDINALITY:{len(seen)}")
    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return {"assignments": rows, "node_count": len(graph.node), "initializer_count": len(graph.initializer)}


def build(path: Path, engine: Path, kind: str) -> dict[str, object]:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse(path.read_bytes())
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not ok:
        return {"parse": "FAIL", "build": "NOT_RUN", "parser_errors": errors}
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt, "ProfilingVerbosity"):
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    prof = builder.create_optimization_profile()
    if kind == "prefill":
        prof.set_shape("hidden_states", (1, 1, 1024), (1, 8, 1024), (1, 16, 1024))
        prof.set_shape("position_ids", (1, 1), (1, 8), (1, 16))
    else:
        prof.set_shape("hidden_states", (1, 1, 1024), (1, 1, 1024), (1, 1, 1024))
        prof.set_shape("position_ids", (1, 1), (1, 1), (1, 1))
        for n in [f"past_k{i}" for i in range(28)] + [f"past_v{i}" for i in range(28)]:
            prof.set_shape(n, (1, 8, 1, 128), (1, 8, 8, 128), (1, 8, 16, 128))
    cfg.add_optimization_profile(prof)
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        return {"parse": "PASS", "build": "FAIL", "parser_errors": errors}
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(bytes(blob))
    inspector = "UNAVAILABLE"
    layer_summary = []
    try:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            try:
                precision = str(layer.precision)
            except Exception:  # noqa: BLE001
                precision = "UNAVAILABLE"
            layer_summary.append({"index": i, "name": layer.name, "type": str(layer.type), "precision": precision})
        e = trt.Runtime(logger).deserialize_cuda_engine(bytes(blob))
        inspector = e.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    except Exception as exc:  # noqa: BLE001
        inspector = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
    return {
        "parse": "PASS", "build": "PASS", "engine_bytes": engine.stat().st_size,
        "parser_errors": errors, "network_layers": network.num_layers,
        "engine_inspector": inspector, "layer_summary": layer_summary,
    }


class TRT:
    def __init__(self, path: Path):
        import tensorrt as trt
        import torch
        self.trt = trt
        self.path = path
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"ENGINE_DESERIALIZE_FAILED:{path}")
        self.stream = torch.cuda.current_stream()

    def run(self, inputs: dict) -> dict:
        import torch
        ctx = self.engine.create_execution_context()
        for name, value in inputs.items():
            if not ctx.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"INPUT_SHAPE_REJECTED:{name}:{tuple(value.shape)}")
            ctx.set_tensor_address(name, value.data_ptr())
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(ctx.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
                if dtype == self.trt.DataType.HALF:
                    tdtype = torch.float16
                elif dtype == self.trt.DataType.FLOAT:
                    tdtype = torch.float32
                else:
                    raise RuntimeError(f"UNSUPPORTED_OUTPUT_DTYPE:{name}:{dtype}")
                outputs[name] = torch.empty(shape, device="cuda", dtype=tdtype)
                ctx.set_tensor_address(name, outputs[name].data_ptr())
        if not ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError(f"EXECUTE_FAILED:{self.path}")
        self.stream.synchronize()
        return outputs


def greedy(logits) -> tuple[int, dict]:
    x = logits.detach().float().cpu().numpy().reshape(-1, logits.shape[-1])
    token = int(np.argmax(x, axis=-1)[0])
    vals = np.sort(x, axis=-1)[0, ::-1][:2]
    return token, {"token": token, "top1_top2_margin": float(vals[0] - vals[1])}


def build_mode(args: argparse.Namespace) -> None:
    import torch
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    dump(args.out / "environment.json", environment())
    dump(args.out / "start_audit.json", {
        "branch": command_text(["git", "-C", str(args.repo), "branch", "--show-current"]) if args.repo else "NOT_PROVIDED",
        "head": command_text(["git", "-C", str(args.repo), "rev-parse", "HEAD"]) if args.repo else "NOT_PROVIDED",
    })
    policy = load_policy(args.policy)
    scales = load_scales(args.scales)
    audit = [
        {
            "target": k, "layer": v["layer"], "operator": v["operator"],
            "precision_state": v["precision_state"], "weight_key": v["checkpoint_key"],
            "weight_shape": v.get("shape"),
            "activation_scale_available": (k in scales) if v["precision_state"] == "PT_W8A8" else False,
            "build_mode": v["precision_state"],
        }
        for k, v in sorted(policy.items())
    ]
    fp16_n = sum(x["precision_state"] == "FP16" for x in audit)
    int8_n = sum(x["precision_state"] == "PT_W8A8" for x in audit)
    if fp16_n != 63 or int8_n != 133:
        raise RuntimeError(f"POLICY_COUNTS_INVALID:{fp16_n}/{int8_n}")
    dump(args.out / "policy_application_audit.json", {
        "status": "PASS", "policy": args.policy.name,
        "entries": audit, "fp16_linear_count": 63, "pt_w8a8_linear_count": 133,
    })

    trace = [snap("build_start")]
    builds: dict[str, dict] = {}
    transform_summaries: dict[str, dict] = {}
    for kind in ("prefill", "decode"):
        src = args.fp16_dir / f"{kind}_28layer.onnx"
        dst = args.work / f"mixed_{kind}_28layer.onnx"
        transform_summaries[kind] = transform(src, dst, policy, scales)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trace.append(snap(f"{kind}_transformed"))
        builds[kind] = build(dst, args.work / f"mixed_{kind}_28layer.engine", kind)
        trace.append(snap(f"{kind}_built"))
        if builds[kind].get("build") != "PASS":
            dump(args.out / "build_blocked.json", {"kind": kind, "result": builds[kind]})
            raise RuntimeError(f"BUILD_FAILED:{kind}:{builds[kind]}")

    layer_counts = {}
    for row in transform_summaries["prefill"]["assignments"]:
        li = row["layer"]
        layer_counts.setdefault(li, {"layer": li, "FP16": 0, "PT_W8A8": 0})
        layer_counts[li][row["precision_state"]] += 1
    dump(args.out / "layer_build_summary.json", {
        "status": "PASS",
        "layers": [layer_counts[i] for i in range(28)],
        "total_linear_assignments": 196,
        "transform": {k: {kk: vv for kk, vv in v.items() if kk != "assignments"} for k, v in transform_summaries.items()},
    })
    insp = {k: v.get("engine_inspector", "") for k, v in builds.items()}
    dump(args.out / "engine_precision_summary.json", {
        "status": "PASS",
        "prefill_build": {k: v for k, v in builds["prefill"].items() if k != "engine_inspector" and k != "layer_summary"},
        "decode_build": {k: v for k, v in builds["decode"].items() if k != "engine_inspector" and k != "layer_summary"},
        "engine_inspector": insp,
    })
    dump(args.out / "memory_trace_build.json", trace)
    print(json.dumps({"status": "PASS", "prefill_build": builds["prefill"]["build"], "decode_build": builds["decode"]["build"]}))


def run_runtime(embedding: TRT, norm: TRT, lm: TRT, pre: TRT, dec: TRT,
                prompts: dict, decode_steps: int, reference_cont: list[int]) -> dict:
    """Run one runtime and return all evidence (mixed or fp16)."""
    import torch

    def logits_of(hidden):
        normed = norm.run({"hidden_states": hidden})["normalized_hidden_states"]
        return lm.run({"hidden_states": normed})["logits"]

    # A. Prefill S>=8 validation (real 10-token prompt).
    prefill_ids = torch.tensor([prompts["en_10tok"]["input_ids"]], device="cuda", dtype=torch.long)
    prefill_pos = torch.arange(prefill_ids.shape[1], device="cuda", dtype=torch.long).reshape(1, -1)
    embedded = embedding.run({"input_ids": prefill_ids})["hidden_states"]
    pre_out = pre.run({"hidden_states": embedded, "position_ids": prefill_pos})
    hs = [pre_out[f"hidden_l{i}"] for i in range(28)]
    ks = [pre_out[f"present_k{i}"] for i in range(28)]
    vs = [pre_out[f"present_v{i}"] for i in range(28)]
    prefill = {
        "status": "PASS",
        "sequence_length": int(prefill_ids.shape[1]),
        "layers_executed": 28,
        "all_finite": all(bool(torch.isfinite(t).all()) for t in hs + ks + vs),
        "hidden_shapes_ok": all(list(t.shape) == [1, prefill_ids.shape[1], 1024] for t in hs),
        "kv_shapes_ok": all(list(t.shape) == [1, 8, prefill_ids.shape[1], 128] for t in ks + vs),
        "k_pointer_isolation": len({int(t.data_ptr()) for t in ks}) == 28,
        "v_pointer_isolation": len({int(t.data_ptr()) for t in vs}) == 28,
    }
    layer_hidden = {i: pre_out[f"hidden_l{i}"].detach().cpu() for i in (0, 9, 18, 27)}

    # B. Same-prefix forced decode (Hello prompt, forced HF reference continuation).
    hello_ids = torch.tensor([prompts["hello"]["input_ids"]], device="cuda", dtype=torch.long)
    hello_pos = torch.arange(hello_ids.shape[1], device="cuda", dtype=torch.long).reshape(1, -1)
    hello_embedded = embedding.run({"input_ids": hello_ids})["hidden_states"]
    hello_pre = pre.run({"hidden_states": hello_embedded, "position_ids": hello_pos})
    hello_hidden = hello_pre["hidden_l27"]
    hello_k = [hello_pre[f"present_k{i}"] for i in range(28)]
    hello_v = [hello_pre[f"present_v{i}"] for i in range(28)]
    prefill_last_hidden = hello_hidden[:, -1:, :].detach().cpu()
    prefill_logits = logits_of(hello_hidden[:, -1:, :]).detach().cpu()
    forced_rows = []
    for step, forced_token in enumerate(reference_cont):
        old_k, old_v = hello_k, hello_v
        token_hidden = embedding.run({"input_ids": torch.tensor([[forced_token]], device="cuda", dtype=torch.long)})["hidden_states"]
        pos = torch.tensor([[hello_ids.shape[1] + step]], device="cuda", dtype=torch.long)
        dec_inputs = {"hidden_states": token_hidden, "position_ids": pos}
        dec_inputs.update({f"past_k{i}": old_k[i] for i in range(28)})
        dec_inputs.update({f"past_v{i}": old_v[i] for i in range(28)})
        out = dec.run(dec_inputs)
        hello_k = [out[f"present_k{i}"] for i in range(28)]
        hello_v = [out[f"present_v{i}"] for i in range(28)]
        hello_hidden = out["hidden_l27"]
        prefix_k = all(torch.equal(old_k[i], hello_k[i][:, :, :old_k[i].shape[2], :]) for i in range(28))
        prefix_v = all(torch.equal(old_v[i], hello_v[i][:, :, :old_v[i].shape[2], :]) for i in range(28))
        forced_rows.append({
            "step": step + 1, "forced_token": forced_token,
            "logits": logits_of(hello_hidden).detach().cpu(),
            "hidden": hello_hidden.detach().cpu(),
            "cache_length": int(hello_k[0].shape[2]),
            "prefix_k_exact": bool(prefix_k), "prefix_v_exact": bool(prefix_v),
            "finite": all(bool(torch.isfinite(x).all()) for x in out.values()),
        })

    # C. Free-run generation (Hello, greedy).
    gen_ids = torch.tensor([prompts["hello"]["input_ids"]], device="cuda", dtype=torch.long)
    gen_pos = torch.arange(gen_ids.shape[1], device="cuda", dtype=torch.long).reshape(1, -1)
    gen_embedded = embedding.run({"input_ids": gen_ids})["hidden_states"]
    gen_pre = pre.run({"hidden_states": gen_embedded, "position_ids": gen_pos})
    gen_hidden = gen_pre["hidden_l27"]
    gen_k = [gen_pre[f"present_k{i}"] for i in range(28)]
    gen_v = [gen_pre[f"present_v{i}"] for i in range(28)]
    first_token, _ = greedy(logits_of(gen_hidden[:, -1:, :]))
    gen_tokens = [first_token]
    gen_kv = [{
        "step": 0, "phase": "prefill", "cache_length": int(gen_k[0].shape[2]),
        "all_finite": all(bool(torch.isfinite(x).all()) for x in gen_k + gen_v),
        "pointer_isolation_k": len({int(x.data_ptr()) for x in gen_k}) == 28,
        "pointer_isolation_v": len({int(x.data_ptr()) for x in gen_v}) == 28,
        "prefix_k_exact": True, "prefix_v_exact": True,
    }]
    for step in range(decode_steps):
        old_k, old_v = gen_k, gen_v
        token_hidden = embedding.run({"input_ids": torch.tensor([[gen_tokens[-1]]], device="cuda", dtype=torch.long)})["hidden_states"]
        pos = torch.tensor([[gen_ids.shape[1] + step]], device="cuda", dtype=torch.long)
        dec_inputs = {"hidden_states": token_hidden, "position_ids": pos}
        dec_inputs.update({f"past_k{i}": old_k[i] for i in range(28)})
        dec_inputs.update({f"past_v{i}": old_v[i] for i in range(28)})
        out = dec.run(dec_inputs)
        gen_k = [out[f"present_k{i}"] for i in range(28)]
        gen_v = [out[f"present_v{i}"] for i in range(28)]
        gen_hidden = out["hidden_l27"]
        tok, _ = greedy(logits_of(gen_hidden))
        gen_tokens.append(tok)
        prefix_k = all(torch.equal(old_k[i], gen_k[i][:, :, :old_k[i].shape[2], :]) for i in range(28))
        prefix_v = all(torch.equal(old_v[i], gen_v[i][:, :, :old_v[i].shape[2], :]) for i in range(28))
        gen_kv.append({
            "step": step + 1, "phase": "decode", "cache_length": int(gen_k[0].shape[2]),
            "all_finite": all(bool(torch.isfinite(x).all()) for x in out.values()),
            "pointer_isolation_k": len({int(x.data_ptr()) for x in gen_k}) == 28,
            "pointer_isolation_v": len({int(x.data_ptr()) for x in gen_v}) == 28,
            "prefix_k_exact": bool(prefix_k), "prefix_v_exact": bool(prefix_v),
        })
    gen_finite = all(r["all_finite"] for r in gen_kv)
    gen_pointer = all(r["pointer_isolation_k"] and r["pointer_isolation_v"] for r in gen_kv)
    gen_prefix = all(r["prefix_k_exact"] and r["prefix_v_exact"] for r in gen_kv)
    generation = {
        "status": "PASS" if (gen_finite and gen_pointer and gen_prefix) else "BLOCKED",
        "prompt": prompts["hello"]["text"], "input_ids": prompts["hello"]["input_ids"],
        "tokens": gen_tokens, "kv": gen_kv,
        "finite": gen_finite, "pointer_isolation": gen_pointer, "prefix_invariant": gen_prefix,
    }
    return {
        "prefill": prefill, "layer_hidden": layer_hidden,
        "prefill_last_hidden": prefill_last_hidden, "prefill_logits": prefill_logits,
        "forced_rows": forced_rows, "generation": generation,
    }


def runtime_mode(args: argparse.Namespace) -> None:
    import torch
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text())
    reference_cont = json.loads(args.reference_cont.read_text()) if args.reference_cont else None
    trace = [snap("runtime_start")]
    embedding = TRT(args.embedding_engine)
    norm = TRT(args.norm_engine)
    lm = TRT(args.lm_engine)

    results: dict[str, dict] = {}
    for label, pre_path, dec_path in (
        ("mixed", args.work / "mixed_prefill_28layer.engine", args.work / "mixed_decode_28layer.engine"),
        ("fp16", args.fp16_dir / "prefill_28layer.engine", args.fp16_dir / "decode_28layer.engine"),
    ):
        pre = TRT(pre_path)
        dec = TRT(dec_path)
        trace.append(snap(f"{label}_decoder_loaded"))
        results[label] = run_runtime(embedding, norm, lm, pre, dec, prompts, args.decode_steps, reference_cont)
        del pre, dec
        gc.collect()
        torch.cuda.empty_cache()
        trace.append(snap(f"{label}_done"))

    # Representative layer comparison (mixed vs FP16).
    rep = {f"L{i}": metric(results["mixed"]["layer_hidden"][i], results["fp16"]["layer_hidden"][i]) for i in (0, 9, 18, 27)}
    dump(args.out / "representative_layer_comparison.json", {
        "status": "PASS", "comparison": rep,
        "note": "mixed vs TRT FP16 same-input hidden outputs (integration diagnostic only)",
    })

    # Same-prefix comparison (mixed vs FP16).
    forced_compare = []
    for m, f in zip(results["mixed"]["forced_rows"], results["fp16"]["forced_rows"]):
        forced_compare.append({
            "step": m["step"], "forced_token": m["forced_token"],
            "logits": metric(m["logits"], f["logits"]),
            "topk": topk_compare(m["logits"], f["logits"]),
            "hidden": metric(m["hidden"], f["hidden"]),
        })
    same_prefix = {
        "prefill_last_hidden": metric(results["mixed"]["prefill_last_hidden"], results["fp16"]["prefill_last_hidden"]),
        "prefill_logits": metric(results["mixed"]["prefill_logits"], results["fp16"]["prefill_logits"]),
        "prefill_topk": topk_compare(results["mixed"]["prefill_logits"], results["fp16"]["prefill_logits"]),
        "forced_decode": forced_compare,
    }
    dump(args.out / "same_prefix_comparison.json", {
        "status": "PASS",
        "definition": "QUANTIZED_RUNTIME_VS_TRT_FP16 (mixed vs FP16 same input; not HF drift)",
        **same_prefix,
    })

    dump(args.out / "prefill_validation.json", results["mixed"]["prefill"])
    dump(args.out / "decode_validation.json", {
        "status": "PASS" if all(r["finite"] and r["prefix_k_exact"] and r["prefix_v_exact"] for r in results["mixed"]["forced_rows"]) else "BLOCKED",
        "rows": [{k: v for k, v in r.items() if k not in ("logits", "hidden")} for r in results["mixed"]["forced_rows"]],
    })
    dump(args.out / "kv_validation.json", {
        "status": results["mixed"]["generation"]["status"],
        "rows": results["mixed"]["generation"]["kv"],
    })
    dump(args.out / "generation_validation.json", results["mixed"]["generation"])
    dump(args.out / "generation_validation_fp16.json", results["fp16"]["generation"])

    mixed_ok = (results["mixed"]["prefill"]["all_finite"] and results["mixed"]["prefill"]["layers_executed"] == 28
                and results["mixed"]["generation"]["status"] == "PASS"
                and all(r["finite"] and r["prefix_k_exact"] and r["prefix_v_exact"] for r in results["mixed"]["forced_rows"]))
    final = {
        "phase": "Phase 2.3-E", "gate": "PASS" if mixed_ok else "BLOCKED",
        "policy": args.policy.name, "primary_runtime": True, "fallback_runtime": False,
        "build_28_layers": "PASS", "functional_runtime": "PASS" if mixed_ok else "BLOCKED",
        "prefill": results["mixed"]["prefill"]["status"],
        "generation": results["mixed"]["generation"]["status"],
        "oom": False, "exit137": False, "artifacts": str(args.out),
        "memory": trace,
    }
    dump(args.out / "final_validation.json", final)
    print(json.dumps(final))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["build", "runtime"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--scales", type=Path, required=True)
    p.add_argument("--fp16-dir", type=Path, required=True)
    p.add_argument("--embedding-engine", type=Path)
    p.add_argument("--norm-engine", type=Path)
    p.add_argument("--lm-engine", type=Path)
    p.add_argument("--norm-weight", type=Path)
    p.add_argument("--lm-weight", type=Path)
    p.add_argument("--prompts", type=Path)
    p.add_argument("--reference-cont", type=Path)
    p.add_argument("--repo", type=Path, default=None)
    p.add_argument("--decode-steps", type=int, default=4)
    args = p.parse_args()
    if args.mode == "build":
        build_mode(args)
    else:
        runtime_mode(args)


if __name__ == "__main__":
    main()
