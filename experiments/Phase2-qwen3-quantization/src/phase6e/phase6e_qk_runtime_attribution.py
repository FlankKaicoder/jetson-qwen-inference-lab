from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import tensorrt as trt


SHAPE_FIELDS = [
    "invocation_id", "phase", "decode_step", "timestamp",
    "query_length", "past_kv_length", "key_length", "tensor_name",
    "tensor_mode", "runtime_shape", "shape_evidence_level",
    "shape_source_api", "engine_or_context_identity",
    "engine_invocation_id", "tensor_dtype", "shape_sample_time",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def command_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=20, check=False)
        return result.stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def environment_snapshot() -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "tegra_release": read_text("/etc/nv_tegra_release"),
        "nvpmodel": command_text(["nvpmodel", "-q"]),
        "jetson_clocks_show": command_text(["jetson_clocks", "--show"]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "tensorrt": trt.__version__,
        "nsys": command_text(["nsys", "--version"]),
        "timestamp": utc_now(),
    }


def shape_tuple(value: object) -> str:
    return json.dumps([int(item) for item in tuple(value)], separators=(",", ":"))


def append_shape_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHAPE_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


class FrozenEngine:
    def __init__(self, path: Path, role: str, expected_sha256: str | None = None):
        self.path = path
        self.role = role
        self.expected_sha256 = expected_sha256
        self.sha256 = sha256_file(path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise RuntimeError(f"ENGINE_SHA256_MISMATCH:{path}")
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"ENGINE_DESERIALIZE_FAILED:{path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"CONTEXT_CREATION_FAILED:{path}")
        self.stream = torch.cuda.current_stream()
        self.invocations = 0
        self.identity = f"{role}|{path}|sha256:{self.sha256}|persistent_context"

    def verify_runtime_api(self) -> dict:
        required = ["set_input_shape", "set_tensor_address", "get_tensor_shape",
                    "get_tensor_strides", "get_tensor_address", "infer_shapes",
                    "execute_async_v3"]
        return {name: hasattr(self.context, name) for name in required}

    def run(self, inputs: dict[str, torch.Tensor], invocation_id: str,
            phase: str, decode_step: int, lengths: tuple[int, int, int],
            shape_log_path: Path) -> dict[str, torch.Tensor]:
        self.invocations += 1
        engine_invocation_id = (
            f"PHASE6E_{self.role}_{self.invocations:04d}"
        )
        nvtx_name = (
            f"PHASE6E_ENGINE_CALL|engine_invocation_id={engine_invocation_id}"
            f"|role={self.role}|runtime_invocation_id={invocation_id}"
        )
        torch.cuda.nvtx.range_push(nvtx_name)
        try:
            for name, value in inputs.items():
                if not self.context.set_input_shape(name, tuple(value.shape)):
                    raise RuntimeError(
                        f"INPUT_SHAPE_REJECTED:{self.role}:{name}:{tuple(value.shape)}")
                self.context.set_tensor_address(name, value.data_ptr())
            outputs: dict[str, torch.Tensor] = {}
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    resolved = tuple(self.context.get_tensor_shape(name))
                    dtype = self.engine.get_tensor_dtype(name)
                    torch_dtype = (torch.float16 if dtype == trt.DataType.HALF
                                   else torch.float32)
                    outputs[name] = torch.empty(resolved, device="cuda",
                                                dtype=torch_dtype)
                    self.context.set_tensor_address(name, outputs[name].data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError(f"EXECUTE_FAILED:{self.role}:{engine_invocation_id}")
            self.stream.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()

        timestamp = utc_now()
        rows: list[dict] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = str(self.engine.get_tensor_mode(name))
            shape = self.context.get_tensor_shape(name)
            self.context.get_tensor_strides(name)
            self.context.get_tensor_address(name)
            rows.append({
                "invocation_id": invocation_id,
                "phase": phase,
                "decode_step": decode_step,
                "timestamp": timestamp,
                "query_length": lengths[0],
                "past_kv_length": lengths[1],
                "key_length": lengths[2],
                "tensor_name": name,
                "tensor_mode": mode,
                "runtime_shape": shape_tuple(shape),
                "shape_evidence_level": "DIRECT_RUNTIME_EVIDENCE",
                "shape_source_api": "IExecutionContext.get_tensor_shape",
                "engine_or_context_identity": self.identity,
                "engine_invocation_id": engine_invocation_id,
                "tensor_dtype": str(self.engine.get_tensor_dtype(name)),
                "shape_sample_time": "after_execute_and_stream_sync",
            })
        append_shape_rows(shape_log_path, rows)
        return outputs


def load_manifest_sample(path: Path) -> tuple[str, list[int]]:
    manifest = json.loads(path.read_text())
    rows = [row for row in manifest["rows"] if row.get("split") == "evaluation"]
    if not rows:
        raise RuntimeError("NO_EVALUATION_MANIFEST_ROW")
    sample = rows[0]
    return str(sample["sample_id"]), [int(token) for token in sample["token_ids"]]


class Workload:
    def __init__(self, args):
        self.args = args
        self.sample_id, self.token_ids = load_manifest_sample(args.eval_manifest)
        expected_ids = [840, 20772, 54809, 23045, 304, 825, 11652, 624]
        if self.token_ids[:8] != expected_ids or self.sample_id != "eva_025":
            raise RuntimeError(f"UNEXPECTED_EVAL_SAMPLE:{self.sample_id}")
        self.force_cont = [int(token) for token in
                           json.loads(args.force_cont.read_text())]
        if self.force_cont != [21806, 0, 358, 2776, 11, 14582, 2585, 1184]:
            raise RuntimeError("UNEXPECTED_FORCED_CONTINUATION")
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

    def embed(self, embedding: FrozenEngine, token_ids: list[int],
              invocation_id: str, phase: str, decode_step: int,
              shape_log_path: Path) -> torch.Tensor:
        tokens = torch.tensor([token_ids], device="cuda", dtype=torch.long)
        return embedding.run({"input_ids": tokens}, invocation_id, phase,
                             decode_step, (len(token_ids), 0, len(token_ids)),
                             shape_log_path)["hidden_states"]

    def logits(self, norm: FrozenEngine, lm: FrozenEngine, hidden: torch.Tensor,
               invocation_id: str, phase: str, decode_step: int,
               shape_log_path: Path) -> torch.Tensor:
        normed = norm.run({"hidden_states": hidden}, invocation_id, phase,
                          decode_step, (int(hidden.shape[-2]), 0,
                                        int(hidden.shape[-2])),
                          shape_log_path)["normalized_hidden_states"]
        return lm.run({"hidden_states": normed}, invocation_id, phase,
                      decode_step, (int(hidden.shape[-2]), 0,
                                    int(hidden.shape[-2])),
                      shape_log_path)["logits"]

    def prefill(self, pre: FrozenEngine, embedding: FrozenEngine, norm: FrozenEngine,
                lm: FrozenEngine, token_ids: list[int], invocation_id: str,
                phase: str, shape_log_path: Path) -> tuple:
        seq_len = len(token_ids)
        hidden = self.embed(embedding, token_ids, invocation_id, phase, -1,
                            shape_log_path)
        positions = torch.arange(seq_len, device="cuda", dtype=torch.long).reshape(1, -1)
        outputs = pre.run({"hidden_states": hidden, "position_ids": positions},
                          invocation_id, phase, -1, (seq_len, 0, seq_len),
                          shape_log_path)
        final_hidden = self.logits(norm, lm, outputs["hidden_l27"][:, -1:, :],
                                   invocation_id, phase, -1, shape_log_path)
        if not bool(torch.isfinite(final_hidden).all()):
            raise RuntimeError(f"NONFINITE_LOGITS:{invocation_id}")
        keys = [outputs[f"present_k{layer}"] for layer in range(28)]
        values = [outputs[f"present_v{layer}"] for layer in range(28)]
        return outputs["hidden_l27"], keys, values

    def decode(self, decode_engine: FrozenEngine, embedding: FrozenEngine,
               norm: FrozenEngine, lm: FrozenEngine, token: int, old_len: int,
               keys: list[torch.Tensor], values: list[torch.Tensor],
               invocation_id: str, phase: str, decode_step: int,
               shape_log_path: Path) -> tuple:
        hidden = self.embed(embedding, [token], invocation_id, phase, decode_step,
                            shape_log_path)
        positions = torch.tensor([[old_len]], device="cuda", dtype=torch.long)
        inputs = {"hidden_states": hidden, "position_ids": positions}
        inputs.update({f"past_k{layer}": keys[layer] for layer in range(28)})
        inputs.update({f"past_v{layer}": values[layer] for layer in range(28)})
        outputs = decode_engine.run(inputs, invocation_id, phase, decode_step,
                                    (1, old_len, old_len + 1), shape_log_path)
        final_hidden = self.logits(norm, lm, outputs["hidden_l27"], invocation_id,
                                   phase, decode_step, shape_log_path)
        if not bool(torch.isfinite(final_hidden).all()):
            raise RuntimeError(f"NONFINITE_LOGITS:{invocation_id}")
        new_keys = [outputs[f"present_k{layer}"] for layer in range(28)]
        new_values = [outputs[f"present_v{layer}"] for layer in range(28)]
        invariants = self.cache_invariants(keys, values, new_keys, new_values,
                                           old_len)
        return outputs["hidden_l27"], new_keys, new_values, invariants

    def cache_invariants(self, old_keys: list[torch.Tensor],
                         old_values: list[torch.Tensor],
                         new_keys: list[torch.Tensor],
                         new_values: list[torch.Tensor], old_len: int) -> dict:
        expected_shape = (1, 8, old_len + 1, 128)
        shape_pass = all(tuple(tensor.shape) == expected_shape
                         for tensor in new_keys + new_values)
        prefix_pass = all(torch.equal(old_keys[layer],
                                      new_keys[layer][:, :, :old_len, :])
                          and torch.equal(old_values[layer],
                                          new_values[layer][:, :, :old_len, :])
                          for layer in range(28))
        return {"expected_shape": expected_shape, "shape_pass": shape_pass,
                "prefix_pass": prefix_pass,
                "old_len": old_len, "new_len": old_len + 1}


def engine_specs(args) -> list[tuple[str, Path, str]]:
    return [
        ("embedding", args.embedding_engine,
         "f13851d5236767e62ef1f596fe95d4f1f721282863f49f1751682a7cd615a8c9"),
        ("final_rmsnorm", args.norm_engine,
         "27724efd7cde97b3b6a71b5722cc2477091770a509c912e872523caf1f7def83"),
        ("lm_head", args.lm_engine,
         "ff9c0dd793345b922083be0a650dae7b5f415411f7ae365ef49ecef7d897a6bd"),
        ("mixed_prefill", args.prefill_engine,
         "3258b0a82fd4884a291e2bf0e3dded179434a9d1a5d034c4fc4e9f772c8964ec"),
        ("mixed_decode", args.decode_engine,
         "445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c"),
    ]


def execute_workload(args, workload: Workload, engines: dict[str, FrozenEngine]) -> dict:
    shape_log_path = args.out / "runtime_shape_log.csv"
    if shape_log_path.exists():
        raise RuntimeError(f"OUTPUT_SHAPE_LOG_EXISTS:{shape_log_path}")
    invariants = []
    ids = workload.token_ids[:8]
    warmup_id = f"PHASE6E_{workload.run_id}_WARMUP_PREFILL_S8"
    torch.cuda.nvtx.range_push(warmup_id)
    try:
        workload.prefill(engines["mixed_prefill"], engines["embedding"],
                         engines["final_rmsnorm"], engines["lm_head"], ids,
                         warmup_id, "WARMUP_PREFILL_S8", shape_log_path)
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    steady_id = f"PHASE6E_{workload.run_id}_STEADY_PREFILL_S8"
    torch.cuda.nvtx.range_push(steady_id)
    try:
        hidden, keys, values = workload.prefill(
            engines["mixed_prefill"], engines["embedding"], engines["final_rmsnorm"],
            engines["lm_head"], ids, steady_id, "STEADY_PREFILL_S8", shape_log_path)
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    for step in range(4):
        token = workload.force_cont[step]
        old_len = 8 + step
        invocation_id = f"PHASE6E_{workload.run_id}_DECODE_STEP_{step}"
        torch.cuda.nvtx.range_push(invocation_id)
        try:
            hidden, keys, values, invariant = workload.decode(
                engines["mixed_decode"], engines["embedding"], engines["final_rmsnorm"],
                engines["lm_head"], token, old_len, keys, values, invocation_id,
                "DECODE", step, shape_log_path)
        finally:
            torch.cuda.nvtx.range_pop()
        invariants.append(invariant)
        torch.cuda.synchronize()
        if not invariant["shape_pass"] or not invariant["prefix_pass"]:
            raise RuntimeError(f"CACHE_INVARIANT_FAILED:{step}:{invariant}")

    return {
        "run_id": workload.run_id,
        "sample_id": workload.sample_id,
        "prefill_token_ids": ids,
        "forced_tokens_steps_0_3": workload.force_cont[:4],
        "engine_invocation_counts": {role: engine.invocations
                                     for role, engine in engines.items()},
        "cache_invariants": invariants,
        "runtime_shape_log": str(shape_log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embedding-engine", type=Path, required=True)
    parser.add_argument("--norm-engine", type=Path, required=True)
    parser.add_argument("--lm-engine", type=Path, required=True)
    parser.add_argument("--prefill-engine", type=Path, required=True)
    parser.add_argument("--decode-engine", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--force-cont", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise RuntimeError(f"OUTPUT_DIRECTORY_EXISTS:{args.out}")
    args.out.mkdir(parents=True)
    workload = Workload(args)
    environment = environment_snapshot()
    required_api = {"set_input_shape", "set_tensor_address", "get_tensor_shape",
                    "get_tensor_strides", "get_tensor_address", "infer_shapes",
                    "execute_async_v3"}
    engines = {}
    for role, path, sha256 in engine_specs(args):
        engine = FrozenEngine(path, role, sha256)
        available = engine.verify_runtime_api()
        if not required_api.issubset(set(available)):
            raise RuntimeError(f"RUNTIME_API_UNAVAILABLE:{role}:{available}")
        engines[role] = engine
    api_checks = {role: engine.verify_runtime_api() for role, engine in engines.items()}

    config = {
        "experiment": "Phase6E",
        "run_type": "NEW_CONTROLLED_RUN",
        "runtime": "mixed",
        "lifetime": "persistent_context_lifetime",
        "workload": "one_warmup_prefill_S8_one_steady_prefill_S8_decode_steps_0_3",
        "sample_id": workload.sample_id,
        "prefill_token_ids": workload.token_ids[:8],
        "forced_tokens": workload.force_cont,
        "engine_paths": {role: str(path) for role, path, _ in engine_specs(args)},
        "engine_sha256": {role: engine.sha256 for role, engine in engines.items()},
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "clock_or_power_modified": False,
        "tactic_forcing": False,
        "engine_rebuild": False,
        "onnx_change": False,
        "precision_change": False,
    }
    (args.out / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n")
    (args.out / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.out / "runtime_api_check.json").write_text(
        json.dumps(api_checks, indent=2, sort_keys=True) + "\n")
    result = execute_workload(args, workload, engines)
    (args.out / "workload_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
