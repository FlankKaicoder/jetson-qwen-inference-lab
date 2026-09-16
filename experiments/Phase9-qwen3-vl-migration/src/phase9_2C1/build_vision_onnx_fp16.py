#!/usr/bin/env python3
import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enum_name(value):
    return getattr(value, "name", str(value))


def flag_names(flags):
    return sorted(
        name
        for name, flag in trt.BuilderFlag.__members__.items()
        if int(flag) & flags
    )


def scalar(value):
    if isinstance(value, dict):
        return value.get("Name") or value.get("name") or value.get("Value") or value.get("value")
    return value


def collect_tactic_info(value):
    tactic_values = []
    tactic_names = []

    def walk(node):
        if isinstance(node, dict):
            for key, item in node.items():
                if key in ("TacticValue", "tactic_value"):
                    tactic_values.append(str(item))
                if key in ("TacticName", "tactic_name"):
                    tactic_names.append(str(item))
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(value)
    return tactic_values, tactic_names


class BuildLogger(trt.ILogger):
    def __init__(self, log_handle):
        super().__init__()
        self.log_handle = log_handle
        self.severity_counts = Counter()
        self.selected_records = []
        self.tactic_message_count = 0

    def log(self, severity, message):
        try:
            severity_name = severity.name
        except AttributeError:
            severity_name = str(severity)
        self.severity_counts[severity_name] += 1
        self.log_handle.write(f"[TRT:{severity_name}] {message}\n")
        self.log_handle.flush()
        if severity_name != "VERBOSE" or "tactic" in message.lower():
            self.selected_records.append({"severity": severity_name, "message": message})
        if "tactic" in message.lower():
            self.tactic_message_count += 1


def tensor_record(name, shape, dtype, mode=None, location=None, format_desc=None):
    record = {"name": name, "shape": list(shape), "dtype": enum_name(dtype)}
    if mode is not None:
        record["mode"] = enum_name(mode)
    if location is not None:
        record["location"] = enum_name(location)
    if format_desc is not None:
        record["format"] = format_desc
    return record


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--onnx-path", required=True)
    argument_parser.add_argument("--engine-path", required=True)
    argument_parser.add_argument("--result-path", required=True)
    args = argument_parser.parse_args()

    onnx_path = Path(args.onnx_path).resolve()
    engine_path = Path(args.engine_path).resolve()
    result_path = Path(args.result_path).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    full_log_path = result_path.parent / "build_console.log"
    selected_log_path = result_path.parent / "build_log_selected.jsonl"
    layer_info_path = result_path.parent / "engine_layer_information.jsonl"
    started_utc = datetime.now(timezone.utc).isoformat()

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX graph not found: {onnx_path}")

    with full_log_path.open("w", encoding="utf-8") as log_handle:
        logger = BuildLogger(log_handle)
        result = {
            "phase": "Phase 9.2-C1",
            "audit_mode": "TENSORRT_FP16_ENGINE_BUILD_ONLY",
            "started_utc": started_utc,
            "hostname": platform.node(),
            "tensorrt_version": trt.__version__,
            "onnx_file": {
                "path": str(onnx_path),
                "size_bytes": onnx_path.stat().st_size,
                "sha256": sha256_file(onnx_path),
            },
            "runtime": {
                "runtime_created": False,
                "execution_context_created": False,
            },
        }
        try:
            builder = trt.Builder(logger)
            explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(explicit_batch)
            onnx_parser = trt.OnnxParser(network, logger)
            parse_returned = bool(onnx_parser.parse_from_file(str(onnx_path)))
            parser_errors = []
            for index in range(onnx_parser.num_errors):
                error = onnx_parser.get_error(index)
                parser_errors.append(
                    {
                        "index": index,
                        "code": enum_name(error.code),
                        "description": error.desc,
                        "file": error.file(),
                        "line": error.line(),
                        "function": error.func(),
                        "node": error.node(),
                    }
                )
            network_layer_type_counts = Counter(
                enum_name(network.get_layer(index).type) for index in range(network.num_layers)
            )
            result["parser"] = {
                "invoked": True,
                "invocation_method": "OnnxParser.parse_from_file",
                "parse_returned": parse_returned,
                "num_errors": onnx_parser.num_errors,
                "errors": parser_errors,
                "success": parse_returned and onnx_parser.num_errors == 0,
            }
            result["network"] = {
                "creation_flags": ["EXPLICIT_BATCH"],
                "num_layers": network.num_layers,
                "num_inputs": network.num_inputs,
                "num_outputs": network.num_outputs,
                "layer_type_counts": dict(sorted(network_layer_type_counts.items())),
                "inputs": [
                    tensor_record(
                        network.get_input(index).name,
                        network.get_input(index).shape,
                        network.get_input(index).dtype,
                    )
                    for index in range(network.num_inputs)
                ],
                "outputs": [
                    tensor_record(
                        network.get_output(index).name,
                        network.get_output(index).shape,
                        network.get_output(index).dtype,
                    )
                    for index in range(network.num_outputs)
                ],
            }

            if not result["parser"]["success"]:
                raise RuntimeError("ONNX parsing failed before engine build")

            config = builder.create_builder_config()
            memory_pool_types = [
                trt.MemoryPoolType.WORKSPACE,
                trt.MemoryPoolType.TACTIC_DRAM,
                trt.MemoryPoolType.TACTIC_SHARED_MEMORY,
            ]
            memory_pool_limits_before = {
                pool.name: config.get_memory_pool_limit(pool) for pool in memory_pool_types
            }
            config.set_flag(trt.BuilderFlag.FP16)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, 1 << 30)
            memory_pool_limits_after = {
                pool.name: config.get_memory_pool_limit(pool) for pool in memory_pool_types
            }
            result["builder_config"] = {
                "created": True,
                "flags_before": flag_names(config.flags),
                "flags_after": flag_names(config.flags),
                "fp16_flag_set": config.get_flag(trt.BuilderFlag.FP16),
                "profiling_verbosity": enum_name(config.profiling_verbosity),
                "builder_optimization_level": config.builder_optimization_level,
                "avg_timing_iterations": config.avg_timing_iterations,
                "num_optimization_profiles": config.num_optimization_profiles,
                "memory_pool_limits_before": memory_pool_limits_before,
                "memory_pool_limits_after": memory_pool_limits_after,
                "timing_cache_used": False,
                "algorithm_selector_used": False,
            }

            build_started_at = time.perf_counter()
            serialized_network = None
            build_exception = None
            try:
                serialized_network = builder.build_serialized_network(network, config)
            except Exception as exc:
                build_exception = {"type": type(exc).__name__, "message": str(exc)}
            build_elapsed = time.perf_counter() - build_started_at

            result["engine_build"] = {
                "invoked": True,
                "method": "Builder.build_serialized_network",
                "success": serialized_network is not None,
                "serialized_network_returned": serialized_network is not None,
                "build_exception": build_exception,
                "build_elapsed_seconds": build_elapsed,
            }

            if serialized_network is not None:
                engine_data = memoryview(serialized_network)
                with engine_path.open("wb") as engine_handle:
                    engine_handle.write(engine_data)

                result["engine_file"] = {
                    "path": str(engine_path),
                    "size_bytes": engine_path.stat().st_size,
                    "sha256": sha256_file(engine_path),
                }

                runtime = trt.Runtime(logger)
                result["runtime"] = {
                    "runtime_created": True,
                    "runtime_created_for_metadata_only": True,
                    "execution_context_created": False,
                    "engine_deserialized_for_metadata_only": True,
                    "engine_executed": False,
                }
                engine = None
                deserialize_exception = None
                try:
                    engine = runtime.deserialize_cuda_engine(engine_data)
                except Exception as exc:
                    deserialize_exception = {"type": type(exc).__name__, "message": str(exc)}
                result["engine_deserialization"] = {
                    "success": engine is not None,
                    "exception": deserialize_exception,
                }

                if engine is not None:
                    engine_io_tensors = []
                    for index in range(engine.num_io_tensors):
                        name = engine.get_tensor_name(index)
                        engine_io_tensors.append(
                            tensor_record(
                                name,
                                engine.get_tensor_shape(name),
                                engine.get_tensor_dtype(name),
                                engine.get_tensor_mode(name),
                                engine.get_tensor_location(name),
                                engine.get_tensor_format_desc(name),
                            )
                        )

                    result["engine_metadata"] = {
                        "name": engine.name,
                        "num_layers": engine.num_layers,
                        "num_io_tensors": engine.num_io_tensors,
                        "num_optimization_profiles": engine.num_optimization_profiles,
                        "device_memory_size": engine.device_memory_size,
                        "device_memory_size_v2": engine.device_memory_size_v2,
                        "streamable_weights_size": engine.streamable_weights_size,
                        "refittable": engine.refittable,
                        "profiling_verbosity": enum_name(engine.profiling_verbosity),
                        "engine_capability": enum_name(engine.engine_capability),
                        "hardware_compatibility_level": enum_name(engine.hardware_compatibility_level),
                        "tactic_sources": enum_name(engine.tactic_sources),
                        "num_aux_streams": engine.num_aux_streams,
                        "io_tensors": engine_io_tensors,
                    }

                    engine_inspector = engine.create_engine_inspector()
                    layer_type_counts = Counter()
                    tactic_value_counts = Counter()
                    tactic_name_counts = Counter()
                    layer_info_parse_failures = []
                    layer_info_sample = []
                    with layer_info_path.open("w", encoding="utf-8") as layer_info_handle:
                        for index in range(engine.num_layers):
                            try:
                                layer_info = engine_inspector.get_layer_information(
                                    index, trt.LayerInformationFormat.JSON
                                )
                                layer_info_handle.write(layer_info + "\n")
                                layer_info_json = json.loads(layer_info)
                                layer_type_counts[str(scalar(layer_info_json.get("LayerType")))] += 1
                                tactic_values, tactic_names = collect_tactic_info(layer_info_json)
                                tactic_value_counts.update(tactic_values)
                                tactic_name_counts.update(tactic_names)
                                if index == 0 or index == engine.num_layers - 1:
                                    layer_info_sample.append(
                                        {"index": index, "information": layer_info_json}
                                    )
                            except Exception as exc:
                                layer_info_parse_failures.append(
                                    {"index": index, "type": type(exc).__name__, "message": str(exc)}
                                )

                    result["engine_layer_summary"] = {
                        "layer_type_counts": dict(sorted(layer_type_counts.items())),
                        "tactic_value_counts": dict(sorted(tactic_value_counts.items())),
                        "tactic_name_counts": dict(sorted(tactic_name_counts.items())),
                        "tactic_summary_available": bool(tactic_value_counts or tactic_name_counts),
                        "parse_failures": layer_info_parse_failures,
                        "samples": layer_info_sample,
                    }

                    result["engine_layer_information"] = {
                        "path": str(layer_info_path),
                        "size_bytes": layer_info_path.stat().st_size,
                        "sha256": sha256_file(layer_info_path),
                    }
        except Exception as exc:
            result["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

        result["build_log"] = {
            "full_log_path": str(full_log_path),
            "full_log_size_bytes": full_log_path.stat().st_size,
            "full_log_sha256": sha256_file(full_log_path),
            "selected_log_path": str(selected_log_path),
            "selected_record_count": len(logger.selected_records),
            "tactic_message_count": logger.tactic_message_count,
            "severity_counts": dict(sorted(logger.severity_counts.items())),
            "warning_count": logger.severity_counts.get("WARNING", 0),
            "error_count": logger.severity_counts.get("ERROR", 0)
            + logger.severity_counts.get("INTERNAL_ERROR", 0),
        }

        result["prohibited_operations_performed"] = {
            "execution_context_created": False,
            "engine_executed": False,
            "benchmark_run": False,
            "correctness_comparison_run": False,
            "quantization_performed": False,
            "optimization_performed": False,
            "cuda_kernel_modified": False,
            "environment_modified": False,
            "tactic_selection_forced": False,
            "timing_cache_used": False,
        }

        selected_log_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in logger.selected_records),
            encoding="utf-8",
        )
        result["build_log"]["selected_log_size_bytes"] = selected_log_path.stat().st_size
        result["build_log"]["selected_log_sha256"] = sha256_file(selected_log_path)
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()

        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("PHASE9_2C1_ENGINE_BUILD_DONE", flush=True)
        print(json.dumps(result["build_log"], sort_keys=True), flush=True)
        print(
            "engine_build_success="
            + str(result.get("engine_build", {}).get("success", False)),
            flush=True,
        )
        if not result.get("engine_build", {}).get("success", False):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
