#!/usr/bin/env python3
"""Recover the Phase 4-A unknown /MatMul_* chain from committed evidence only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metadata_nodes(metadata: str) -> list[str]:
    return re.findall(r"\[ONNX Layer: ([^]]+)\]", metadata)


def dims(tensor: dict[str, Any]) -> str:
    return "x".join(str(v) for v in tensor.get("Dimensions", [])) or "UNKNOWN"


def precision(tensor: dict[str, Any]) -> str:
    value = tensor.get("Format/Datatype", "UNKNOWN")
    return str(value)


def tensors_summary(tensors: list[dict[str, Any]]) -> str:
    return ";".join(f"{t['Name']}[{dims(t)}:{precision(t)}]" for t in tensors) or "NONE"


def operand_summary(tensors: list[dict[str, Any]]) -> str:
    return ";".join(f"{t['Name']}[{dims(t)}]" for t in tensors) or "NONE"


def has_operand_ref(
    producer_rows: list[tuple[int, dict[str, Any], list[str]]],
    operand_index: int,
    required_refs: set[str],
) -> bool:
    return any(
        index == operand_index and all(ref in refs for ref in required_refs)
        for index, _, refs in producer_rows
    )


def has_operand_ref_prefix(
    producer_rows: list[tuple[int, dict[str, Any], list[str]]],
    operand_index: int,
    required_prefixes: set[str],
) -> bool:
    return any(
        index == operand_index
        and all(any(ref.startswith(prefix) for ref in refs) for prefix in required_prefixes)
        for index, _, refs in producer_rows
    )


def load_nodes(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[str]], dict[str, list[str]]]:
    by_name: dict[str, dict[str, str]] = {}
    producers: dict[str, list[str]] = defaultdict(list)
    consumers: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_name[row["node_name"]] = row
            for tensor in row["inputs"].split(";"):
                if tensor and tensor != "NONE":
                    producers[tensor].append(row["node_name"])
            for tensor in row["outputs"].split(";"):
                if tensor and tensor != "NONE":
                    consumers[tensor].append(row["node_name"])
    return by_name, producers, consumers


def graph_neighborhood(
    node_name: str,
    by_name: dict[str, dict[str, str]],
    producers: dict[str, list[str]],
    consumers: dict[str, list[str]],
    distance: int = 3,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    queue = [(node_name, "upstream", 0, None)]
    while queue:
        current, direction, hop, via_tensor = queue.pop(0)
        current_row = by_name.get(current)
        if current_row is None or hop >= distance:
            continue
        neighbor_names: list[str]
        if direction == "upstream":
            tensor_names = current_row["inputs"].split(";")
            neighbor_lookup = producers
            neighbor_names = [
                producer
                for tensor in tensor_names
                if tensor and tensor != "NONE"
                for producer in neighbor_lookup.get(tensor, [])
            ]
        else:
            tensor_names = current_row["outputs"].split(";")
            neighbor_lookup = consumers
            neighbor_names = [
                consumer
                for tensor in tensor_names
                if tensor and tensor != "NONE"
                for consumer in neighbor_lookup.get(tensor, [])
            ]
        for neighbor_name in sorted(set(neighbor_names)):
            neighbor = by_name.get(neighbor_name)
            if neighbor is None:
                continue
            connected_tensors = [
                tensor
                for tensor in tensor_names
                if tensor and tensor != "NONE"
                and neighbor_name in neighbor_lookup.get(tensor, [])
            ]
            rows.append(
                {
                    "candidate_id": "",
                    "onnx_node": node_name,
                    "hop_direction": direction,
                    "hop_distance": str(hop + 1),
                    "neighbor_node": neighbor_name,
                    "neighbor_op_type": neighbor["op_type"],
                    "neighbor_inputs": neighbor["inputs"],
                    "neighbor_outputs": neighbor["outputs"],
                    "shape": "UNKNOWN",
                    "semantic_role": "UNKNOWN",
                    "evidence": f"ONNX_NODE_INVENTORY_TENSOR_EDGE via {','.join(connected_tensors)}",
                }
            )
            queue.append((neighbor_name, direction, hop + 1, None))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-summary", required=True)
    parser.add_argument("--decode-inspector", required=True)
    parser.add_argument("--onnx-node-inventory", required=True)
    parser.add_argument("--runtime-mapping", required=True)
    parser.add_argument("--gpu-kernel-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    prefill_summary_path = Path(args.prefill_summary)
    prefill_summary = json.loads(prefill_summary_path.read_text(encoding="utf-8"))
    prefill_layers = json.loads(prefill_summary["engine_inspector"]["prefill"])["Layers"]
    decode_layers = json.loads(prefill_summary["engine_inspector"]["decode"])["Layers"]
    decode_inspector_path = Path(args.decode_inspector)
    decode_inspector_text = decode_inspector_path.read_text(encoding="utf-8")
    decode_local_layers = json.loads(decode_inspector_text)["Layers"]
    if decode_local_layers != decode_layers:
        raise RuntimeError("Decode EngineInspector copies differ")

    decode_hash = sha256(decode_inspector_path)
    prefill_hash = sha256_text(prefill_summary["engine_inspector"]["prefill"])
    decode_embedded_hash = sha256_text(prefill_summary["engine_inspector"]["decode"])
    if decode_hash != decode_embedded_hash:
        raise RuntimeError("Decode inspector hash differs from embedded summary")

    by_name, producers, consumers = load_nodes(Path(args.onnx_node_inventory))
    runtime_rows = list(csv.DictReader(Path(args.runtime_mapping).open("r", encoding="utf-8", newline="")))
    gpu_rows = list(csv.DictReader(Path(args.gpu_kernel_summary).open("r", encoding="utf-8", newline="")))
    gpu_total_ns = sum(int(float(row["Total Time (ns)"])) for row in gpu_rows)

    candidate_pattern = re.compile(r"^/MatMul(?:_(\d+))?_myl0_(\d+)$")
    candidates = []
    for index, layer in enumerate(decode_layers):
        match = candidate_pattern.match(layer["Name"])
        if not match:
            continue
        node_suffix = match.group(1)
        node_name = f"/MatMul_{node_suffix}" if node_suffix else "/MatMul"
        node_index = int(node_suffix) if node_suffix else 3
        if by_name.get(node_name, {}).get("op_type") != "MatMul":
            raise RuntimeError(f"Missing or non-MatMul ONNX node {node_name}")
        candidates.append((index, layer, node_name, node_index, int(node_suffix or 0)))
    if len(candidates) != 56:
        raise RuntimeError(f"Expected 56 decode /MatMul_* candidates, found {len(candidates)}")

    decode_output_by_tensor = {}
    for layer in decode_layers:
        for tensor in layer.get("Outputs", []):
            decode_output_by_tensor.setdefault(tensor["Name"], []).append(layer)

    inventory_rows = []
    graph_rows = []
    semantic_rows = []
    context_rows = []
    fusion_rows = []
    contribution_rows = []
    qk_count = av_count = 0

    for candidate_id, (decode_index, layer, node_name, node_index, node_number) in enumerate(candidates, start=1):
        even_number = node_number % 2 == 0
        decoder_layer = node_number // 2
        operands = [t for t in layer.get("Inputs", []) if t.get("Format/Datatype") == "Half"]
        if len(operands) != 2:
            raise RuntimeError(f"Unexpected Half operands for {node_name}")
        candidate_output = layer.get("Outputs", [])[0]
        tactic = layer.get("TacticName", "UNKNOWN")
        metadata = layer.get("Metadata", "")
        refs = metadata_nodes(metadata)

        producer_rows = []
        for operand_index, operand in enumerate(operands):
            producing = decode_output_by_tensor.get(operand["Name"], [])
            for producer in producing:
                producer_refs = metadata_nodes(producer.get("Metadata", ""))
                producer_rows.append((operand_index, producer, producer_refs))

        q_norm_prefix = f"/q_norm{'_' + str(decoder_layer) if decoder_layer else ''}"
        k_norm_prefix = f"/k_norm{'_' + str(decoder_layer) if decoder_layer else ''}"
        softmax_anchor = f"/Softmax{'_' + str(decoder_layer) if decoder_layer else ''}"
        v_anchor = f"/v_proj{'_' + str(decoder_layer) if decoder_layer else ''}"
        q_provenance = has_operand_ref(
            producer_rows,
            0,
            {f"{q_norm_prefix}/Div", f"{q_norm_prefix}/Mul"},
        )
        k_provenance = has_operand_ref(
            producer_rows,
            1,
            {
                f"{k_norm_prefix}/Div",
                f"{k_norm_prefix}/Mul_1",
            },
        ) and has_operand_ref_prefix(producer_rows, 1, {"/Reshape_", "/Transpose_"})
        softmax_provenance = has_operand_ref(producer_rows, 0, {softmax_anchor})
        v_provenance = has_operand_ref(
            producer_rows,
            1,
            {v_anchor},
        ) or has_operand_ref(producer_rows, 1, {f"/Expand_{node_number}"})

        semantic_identity = "UNKNOWN"
        confidence = "UNKNOWN"
        anchor_evidence = []
        counter_evidence = []
        if even_number:
            expected_q_shape = "16x1x128"
            expected_k_shape = "16x128x-1"
            expected_out_shape = "16x1x-1"
            shape_ok = (
                dims(operands[0]) == expected_q_shape
                and dims(operands[1]) == expected_k_shape
                and dims(candidate_output) == expected_out_shape
            )
            downstream_softmax = any(
                f"/Softmax{'_' + str(decoder_layer) if decoder_layer else ''}" in consumer["Metadata"]
                for consumer in decode_layers
                if any(t["Name"] == candidate_output["Name"] for t in consumer.get("Inputs", []))
            )
            if q_provenance and k_provenance and shape_ok and downstream_softmax:
                semantic_identity = "QK^T"
                confidence = "HIGH"
                qk_count += 1
                anchor_evidence.extend(
                    [
                        f"Q producer references {q_norm_prefix}/Div and {q_norm_prefix}/Mul",
                        f"K producer references {k_norm_prefix}/Div, {k_norm_prefix}/Mul_1, /Reshape_* and /Transpose_*",
                        f"TRT layer {decode_index} tactic and Q/K shape compatibility",
                        "Immediate TRT consumer contains Softmax",
                    ]
                )
            else:
                anchor_evidence.extend(["shape-compatible" if shape_ok else "shape mismatch"])
                counter_evidence.extend(
                    [
                        f"Q provenance={q_provenance}",
                        f"K provenance={k_provenance}",
                        f"downstream Softmax={downstream_softmax}",
                    ]
                )
        else:
            expected_prob_shape = "16x1x-1"
            expected_v_shape = "16x-1x128"
            expected_out_shape = "16x1x128"
            shape_ok = (
                dims(operands[0]) == expected_prob_shape
                and dims(operands[1]) == expected_v_shape
                and dims(candidate_output) == expected_out_shape
            )
            downstream_o_proj = any(
                f"/o_proj{'_' + str(decoder_layer) if decoder_layer else ''}" in consumer["Metadata"]
                for consumer in decode_layers
                if any(t["Name"] == candidate_output["Name"] for t in consumer.get("Inputs", []))
            )
            if softmax_provenance and v_provenance and shape_ok and downstream_o_proj:
                semantic_identity = "Attention_x_V"
                confidence = "HIGH"
                av_count += 1
                anchor_evidence.extend(
                    [
                        "Softmax output is first operand producer input",
                        f"V/GQA Expand producer references {v_anchor}",
                        f"TRT layer {decode_index} tactic and probability/V shape compatibility",
                        "Immediate TRT consumer contains o_proj",
                    ]
                )
            else:
                anchor_evidence.extend(["shape-compatible" if shape_ok else "shape mismatch"])
                counter_evidence.extend(
                    [
                        f"Softmax provenance={softmax_provenance}",
                        f"V provenance={v_provenance}",
                        f"downstream o_proj={downstream_o_proj}",
                    ]
                )

        prefill_fused = [
            (index, prefill_layer)
            for index, prefill_layer in enumerate(prefill_layers)
            if prefill_layer["Name"].startswith("_gemm_mha_v2_")
            and node_name in metadata_nodes(prefill_layer.get("Metadata", ""))
        ]
        prefill_present = "YES" if len(prefill_fused) == 1 else "UNKNOWN"
        decode_present = "YES"
        runtime_context = "SHARED_SEMANTIC_PREFILL_FUSED_DECODE_STANDALONE" if prefill_present == "YES" else "UNKNOWN"
        if prefill_present != "YES" or decode_present != "YES":
            confidence = "UNKNOWN" if confidence == "HIGH" else confidence

        runtime_matches = [
            row
            for row in runtime_rows
            if row["source_node_name"] == node_name
            and row["trt_layer_name"] == layer["Name"]
        ]
        if not runtime_matches:
            raise RuntimeError(f"No runtime mapping row for {node_name}")
        calls = sum(int(row["kern_instances"]) for row in runtime_matches)
        total_ns = sum(int(float(row["total_time_ns"])) for row in runtime_matches)
        tactic_consistent_ns = sum(
            int(float(row["total_time_ns"]))
            for row in runtime_matches
            if row["kernel_name"] == f"{tactic}_execute_kernel_trt"
        )
        tactic_consistent_calls = sum(
            int(row["kern_instances"])
            for row in runtime_matches
            if row["kernel_name"] == f"{tactic}_execute_kernel_trt"
        )

        row_sources = ";".join(row["kernel_name"] for row in runtime_matches)
        inventory_rows.append(
            {
                "candidate_id": f"P6A_{candidate_id:03d}",
                "trt_layer_name": layer["Name"],
                "onnx_node_name": node_name,
                "onnx_node_index": str(node_index),
                "op_type": "MatMul",
                "input_0": operand_summary([operands[0]]),
                "input_1": operand_summary([operands[1]]),
                "output": operand_summary([candidate_output]),
                "input_0_shape": dims(operands[0]),
                "input_1_shape": dims(operands[1]),
                "output_shape": dims(candidate_output),
                "dtype": f"{precision(operands[0])}/{precision(operands[1])}->{precision(candidate_output)}",
                "decoder_layer": str(decoder_layer),
                "runtime_context": runtime_context,
                "historical_calls": str(calls),
                "historical_total_time": f"{total_ns} ns",
                "historical_mean_time": f"{total_ns / calls:.3f} ns" if calls else "UNKNOWN",
                "historical_measurement_boundary": "PHASE3C_MIXED_PERSISTENT_NSYS_NVTX_KERNEL_MAPPING_ALL_TRACE",
                "source_artifact": "results/phase4a_operator_attribution/20260904T134556Z/runtime_nvtx_kernel_mapping.csv",
                "confidence": confidence,
                "notes": f"kernel rows={len(runtime_matches)}; kernels={row_sources}; tactic-consistent time={tactic_consistent_ns} ns",
            }
        )

        for neighbor in graph_neighborhood(node_name, by_name, producers, consumers):
            neighbor["candidate_id"] = f"P6A_{candidate_id:03d}"
            graph_rows.append(neighbor)

        semantic_rows.append(
            {
                "candidate_id": f"P6A_{candidate_id:03d}",
                "onnx_node": node_name,
                "decoder_layer": str(decoder_layer),
                "input_shapes": f"{dims(operands[0])};{dims(operands[1])}",
                "output_shape": dims(candidate_output),
                "graph_neighborhood_summary": "ONNX_NODE_INVENTORY_3HOP;see graph_neighborhood.csv",
                "semantic_hypothesis": semantic_identity,
                "semantic_identity": semantic_identity,
                "confidence": confidence,
                "anchor_evidence": ";".join(anchor_evidence),
                "counter_evidence": ";".join(counter_evidence) or "NONE",
                "unknowns": "TRT internal kernel implementation identity beyond layer/tactic is UNKNOWN",
            }
        )

        context_rows.append(
            {
                "candidate_id": f"P6A_{candidate_id:03d}",
                "semantic_identity": semantic_identity,
                "prefill_present": prefill_present,
                "decode_present": decode_present,
                "prefill_evidence": (
                    f"prefill EngineInspector layer {prefill_fused[0][0]} {prefill_fused[0][1]['Name']}"
                    if prefill_present == "YES"
                    else "NO_MATCHING_PREFILL_LAYER"
                ),
                "decode_evidence": f"decode EngineInspector layer {decode_index} {layer['Name']}",
                "runtime_context": runtime_context,
                "confidence": "HIGH" if prefill_present == "YES" else "UNKNOWN",
                "notes": "prefill uses fused _gemm_mha_v2; decode exposes standalone ONNX-named GEMM layers",
            }
        )

        fusion_rows.append(
            {
                "candidate_id": f"P6A_{candidate_id:03d}",
                "onnx_node": node_name,
                "decoder_layer": str(decoder_layer),
                "onnx_semantic_identity": semantic_identity,
                "prefill_trt_layer": prefill_fused[0][1]["Name"] if prefill_present == "YES" else "UNKNOWN",
                "prefill_fusion_status": "FUSED_INTO_gemm_mha_v2" if prefill_present == "YES" else "UNKNOWN",
                "decode_trt_layer": layer["Name"],
                "decode_fusion_status": "STANDALONE_ONNX_NAMED_GEMM_LAYER",
                "runtime_implementation_identity": (
                    "PREFILL_FUSED;DECODE_STANDALONE;CUDA_KERNEL_SEMANTICS_UNKNOWN"
                    if prefill_present == "YES"
                    else "UNKNOWN"
                ),
                "evidence": f"prefill EngineInspector metadata={node_name}; decode EngineInspector metadata={node_name}",
                "confidence": "HIGH" if prefill_present == "YES" else "UNKNOWN",
            }
        )

        contribution_rows.append(
            {
                "candidate_id": f"P6A_{candidate_id:03d}",
                "onnx_node": node_name,
                "decoder_layer": str(decoder_layer),
                "semantic_identity": semantic_identity,
                "runtime_rows": str(len(runtime_matches)),
                "calls": str(calls),
                "all_rows_total_time_ns": str(total_ns),
                "tactic_consistent_calls": str(tactic_consistent_calls),
                "tactic_consistent_total_time_ns": str(tactic_consistent_ns),
                "gpu_total_time_ns": str(gpu_total_ns),
                "all_rows_gpu_share": f"{100.0 * total_ns / gpu_total_ns:.6f}%",
                "tactic_consistent_gpu_share": f"{100.0 * tactic_consistent_ns / gpu_total_ns:.6f}%",
                "measurement_boundary": "PHASE3C_MIXED_PERSISTENT_NSYS_NVTX_KERNEL_MAPPING_ALL_TRACE",
                "notes": "do not compare to CUDA Event, IProfiler, or NCU durations directly",
            }
        )

    family_total_ns = sum(int(row["all_rows_total_time_ns"]) for row in contribution_rows)
    family_tactic_ns = sum(int(row["tactic_consistent_total_time_ns"]) for row in contribution_rows)
    summary = {
        "phase": "Phase 6-A Unknown Attention MatMul Attribution Recovery",
        "method": "Offline read-only recovery from committed Phase 2.3-E, Phase 4-A.1, Phase 4-A.2 and Phase 3-C artifacts. No engine rebuild, ONNX export, execution, benchmark, or profiling.",
        "artifact_manifest": {
            "prefill_and_decode_engine_inspector_summary": {
                "path": str(prefill_summary_path),
                "sha256": sha256(prefill_summary_path),
                "bytes": prefill_summary_path.stat().st_size,
            },
            "decode_engine_inspector_standalone": {
                "path": str(decode_inspector_path),
                "sha256": decode_hash,
                "bytes": decode_inspector_path.stat().st_size,
            },
            "onnx_node_inventory": {
                "path": str(Path(args.onnx_node_inventory)),
                "sha256": sha256(Path(args.onnx_node_inventory)),
                "bytes": Path(args.onnx_node_inventory).stat().st_size,
            },
            "runtime_nvtx_kernel_mapping": {
                "path": str(Path(args.runtime_mapping)),
                "sha256": sha256(Path(args.runtime_mapping)),
                "bytes": Path(args.runtime_mapping).stat().st_size,
            },
            "mixed_persistent_cuda_gpu_kern_sum": {
                "path": str(Path(args.gpu_kernel_summary)),
                "sha256": sha256(Path(args.gpu_kernel_summary)),
                "bytes": Path(args.gpu_kernel_summary).stat().st_size,
            },
        },
        "counts": {
            "decode_matmul_candidates": len(candidates),
            "qk_high": qk_count,
            "attention_v_high": av_count,
            "prefill_fused_matches": sum(1 for row in context_rows if row["prefill_present"] == "YES"),
            "decode_standalone_matches": len(candidates),
        },
        "contribution_reconciliation": {
            "all_rows_total_time_ms": round(family_total_ns / 1_000_000.0, 6),
            "tactic_consistent_total_time_ms": round(family_tactic_ns / 1_000_000.0, 6),
            "gpu_total_time_ms": round(gpu_total_ns / 1_000_000.0, 6),
            "all_rows_gpu_share_percent": round(100.0 * family_total_ns / gpu_total_ns, 6),
            "tactic_consistent_gpu_share_percent": round(100.0 * family_tactic_ns / gpu_total_ns, 6),
            "measurement_boundary": "Phase 3-C Mixed persistent NSYS derived NVTX-to-kernel aggregate across representative prefill plus four decode steps",
        },
        "gate_interpretation": "Candidate semantics are recovered at HIGH confidence, but the historical 61.815776 ms includes a tactic-inconsistent h16816 aggregate; the tactic-consistent f16f32 xmma subset is smaller. No implementation is authorized.",
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_specs = {
        "candidate_inventory.csv": inventory_rows,
        "graph_neighborhood.csv": graph_rows,
        "semantic_attribution.csv": semantic_rows,
        "runtime_context_attribution.csv": context_rows,
        "fusion_rewrite_attribution.csv": fusion_rows,
        "contribution_reconciliation.csv": contribution_rows,
    }
    for filename, rows in csv_specs.items():
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(summary["counts"], sort_keys=True))
    print(json.dumps(summary["contribution_reconciliation"], sort_keys=True))


if __name__ == "__main__":
    main()
