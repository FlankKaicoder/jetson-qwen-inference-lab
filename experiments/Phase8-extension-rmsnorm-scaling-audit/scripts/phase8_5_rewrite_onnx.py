#!/usr/bin/env python3
"""Rewrite N selected Qwen3 RMSNorm subgraphs with the Phase 8.4-A plugin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def expose_output(model: onnx.ModelProto, name: str) -> None:
    if any(value.name == name for value in model.graph.output):
        return
    value = helper.make_tensor_value_info(
        name, TensorProto.FLOAT16, ["batch", "seq", 1024])
    model.graph.output.append(value)


def norm_group(model: onnx.ModelProto, gamma: str) -> tuple[list[int], str, str, str]:
    initializers = {value.name for value in model.graph.initializer}
    nodes = list(model.graph.node)
    consumers = [node for node in nodes if gamma in node.input]
    candidates = [node for node in consumers if node.op_type == "Mul"]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one gamma multiply for {gamma}, got {len(candidates)}")
    final_index = next(index for index, node in enumerate(nodes)
                       if node.op_type == "Mul" and gamma in node.input)
    final = nodes[final_index]
    target_output = final.output[0]
    slash = final.name.rfind("/")
    if slash < 0:
        raise RuntimeError(f"cannot infer RMSNorm module prefix from {final.name}")
    prefix = final.name[:slash + 1]
    indices = [index for index, node in enumerate(nodes)
               if node.name.startswith(prefix)]
    if not indices:
        raise RuntimeError(f"empty RMSNorm group for {prefix}")
    entry = nodes[indices[0]]
    prefix_outputs = {name for index in indices for name in nodes[index].output}
    hidden_candidates = [
        name for name in entry.input
        if name not in prefix_outputs and name not in initializers
    ]
    if len(hidden_candidates) != 1:
        raise RuntimeError(f"expected one hidden input for {prefix}, got {hidden_candidates}")
    return indices, final.input[0] if len(final.input) < 2 else "", target_output, hidden_candidates[0]


def external_consumers(model: onnx.ModelProto, indices: list[int]) -> dict[str, int]:
    selected = {model.graph.node[index].output[0] for index in indices}
    result = {}
    for index, node in enumerate(model.graph.node):
        if index in indices:
            continue
        for name in node.input:
            if name in selected:
                result[name] = result.get(name, 0) + 1
    for output in model.graph.output:
        if output.name in selected:
            result[output.name] = result.get(output.name, 0) + 1
    return result


def rewrite(source: Path, output: Path, kind: str, gammas: list[str],
            expose_only: bool) -> dict:
    model = onnx.load(str(source), load_external_data=False)
    replacements = []
    removed_total = 0
    if not expose_only:
        for gamma in gammas:
            indices, _unused, target_output, hidden_input = norm_group(model, gamma)
            plugin_name = f"/{gamma}/RMSNormPlugin"
            plugin = helper.make_node(
                "RMSNormPlugin", [hidden_input, gamma], [target_output],
                name=plugin_name, domain="trt.plugins", epsilon=1.0e-6,
            )
            expose_output(model, target_output)
            external = external_consumers(model, indices)
            keep = [node for index, node in enumerate(model.graph.node)
                    if index not in indices or any(
                        name in external and name != target_output
                        for name in node.output)]
            producer_position = -1
            for position, node in enumerate(keep):
                if hidden_input in node.output:
                    producer_position = position
                    break
            insert_at = producer_position + 1 if producer_position >= 0 else 0
            insert_at = min(insert_at, len(keep))
            keep.insert(insert_at, plugin)
            del model.graph.node[:]
            model.graph.node.extend(keep)
            if not any(op.domain == "trt.plugins" for op in model.opset_import):
                model.opset_import.append(helper.make_opsetid("trt.plugins", 1))
            removed = len(indices) - sum(
                any(name in external and name != target_output for name in node.output)
                for node in model.graph.node)
            replacements.append({
                "gamma": gamma, "target_output": target_output,
                "hidden_input": hidden_input, "target_nodes": len(indices),
                "removed_nodes": removed, "plugin_name": plugin_name,
            })
            removed_total += removed
    else:
        for gamma in gammas:
            _indices, _unused, target_output, _hidden = norm_group(model, gamma)
            replacements.append({"gamma": gamma, "target_output": target_output})
            expose_output(model, target_output)

    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return {
        "source": str(source), "output": str(output), "kind": kind,
        "expose_only": expose_only, "gamma_count": len(gammas),
        "plugin_nodes": sum(node.op_type == "RMSNormPlugin" for node in model.graph.node),
        "removed_nodes": removed_total,
        "final_node_count": len(model.graph.node),
        "bytes": output.stat().st_size,
        "replacements": replacements,
    }


def producers_output_index(model: onnx.ModelProto) -> dict[str, int]:
    return {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["prefill", "decode"], required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    gammas = [f"stack.layers.{i}.input_layernorm.weight" for i in range(min(args.scale, 28))]
    if args.scale == 56:
        gammas += [f"stack.layers.{i}.post_attention_layernorm.weight" for i in range(28)]
    result = {
        "baseline": rewrite(args.source, args.output.parent / f"{args.kind}_baseline_scale{args.scale}.onnx",
                            args.kind, gammas, expose_only=True),
        "plugin": rewrite(args.source, args.output, args.kind, gammas,
                          expose_only=args.scale == 0),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
