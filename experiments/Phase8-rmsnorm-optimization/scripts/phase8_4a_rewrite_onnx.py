#!/usr/bin/env python3
"""Create isolated Phase 8.4-A ONNX copies with one Layer-0 RMSNorm variant."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


TARGET_PREFIX = "/input_layernorm/"
TARGET_OUTPUT = "/input_layernorm/Mul_1_output_0"
GAMMA = "stack.layers.0.input_layernorm.weight"


def expose_node_output(model: onnx.ModelProto) -> None:
    if any(x.name == TARGET_OUTPUT for x in model.graph.output):
        return
    value = helper.make_tensor_value_info(TARGET_OUTPUT, TensorProto.FLOAT16, ["batch", "seq", 1024])
    model.graph.output.append(value)


def rewrite(src: Path, dst: Path, variant: str) -> dict[str, object]:
    model = onnx.load(str(src), load_external_data=False)
    nodes = list(model.graph.node)
    target = [n for n in nodes if n.name.startswith(TARGET_PREFIX)]
    if variant == "baseline":
        expose_node_output(model)
    else:
        target_ids = {id(n) for n in nodes if n.name.startswith(TARGET_PREFIX)}
        target_outputs = {
            output for n in nodes if id(n) in target_ids for output in n.output
        }
        external_consumers = {
            inp
            for n in nodes
            if id(n) not in target_ids
            for inp in n.input
            if inp in target_outputs
        }
        # Keep target constants that are also consumed by unrelated graph nodes.
        kept = [
            n for n in nodes
            if id(n) not in target_ids
            or any(output in external_consumers and output != TARGET_OUTPUT for output in n.output)
        ]
        plugin = helper.make_node(
            "RMSNormPlugin", ["hidden_states", GAMMA], [TARGET_OUTPUT],
            name="/input_layernorm/RMSNormPlugin", domain="trt.plugins",
            epsilon=1.0e-6,
        )
        insert_at = min(
            i for i, n in enumerate(kept)
            if n.name.startswith(TARGET_PREFIX) and id(n) not in target_ids
        ) if any(n.name.startswith(TARGET_PREFIX) and id(n) not in target_ids for n in kept) else 0
        kept.insert(insert_at, plugin)
        del model.graph.node[:]
        model.graph.node.extend(kept)
        # ONNX requires an opset declaration for every non-default operator domain.
        if not any(op.domain == "trt.plugins" for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid("trt.plugins", 1))
        expose_node_output(model)
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return {
        "source": str(src), "output": str(dst), "variant": variant,
        "original_target_nodes": len(target),
        "final_node_count": len(model.graph.node),
        "plugin_nodes": sum(n.op_type == "RMSNormPlugin" for n in model.graph.node),
        "exposed_output": TARGET_OUTPUT,
        "bytes": dst.stat().st_size,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--plugin", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    result = {
        "baseline": rewrite(a.source, a.baseline, "baseline"),
        "plugin": rewrite(a.source, a.plugin, "plugin"),
    }
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
