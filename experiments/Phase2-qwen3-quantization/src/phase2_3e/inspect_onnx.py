from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(args.path), load_external_data=False)
    g = model.graph

    matmul_nodes = []
    init_names = set()
    init_shape = {}
    for init in g.initializer:
        init_names.add(init.name)
        try:
            init_shape[init.name] = list(numpy_helper.to_array(init).shape)
        except Exception as exc:  # noqa: BLE001
            init_shape[init.name] = f"ERR:{type(exc).__name__}"

    for node in g.node:
        if node.op_type == "MatMul":
            matmul_nodes.append({
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "weight_is_initializer": node.input[1] in init_names,
                "weight_shape": init_shape.get(node.input[1]),
            })

    print(json.dumps({
        "graph_inputs": [x.name for x in g.input],
        "graph_outputs": [x.name for x in g.output],
        "node_count": len(g.node),
        "initializer_count": len(g.initializer),
        "matmul_count": len(matmul_nodes),
        "matmul_nodes": matmul_nodes,
    }, indent=2))


if __name__ == "__main__":
    main()
