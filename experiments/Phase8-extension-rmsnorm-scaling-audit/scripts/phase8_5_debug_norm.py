#!/usr/bin/env python3
"""Print one RMSNorm subgraph for graph-rewrite diagnosis."""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--gamma", required=True)
args = parser.parse_args()
model = onnx.load(str(args.source), load_external_data=False)
nodes = list(model.graph.node)
indices = [i for i, node in enumerate(nodes) if args.gamma in node.input]
print("gamma_consumers", indices)
for i in indices:
    node = nodes[i]
    print(i, node.name, node.op_type, list(node.input), "->", list(node.output))
selected = {nodes[i].output[0] for i in indices}
external = []
for i, node in enumerate(nodes):
    if i not in indices:
        for name in node.input:
            if name in selected:
                external.append((i, node.name, name))
print("external_consumers", external)
producers = {output: index for index, node in enumerate(nodes) for output in node.output}
initializers = {value.name for value in model.graph.initializer}
for final_index in indices:
    stack = [final_index]
    visited = set()
    roots = set()
    while stack:
        index = stack.pop()
        if index in visited:
            continue
        visited.add(index)
        node = nodes[index]
        for name in node.input:
            if name in producers:
                stack.append(producers[name])
            elif name not in initializers:
                roots.add(name)
    print("ancestors", sorted(visited), "roots", sorted(roots))
    start = max(0, final_index - 20)
    print("local_nodes")
    for index in range(start, min(len(nodes), final_index + 3)):
        node = nodes[index]
        print(index, node.name, node.op_type, list(node.input), "->", list(node.output))
prefix = "/input_layernorm/"
print("nodes_with_prefix")
for i, node in enumerate(nodes):
    if node.name.startswith(prefix):
        print(i, node.name, node.op_type, list(node.input), "->", list(node.output))
