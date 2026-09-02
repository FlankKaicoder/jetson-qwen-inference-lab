from __future__ import annotations

import sys
from pathlib import Path

_b3 = Path(__file__).resolve().parents[1] / 'phase2_2b3_real_stack'
if str(_b3) not in sys.path:
    sys.path.insert(0, str(_b3))

from portable_qwen3_stack import PortableQwen3Layer
import torch
from torch import nn


class PortableTwentyEightLayerStack(nn.Module):
    """Real Qwen3 decoder-only stack; embedding/head remain outside this scope."""

    def __init__(self, device='cuda', dtype=torch.float16):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(28):
            # Materialize only one temporary FP32 layer before converting it.
            layers.append(PortableQwen3Layer().to(device=device, dtype=dtype))
        self.layers = layers

    def forward_prefill(self, hidden, pos):
        hs, ks, vs = [], [], []
        for layer in self.layers:
            hidden, k, v, _ = layer.forward_prefill(hidden, pos)
            hs.append(hidden); ks.append(k); vs.append(v)
        return hs, ks, vs

    def forward_decode(self, hidden, pos, past_ks, past_vs):
        hs, ks, vs = [], [], []
        for i, layer in enumerate(self.layers):
            hidden, k, v, _ = layer.forward_decode(hidden, pos, past_ks[i], past_vs[i])
            hs.append(hidden); ks.append(k); vs.append(v)
        return hs, ks, vs
