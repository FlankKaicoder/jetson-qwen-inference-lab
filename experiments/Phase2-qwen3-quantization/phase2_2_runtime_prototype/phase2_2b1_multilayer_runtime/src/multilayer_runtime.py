"""Bounded multi-layer TensorRT scheduler for Phase 2.2-B1.

The scheduler owns ordering, stream use and per-layer cache state. TensorRT
owns only one layer invocation; all four logical layers reuse the same engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import torch

_SINGLE_LAYER = Path(__file__).resolve().parents[2] / "src" / "trt_single_layer"
if str(_SINGLE_LAYER) not in sys.path:
    sys.path.insert(0, str(_SINGLE_LAYER))
from trt_runtime import TRTRuntime  # noqa: E402


@dataclass
class LayerState:
    layer_id: int
    past_k: torch.Tensor | None = None
    past_v: torch.Tensor | None = None
    current_position: int = 0


class MultiLayerRuntime:
    """Runs a fixed number of logical layers with isolated KV ownership."""

    def __init__(self, prefill_engine: Path, decode_engine: Path, num_layers: int = 4):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.num_layers = num_layers
        self.layers = [LayerState(i) for i in range(num_layers)]
        self.prefill_engines = [TRTRuntime(prefill_engine) for _ in self.layers]
        self.decode_engines = [TRTRuntime(decode_engine) for _ in self.layers]
        self.stream = torch.cuda.current_stream()

    def prefill(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
        """Schedule all layers for [B,S], creating an independent cache per layer."""
        hidden = hidden_states
        records = []
        for state, engine in zip(self.layers, self.prefill_engines):
            if state.current_position != 0:
                raise RuntimeError("prefill requires reset layer state")
            out = engine.execute({"hidden_states": hidden, "position_ids": position_ids})
            state.past_k, state.past_v = out["present_k"], out["present_v"]
            state.current_position = int(state.past_k.shape[2])
            hidden = out["hidden_out"]
            records.append(out)
        return hidden, records

    def decode(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
        """Schedule one token through every layer and append each layer's cache."""
        hidden = hidden_states
        records = []
        expected_position = self.layers[0].current_position
        for state, engine in zip(self.layers, self.decode_engines):
            if state.past_k is None or state.past_v is None:
                raise RuntimeError(f"layer {state.layer_id} has no prefill cache")
            if state.current_position != expected_position:
                raise RuntimeError("layer cache positions are not synchronized")
            if int(position_ids[0, 0].item()) != state.current_position:
                raise ValueError("position_ids must equal the visible cache length")
            out = engine.execute({
                "hidden_states": hidden,
                "position_ids": position_ids,
                "past_k": state.past_k,
                "past_v": state.past_v,
            })
            state.past_k, state.past_v = out["present_k"], out["present_v"]
            state.current_position = int(state.past_k.shape[2])
            hidden = out["hidden_out"]
            records.append(out)
        return hidden, records

    def cache_bytes(self) -> int:
        """Actual tensor allocation bytes for all current layer K/V tensors."""
        return sum(
            t.numel() * t.element_size()
            for s in self.layers
            for t in (s.past_k, s.past_v)
            if t is not None
        )
