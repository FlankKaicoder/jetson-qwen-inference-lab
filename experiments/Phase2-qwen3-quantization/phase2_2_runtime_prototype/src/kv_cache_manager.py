"""CPU-only ownership model for a Qwen3 decoder KV cache.

This module intentionally uses bytearray rather than CUDA or TensorRT APIs. It
defines the persistent-cache contract a future runtime must satisfy without
claiming that the layout is an engine binding or a production allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CacheDType = Literal["fp16", "bf16"]


@dataclass(frozen=True)
class KVCacheConfig:
    """Qwen3-0.6B cache dimensions; K and V are stored separately per layer."""

    num_layers: int = 28
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: CacheDType = "fp16"

    @property
    def dtype_size(self) -> int:
        # FP16 and BF16 each use one 16-bit element. No numeric conversion occurs here.
        return 2


@dataclass
class LayerCache:
    """CPU-owned storage with layout [batch, kv_head, sequence, head_dim]."""

    key: bytearray
    value: bytearray
    length: int = 0


class KVCacheManager:
    """Owns a single sequence's persistent K/V storage and logical position.

    The manager owns the bytearrays. A future TensorRT runtime would instead own
    device allocations and bind their addresses for each prefill/decode enqueue.
    `sequence_length` is the shortest contiguous prefix written by every layer;
    this prevents a partially appended decoder step from becoming visible.
    """

    def __init__(self, config: KVCacheConfig | None = None) -> None:
        self.config = config or KVCacheConfig()
        self.batch_size = 0
        self.capacity_tokens = 0
        self._layers: list[LayerCache] = []

    def allocate_cache(self, batch_size: int, capacity_tokens: int) -> None:
        """Allocate CPU buffers for all layers, invalidating any prior sequence.

        Per K or V layer allocation is
        `B * H_kv * T_capacity * D * dtype_size` bytes. K and V are separate
        contiguous buffers, which makes their ownership and engine binding explicit.
        """
        if batch_size <= 0 or capacity_tokens <= 0:
            raise ValueError("batch_size and capacity_tokens must be positive")
        per_tensor_bytes = (
            batch_size
            * self.config.num_kv_heads
            * capacity_tokens
            * self.config.head_dim
            * self.config.dtype_size
        )
        self.batch_size = batch_size
        self.capacity_tokens = capacity_tokens
        self._layers = [
            LayerCache(bytearray(per_tensor_bytes), bytearray(per_tensor_bytes))
            for _ in range(self.config.num_layers)
        ]

    @property
    def sequence_length(self) -> int:
        """Fully materialized token count shared by every decoder layer."""
        return min((layer.length for layer in self._layers), default=0)

    @property
    def bytes_allocated(self) -> int:
        return sum(len(layer.key) + len(layer.value) for layer in self._layers)

    def append_kv(
        self,
        layer_index: int,
        key: bytes | bytearray | memoryview,
        value: bytes | bytearray | memoryview,
        *,
        token_count: int = 1,
        start_position: int | None = None,
    ) -> int:
        """Append one contiguous K/V span for one layer and return its end position.

        Call each layer with the same start position during prefill/decode. A layer
        cannot skip positions, and data are copied into manager-owned memory. The
        cache position is a token position, independent of an input-id value.
        """
        layer = self.get_layer_cache(layer_index)
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        position = layer.length if start_position is None else start_position
        if position != layer.length:
            raise ValueError("append must start at the layer's contiguous tail")
        end_position = position + token_count
        if end_position > self.capacity_tokens:
            raise ValueError("append exceeds cache capacity")

        expected_bytes = (
            self.batch_size
            * self.config.num_kv_heads
            * token_count
            * self.config.head_dim
            * self.config.dtype_size
        )
        key_bytes = bytes(key)
        value_bytes = bytes(value)
        if len(key_bytes) != expected_bytes or len(value_bytes) != expected_bytes:
            raise ValueError("K and V payloads must match [B,H_kv,T,D] byte size")

        self._write_tokens(layer.key, position, token_count, key_bytes)
        self._write_tokens(layer.value, position, token_count, value_bytes)
        layer.length = end_position
        return end_position

    def get_layer_cache(self, layer_index: int) -> LayerCache:
        """Return a layer-owned cache record; callers must not replace its buffers."""
        if not self._layers:
            raise RuntimeError("allocate_cache must be called before cache access")
        if not 0 <= layer_index < self.config.num_layers:
            raise IndexError("layer_index out of range")
        return self._layers[layer_index]

    def reset_sequence(self) -> None:
        """Forget logical positions while retaining owned allocation for reuse.

        Existing bytes are not exposed because each layer length becomes zero. A
        production runtime may add secure erase separately if its threat model needs it.
        """
        for layer in self._layers:
            layer.length = 0

    def _write_tokens(
        self, storage: bytearray, position: int, token_count: int, payload: bytes
    ) -> None:
        """Scatter token-major payload [B,H,T,D] into capacity-strided storage."""
        element_bytes = self.config.head_dim * self.config.dtype_size
        payload_offset = 0
        for batch in range(self.batch_size):
            for head in range(self.config.num_kv_heads):
                base = (
                    ((batch * self.config.num_kv_heads + head) * self.capacity_tokens + position)
                    * element_bytes
                )
                count_bytes = token_count * element_bytes
                storage[base : base + count_bytes] = payload[
                    payload_offset : payload_offset + count_bytes
                ]
                payload_offset += count_bytes
