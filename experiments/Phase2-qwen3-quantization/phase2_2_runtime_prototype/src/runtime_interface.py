"""Type-level prefill/decode interface; it deliberately performs no inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from kv_cache_manager import KVCacheManager


TokenRows = Sequence[Sequence[int]]


def _validate_token_rows(input_ids: TokenRows, *, require_single_token: bool) -> None:
    if not input_ids:
        raise ValueError("input_ids must contain at least one batch row")
    length = len(input_ids[0])
    if length == 0 or (require_single_token and length != 1):
        raise ValueError("decode requires [B,1]; prefill requires a nonempty [B,S]")
    if any(len(row) != length for row in input_ids):
        raise ValueError("input_ids rows must form a rectangular [B,S] matrix")


@dataclass(frozen=True)
class PrefillRequest:
    """Prompt input [B,S]; future execution creates K/V for all S positions."""

    input_ids: TokenRows
    attention_mask: TokenRows | None = None
    position_ids: TokenRows | None = None

    def __post_init__(self) -> None:
        _validate_token_rows(self.input_ids, require_single_token=False)


@dataclass(frozen=True)
class DecodeRequest:
    """New tokens [B,1]; future execution reads cache then appends one new K/V."""

    input_ids: TokenRows
    position_ids: TokenRows | None = None

    def __post_init__(self) -> None:
        _validate_token_rows(self.input_ids, require_single_token=True)


@dataclass
class KVCacheState:
    """Runtime-visible state: the manager owns memory, state exposes its position."""

    manager: KVCacheManager
    sequence_id: str

    @property
    def sequence_length(self) -> int:
        return self.manager.sequence_length

    def validate_prefill(self, request: PrefillRequest) -> None:
        if len(request.input_ids) != self.manager.batch_size:
            raise ValueError("prefill batch size must match allocated KV cache")
        if len(request.input_ids[0]) > self.manager.capacity_tokens:
            raise ValueError("prefill length exceeds allocated KV cache")

    def validate_decode(self, request: DecodeRequest) -> int:
        if len(request.input_ids) != self.manager.batch_size:
            raise ValueError("decode batch size must match allocated KV cache")
        if self.sequence_length >= self.manager.capacity_tokens:
            raise ValueError("decode would exceed KV cache capacity")
        return self.sequence_length
