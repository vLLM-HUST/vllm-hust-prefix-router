"""Framework-neutral prefix-cache event types."""

from dataclasses import dataclass
from typing import TypeAlias

BlockHash: TypeAlias = bytes


@dataclass(frozen=True)
class BlockStored:
    """Add block hashes to one cache group."""

    block_hashes: tuple[BlockHash, ...]
    block_size: int
    group_idx: int = 0


@dataclass(frozen=True)
class BlockRemoved:
    """Remove block hashes from one cache group."""

    block_hashes: tuple[BlockHash, ...]
    group_idx: int = 0


@dataclass(frozen=True)
class AllBlocksCleared:
    """Invalidate every cached block for a node and rank."""


CacheEvent: TypeAlias = BlockStored | BlockRemoved | AllBlocksCleared


@dataclass(frozen=True)
class CacheEventBatch:
    """A cache event batch normalized by a framework adapter."""

    events: tuple[CacheEvent, ...]
    data_parallel_rank: int | None = None


@dataclass(frozen=True)
class CacheReplayResponse:
    """Normalized replay response; batch payloads remain codec-owned bytes."""

    publisher_epoch: str
    next_seq: int
    replayed_batches: tuple[tuple[int, bytes], ...]
    snapshot: bytes | None = None
