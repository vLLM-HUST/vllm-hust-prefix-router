"""Framework-neutral event gap and replay reconciliation state machine."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.cache_events import (
    CacheEventBatch,
    CacheReplayResponse,
)


class CacheEventCodec(Protocol):
    """Decode batches produced by one compatible worker implementation."""

    def decode_batch(self, payload: bytes) -> CacheEventBatch: ...


@dataclass
class RecoveryState:
    """Publisher generation and next expected sequence number."""

    publisher_epoch: str | None = None
    next_seq: int | None = None


Replay = Callable[[RecoveryState, bool], Awaitable[CacheReplayResponse]]


class CacheEventRecovery:
    """Apply live events only when their sequence is fully reconciled."""

    def __init__(
        self,
        *,
        node_id: str,
        data_parallel_rank: int | None,
        index: GlobalPrefixIndex,
        codec: CacheEventCodec,
        replay: Replay,
    ) -> None:
        self.node_id = node_id
        self.data_parallel_rank = data_parallel_rank
        self.index = index
        self.codec = codec
        self.replay = replay
        self.state = RecoveryState()

    async def initialize(self) -> None:
        """Require a full snapshot before trusting live cache claims."""
        self.index.invalidate_node(self.node_id, self.data_parallel_rank)
        response = await self.replay(self.state, True)
        self.apply_replay(response, require_snapshot=True)

    async def apply_live(self, sequence: int, payload: bytes) -> None:
        """Apply one live batch, recovering any missing interval first."""
        if sequence < 0:
            raise ValueError("event sequence must be non-negative")
        needed_snapshot = self.state.next_seq is None
        if needed_snapshot or sequence != self.state.next_seq:
            self.index.invalidate_node(self.node_id, self.data_parallel_rank)
            response = await self.replay(self.state, needed_snapshot)
            self.apply_replay(response, require_snapshot=needed_snapshot)
        if sequence != self.state.next_seq:
            raise ValueError(
                f"live event gap remains after replay: expected "
                f"{self.state.next_seq}, got {sequence}"
            )
        self._apply_payload(payload)
        self.state.next_seq = sequence + 1

    def apply_replay(
        self, response: CacheReplayResponse, *, require_snapshot: bool = False
    ) -> None:
        """Validate an atomic replay response before restoring trust."""
        epoch_changed = (
            self.state.publisher_epoch is not None
            and response.publisher_epoch != self.state.publisher_epoch
        )
        if (require_snapshot or epoch_changed) and response.snapshot is None:
            raise ValueError("publisher generation change requires a full snapshot")

        if response.snapshot is not None:
            snapshot_batch = self.codec.decode_batch(response.snapshot)
            self.index.invalidate_node(self.node_id, self.data_parallel_rank)
            self.index.apply_event_batch(self.node_id, snapshot_batch)
            if response.replayed_batches:
                raise ValueError("snapshot replay response must not mix delta batches")
            expected = response.next_seq
        else:
            expected = self.state.next_seq
            if expected is None:
                raise ValueError("replay without a snapshot has no starting sequence")

        decoded: list[tuple[int, CacheEventBatch]] = []
        for sequence, payload in response.replayed_batches:
            if sequence != expected:
                raise ValueError(
                    f"non-contiguous replay: expected {expected}, got {sequence}"
                )
            decoded.append((sequence, self.codec.decode_batch(payload)))
            expected += 1
        if expected != response.next_seq:
            raise ValueError(
                f"replay next_seq mismatch: decoded {expected}, "
                f"reported {response.next_seq}"
            )
        for _, batch in decoded:
            self.index.apply_event_batch(self.node_id, batch)
        self.state.publisher_epoch = response.publisher_epoch
        self.state.next_seq = response.next_seq
        self.index.trust_node(self.node_id, self.data_parallel_rank)

    def _apply_payload(self, payload: bytes) -> None:
        batch = self.codec.decode_batch(payload)
        if (
            self.data_parallel_rank is not None
            and batch.data_parallel_rank not in (None, self.data_parallel_rank)
        ):
            raise ValueError("KV event rank does not match the configured backend")
        self.index.apply_event_batch(self.node_id, batch)
