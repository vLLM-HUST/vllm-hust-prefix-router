"""Framework-neutral event gap and replay reconciliation state machine."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Protocol

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
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
        snapshot_replay: bool = True,
    ) -> None:
        self.node_id = node_id
        self.data_parallel_rank = data_parallel_rank
        self.index = index
        self.codec = codec
        self.replay = replay
        self.snapshot_replay = snapshot_replay
        self.state = RecoveryState()
        self._lock = asyncio.Lock()
        self._retired_epochs: set[str] = set()
        self._needs_snapshot = True
        self._rank_bound = False
        self._observed_rank: int | None = data_parallel_rank
        self._last_live_seq: int | None = None

    async def initialize(self) -> None:
        """Rebuild from a snapshot, or explicitly opted-in partial history."""
        async with self._lock:
            self._needs_snapshot = True
            if not self.snapshot_replay:
                self.state = RecoveryState()
                self._last_live_seq = None
            self.index.invalidate_node(self.node_id, self.data_parallel_rank)
            await self._replay_locked(force_snapshot=self.snapshot_replay)

    async def synchronize(self) -> None:
        """Serialize periodic reconciliation with live gap recovery."""
        async with self._lock:
            await self._replay_locked(
                force_snapshot=self._needs_snapshot and self.snapshot_replay
            )

    async def _replay_locked(self, *, force_snapshot: bool) -> None:
        try:
            # Transport receives a value snapshot, never our mutable watermark.
            response = await self.replay(replace(self.state), force_snapshot)
            self.apply_replay(response, require_snapshot=force_snapshot)
        except BaseException:
            self.index.invalidate_node(
                self.node_id, self.data_parallel_rank, clear=False
            )
            raise

    async def apply_live(self, sequence: int, payload: bytes) -> None:
        """Apply one live batch, recovering any missing interval first."""
        if type(sequence) is not int or sequence < 0:
            raise ValueError("event sequence must be non-negative")
        async with self._lock:
            try:
                if (
                    not self.snapshot_replay
                    and self._last_live_seq is not None
                    and sequence < self._last_live_seq
                ):
                    # This is only a restart heuristic, never publisher identity.
                    self.state = RecoveryState()
                    self._needs_snapshot = True
                    self._last_live_seq = None
                    self.index.invalidate_node(self.node_id, self.data_parallel_rank)
                if (
                    not self._needs_snapshot
                    and self.state.next_seq is not None
                    and sequence < self.state.next_seq
                ):
                    return  # Already covered by snapshot/replay, not a new delta.
                needed_snapshot = self._needs_snapshot
                if needed_snapshot or sequence != self.state.next_seq:
                    self.index.invalidate_node(
                        self.node_id, self.data_parallel_rank, clear=False
                    )
                    previous_epoch = self.state.publisher_epoch
                    await self._replay_locked(
                        force_snapshot=needed_snapshot and self.snapshot_replay
                    )
                    if (
                        self.snapshot_replay
                        and previous_epoch != self.state.publisher_epoch
                    ):
                        # An untagged queued frame cannot be assigned to the new epoch.
                        return
                if self.state.next_seq is not None and sequence < self.state.next_seq:
                    return  # Replay may already include the triggering live batch.
                if sequence != self.state.next_seq:
                    raise ValueError(
                        f"live event gap remains after replay: expected "
                        f"{self.state.next_seq}, got {sequence}"
                    )
                batch = self._decode_payload(payload)
                try:
                    self.index.apply_event_batch(self.node_id, batch)
                except Exception:
                    # A later event may reject a layout after an earlier delta
                    # changed the index. Never recover that partial base through
                    # an empty heartbeat. Rebuild only on this exceptional path;
                    # normal live updates do not copy the full cache index.
                    self._needs_snapshot = True
                    self.index.invalidate_node(self.node_id, self.data_parallel_rank)
                    if not self.snapshot_replay:
                        self.state = RecoveryState()
                        self._last_live_seq = None
                    raise
                self.state.next_seq = sequence + 1
                self._last_live_seq = sequence
            except BaseException:
                self.index.invalidate_node(
                    self.node_id, self.data_parallel_rank, clear=False
                )
                raise

    def apply_replay(
        self, response: CacheReplayResponse, *, require_snapshot: bool = False
    ) -> None:
        """Validate and commit a response; transport callers use synchronize()."""
        try:
            self._apply_replay(response, require_snapshot=require_snapshot)
        except Exception:
            self.index.invalidate_node(
                self.node_id, self.data_parallel_rank, clear=False
            )
            raise

    def _apply_replay(
        self, response: CacheReplayResponse, *, require_snapshot: bool
    ) -> None:
        if not response.publisher_epoch:
            raise ValueError("publisher epoch must be non-empty")
        if type(response.next_seq) is not int or response.next_seq < 0:
            raise ValueError("replay next_seq must be non-negative")
        if response.publisher_epoch in self._retired_epochs:
            raise ValueError("replay belongs to a retired publisher generation")
        epoch_changed = (
            self.state.publisher_epoch is not None
            and response.publisher_epoch != self.state.publisher_epoch
        )
        if (
            self.snapshot_replay
            and (require_snapshot or self._needs_snapshot or epoch_changed)
            and response.snapshot is None
        ):
            raise ValueError("publisher generation change requires a full snapshot")
        if (
            not epoch_changed
            and self.state.next_seq is not None
            and response.next_seq < self.state.next_seq
        ):
            raise ValueError("stale replay would roll back the sequence watermark")

        decoded: list[CacheEventBatch] = []
        replace_cache = response.snapshot is not None or self._needs_snapshot
        if response.snapshot is not None:
            if response.replayed_batches:
                raise ValueError("snapshot replay response must not mix delta batches")
            decoded.append(self._decode_payload(response.snapshot))
            expected = response.next_seq
        else:
            expected = self.state.next_seq
            if not self.snapshot_replay and response.replayed_batches:
                first = response.replayed_batches[0][0]
                if expected is None or first > expected:
                    # A truncated replay cannot reconstruct old cache entries.
                    # Rebuild a conservative lower-bound view from retained deltas.
                    replace_cache = True
                    expected = first
            if not self.snapshot_replay and expected is None:
                expected = response.next_seq
            if expected is None:
                raise ValueError("replay without a snapshot has no starting sequence")

        for sequence, payload in response.replayed_batches:
            if type(sequence) is not int or sequence != expected:
                raise ValueError(
                    f"non-contiguous replay: expected {expected}, got {sequence}"
                )
            decoded.append(self._decode_payload(payload))
            expected += 1
        if expected != response.next_seq:
            raise ValueError(
                f"replay next_seq mismatch: decoded {expected}, "
                f"reported {response.next_seq}"
            )
        ranks = {batch.data_parallel_rank for batch in decoded}
        if len(ranks) > 1:
            raise ValueError("one replay response must describe one backend rank")
        recovered_rank = (
            decoded[0].data_parallel_rank if decoded else self._observed_rank
        )
        self.index.apply_recovery_batches(
            self.node_id,
            decoded,
            data_parallel_rank=recovered_rank,
            replace_cache=replace_cache,
            replace_layout=response.snapshot is not None,
        )
        if epoch_changed:
            self._retired_epochs.add(self.state.publisher_epoch)
        self.state.publisher_epoch = response.publisher_epoch
        self.state.next_seq = response.next_seq
        self._needs_snapshot = False
        self._observed_rank = recovered_rank
        self._rank_bound = True

    def _decode_payload(self, payload: bytes) -> CacheEventBatch:
        batch = self.codec.decode_batch(payload)
        rank = self._observed_rank if self._rank_bound else self.data_parallel_rank
        if (self._rank_bound or rank is not None) and batch.data_parallel_rank not in (
            None,
            rank,
        ):
            raise ValueError("KV event rank does not match the configured backend")
        if batch.data_parallel_rank is None and rank is not None:
            batch = replace(batch, data_parallel_rank=rank)
        for event in batch.events:
            if not isinstance(event, (BlockStored, BlockRemoved, AllBlocksCleared)):
                raise ValueError("unsupported normalized cache event")
            if isinstance(event, (BlockStored, BlockRemoved)):
                if type(event.group_idx) is not int or event.group_idx < 0:
                    raise ValueError("cache group index must be non-negative")
                if any(not isinstance(value, bytes) for value in event.block_hashes):
                    raise ValueError("normalized block hashes must be bytes")
            if isinstance(event, BlockStored) and (
                type(event.block_size) is not int or event.block_size <= 0
            ):
                raise ValueError("cache block size must be positive")
        return batch
