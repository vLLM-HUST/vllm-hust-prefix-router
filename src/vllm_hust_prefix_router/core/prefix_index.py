# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-neutral prefix-cache index and longest-prefix router."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import count

from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    BlockHash,
    BlockRemoved,
    BlockStored,
    CacheEvent,
    CacheEventBatch,
)


@dataclass(frozen=True)
class PrefixRouteDecision:
    """One node's prefix score or the selected routing target."""

    node_id: str
    matched_tokens: int
    data_parallel_rank: int | None = None
    decision_id: str | None = None
    cache_size_blocks: int = 0


@dataclass(frozen=True)
class PrefixCacheSnapshot:
    """A complete cache view for one node and data-parallel rank."""

    node_id: str
    hash_block_size: int
    group_block_sizes: dict[int, int]
    group_hashes: dict[int, set[BlockHash]]
    data_parallel_rank: int | None = None


@dataclass
class NodePrefixCacheState:
    """In-memory prefix-cache state for one node and rank."""

    node_id: str
    hash_block_size: int
    data_parallel_rank: int | None = None
    group_block_sizes: dict[int, int] = field(default_factory=dict)
    group_hashes: dict[int, set[BlockHash]] = field(
        default_factory=lambda: defaultdict(set)
    )
    trusted: bool = True

    @property
    def cache_size_blocks(self) -> int:
        """Return the number of distinct logical blocks in the cache index."""
        if not self.group_hashes:
            return 0
        return len(set().union(*self.group_hashes.values()))

    @classmethod
    def from_snapshot(cls, snapshot: PrefixCacheSnapshot) -> "NodePrefixCacheState":
        """Create trusted state from a full snapshot."""
        return cls(
            node_id=snapshot.node_id,
            data_parallel_rank=snapshot.data_parallel_rank,
            hash_block_size=snapshot.hash_block_size,
            group_block_sizes=dict(snapshot.group_block_sizes),
            group_hashes=defaultdict(
                set,
                {
                    group_id: set(hashes)
                    for group_id, hashes in snapshot.group_hashes.items()
                },
            ),
            trusted=True,
        )

    def apply_snapshot(self, snapshot: PrefixCacheSnapshot) -> None:
        """Atomically replace this node's cache state."""
        if snapshot.node_id != self.node_id:
            raise ValueError(
                f"snapshot for node {snapshot.node_id!r} cannot update "
                f"state for node {self.node_id!r}"
            )
        self.data_parallel_rank = snapshot.data_parallel_rank
        self.hash_block_size = snapshot.hash_block_size
        self.group_block_sizes = dict(snapshot.group_block_sizes)
        self.group_hashes = defaultdict(
            set,
            {
                group_id: set(hashes)
                for group_id, hashes in snapshot.group_hashes.items()
            },
        )
        self.trusted = True

    def apply_events(self, events: Iterable[CacheEvent]) -> None:
        """Apply normalized cache deltas."""
        for event in events:
            if isinstance(event, BlockStored):
                self.group_block_sizes[event.group_idx] = event.block_size
                self.group_hashes[event.group_idx].update(event.block_hashes)
            elif isinstance(event, BlockRemoved):
                hashes = self.group_hashes.get(event.group_idx)
                if hashes is not None:
                    hashes.difference_update(event.block_hashes)
            elif isinstance(event, AllBlocksCleared):
                self.group_hashes.clear()

    def invalidate(self) -> None:
        """Stop using cached-prefix claims until a full recovery completes."""
        self.group_hashes.clear()
        self.trusted = False

    def longest_prefix_match(
        self,
        block_hashes: Sequence[BlockHash],
        prompt_num_tokens: int,
        max_cache_hit_length: int | None = None,
    ) -> int:
        """Return the longest trusted cached prefix in tokens."""
        if not self.trusted or not block_hashes or not self.group_hashes:
            return 0

        max_length = prompt_num_tokens - 1
        if max_cache_hit_length is not None:
            max_length = min(max_length, max_cache_hit_length)
        if max_length <= 0:
            return 0

        group_block_sizes = self.group_block_sizes or {
            group_id: self.hash_block_size for group_id in self.group_hashes
        }
        group_hits = [
            self._longest_group_match(
                block_hashes=block_hashes,
                hashes=self.group_hashes.get(group_id, set()),
                block_size=block_size,
                max_cache_hit_length=max_length,
            )
            for group_id, block_size in group_block_sizes.items()
        ]
        return min(group_hits, default=0)

    def _longest_group_match(
        self,
        block_hashes: Sequence[BlockHash],
        hashes: set[BlockHash],
        block_size: int,
        max_cache_hit_length: int,
    ) -> int:
        if block_size <= 0 or block_size % self.hash_block_size != 0:
            return 0

        scale = block_size // self.hash_block_size
        max_blocks = min(
            max_cache_hit_length // block_size,
            len(block_hashes) // scale,
        )
        matched_blocks = 0
        for block_idx in range(max_blocks):
            hash_idx = (block_idx + 1) * scale - 1
            if block_hashes[hash_idx] not in hashes:
                break
            matched_blocks += 1
        return matched_blocks * block_size


class GlobalPrefixIndex:
    """Longest-prefix-first index for a set of remote inference nodes."""

    def __init__(self) -> None:
        self._nodes: dict[tuple[str, int | None], NodePrefixCacheState] = {}
        self._node_defaults: dict[str, tuple[int, int | None, dict[int, int]]] = {}
        self._tie_breaker = count()

    def register_node(
        self,
        node_id: str,
        *,
        hash_block_size: int,
        data_parallel_rank: int | None = None,
        group_block_sizes: Mapping[int, int] | None = None,
    ) -> NodePrefixCacheState:
        """Register one remote node."""
        if hash_block_size <= 0:
            raise ValueError("hash_block_size must be positive")
        sizes = dict(group_block_sizes or {})
        state = NodePrefixCacheState(
            node_id=node_id,
            hash_block_size=hash_block_size,
            data_parallel_rank=data_parallel_rank,
            group_block_sizes=sizes,
        )
        self._node_defaults[node_id] = (
            hash_block_size,
            data_parallel_rank,
            sizes,
        )
        self._nodes[node_id, data_parallel_rank] = state
        return state

    def update_snapshot(self, snapshot: PrefixCacheSnapshot) -> None:
        """Replace one node/rank state with a recovered snapshot."""
        defaults = self._node_defaults.get(snapshot.node_id)
        if defaults is not None:
            _, configured_rank, _ = defaults
            if (
                snapshot.data_parallel_rank is not None
                and configured_rank is not None
                and snapshot.data_parallel_rank != configured_rank
            ):
                raise ValueError(
                    f"node {snapshot.node_id!r} is configured for rank "
                    f"{configured_rank}, but reported {snapshot.data_parallel_rank}"
                )
            if snapshot.data_parallel_rank is None and configured_rank is not None:
                snapshot = replace(snapshot, data_parallel_rank=configured_rank)
        self._node_defaults.setdefault(
            snapshot.node_id,
            (snapshot.hash_block_size, None, dict(snapshot.group_block_sizes)),
        )
        key = (snapshot.node_id, snapshot.data_parallel_rank)
        state = self._nodes.get(key)
        if state is None:
            self._discard_unranked_placeholder(snapshot.node_id)
            self._nodes[key] = NodePrefixCacheState.from_snapshot(snapshot)
        else:
            state.apply_snapshot(snapshot)

    def apply_events(
        self,
        node_id: str,
        events: Iterable[CacheEvent],
        data_parallel_rank: int | None = None,
    ) -> None:
        """Apply normalized incremental events."""
        self._state_for_events(node_id, data_parallel_rank).apply_events(events)

    def apply_event_batch(self, node_id: str, batch: CacheEventBatch) -> None:
        """Apply a normalized event batch."""
        self.apply_events(node_id, batch.events, batch.data_parallel_rank)

    def invalidate_node(
        self, node_id: str, data_parallel_rank: int | None = None
    ) -> None:
        """Invalidate prefix claims after an event gap or publisher restart."""
        matched = False
        for (candidate_id, rank), state in self._nodes.items():
            if candidate_id == node_id and (
                data_parallel_rank is None or rank == data_parallel_rank
            ):
                state.invalidate()
                matched = True
        if not matched:
            self._state_for_events(node_id, data_parallel_rank).invalidate()

    def trust_node(
        self, node_id: str, data_parallel_rank: int | None = None
    ) -> None:
        """Mark a recovered node state usable for prefix matching."""
        state = self._state_for_events(node_id, data_parallel_rank)
        state.trusted = True

    def remove_node(self, node_id: str) -> None:
        """Remove all ranks belonging to a node."""
        self._node_defaults.pop(node_id, None)
        for key in [key for key in self._nodes if key[0] == node_id]:
            del self._nodes[key]

    def _state_for_events(
        self, node_id: str, data_parallel_rank: int | None
    ) -> NodePrefixCacheState:
        try:
            hash_block_size, configured_rank, group_block_sizes = self._node_defaults[
                node_id
            ]
        except KeyError as exc:
            raise KeyError(f"unknown prefix routing node {node_id!r}") from exc

        if data_parallel_rank is None:
            data_parallel_rank = configured_rank
        elif configured_rank is not None and data_parallel_rank != configured_rank:
            raise ValueError(
                f"node {node_id!r} is configured for rank {configured_rank}, "
                f"but reported {data_parallel_rank}"
            )

        key = (node_id, data_parallel_rank)
        state = self._nodes.get(key)
        if state is None:
            self._discard_unranked_placeholder(node_id)
            state = NodePrefixCacheState(
                node_id=node_id,
                hash_block_size=hash_block_size,
                data_parallel_rank=data_parallel_rank,
                group_block_sizes=dict(group_block_sizes),
            )
            self._nodes[key] = state
        return state

    def _discard_unranked_placeholder(self, node_id: str) -> None:
        placeholder = self._nodes.get((node_id, None))
        if placeholder is not None and not placeholder.group_hashes:
            del self._nodes[node_id, None]

    def score_nodes(
        self,
        block_hashes: Sequence[BlockHash],
        prompt_num_tokens: int,
        *,
        candidate_node_ids: Iterable[str] | None = None,
        max_cache_hit_length: int | None = None,
    ) -> list[PrefixRouteDecision]:
        """Return prefix scores for every eligible node and rank."""
        node_ids = set(candidate_node_ids) if candidate_node_ids is not None else None
        states = (
            [
                state
                for (node_id, _), state in self._nodes.items()
                if node_id in node_ids
            ]
            if node_ids is not None
            else list(self._nodes.values())
        )
        return [
            PrefixRouteDecision(
                node_id=state.node_id,
                data_parallel_rank=state.data_parallel_rank,
                matched_tokens=state.longest_prefix_match(
                    block_hashes,
                    prompt_num_tokens,
                    max_cache_hit_length,
                ),
                cache_size_blocks=state.cache_size_blocks,
            )
            for state in states
        ]

    def choose_scored_node(
        self,
        scores: Iterable[PrefixRouteDecision],
        *,
        node_loads: Mapping[str, int] | None = None,
    ) -> PrefixRouteDecision | None:
        """Choose a node from precomputed prefix scores."""
        candidates = list(scores)
        if not candidates:
            return None
        best_match = max(decision.matched_tokens for decision in candidates)
        tied = [
            decision
            for decision in candidates
            if decision.matched_tokens == best_match
        ]
        if node_loads is not None and all(
            decision.node_id in node_loads for decision in tied
        ):
            best_load = min(node_loads[decision.node_id] for decision in tied)
            tied = [
                decision
                for decision in tied
                if node_loads[decision.node_id] == best_load
            ]
        return tied[next(self._tie_breaker) % len(tied)]

    def choose_node(
        self,
        block_hashes: Sequence[BlockHash],
        prompt_num_tokens: int,
        *,
        candidate_node_ids: Iterable[str] | None = None,
        max_cache_hit_length: int | None = None,
        node_loads: Mapping[str, int] | None = None,
    ) -> PrefixRouteDecision | None:
        """Choose the longest-prefix node, then load and round-robin tie-break."""
        return self.choose_scored_node(
            self.score_nodes(
                block_hashes,
                prompt_num_tokens,
                candidate_node_ids=candidate_node_ids,
                max_cache_hit_length=max_cache_hit_length,
            ),
            node_loads=node_loads,
        )
