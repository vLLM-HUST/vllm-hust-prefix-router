"""Compose Prefix candidate scores with request-lifecycle load accounting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

from vllm_hust_prefix_router.core.lifecycle import (
    LifecycleLoadTracker,
    PromptLoad,
    RequestLoad,
)
from vllm_hust_prefix_router.core.prefix_index import (
    GlobalPrefixIndex,
    PrefixRouteDecision,
)
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint

RoutingPolicy = Literal["prefix", "lifecycle"]


@dataclass
class _AggregatedWork:
    prompt_tokens: int = 0
    matched_tokens: int = 0
    cache_size_blocks: int = 0
    prompt_work: list[tuple[int, int]] = field(default_factory=list)


class RoutePlanner:
    """Select and reserve a remote backend using one explicit policy."""

    def __init__(
        self,
        prefix_index: GlobalPrefixIndex,
        *,
        policy: RoutingPolicy,
        block_size: int,
        lifecycle_tracker: LifecycleLoadTracker | None = None,
    ) -> None:
        if policy not in ("prefix", "lifecycle"):
            raise ValueError("policy must be 'prefix' or 'lifecycle'")
        if policy == "lifecycle" and lifecycle_tracker is None:
            lifecycle_tracker = LifecycleLoadTracker(block_size=block_size)
        self.prefix_index = prefix_index
        self.policy = policy
        self.block_size = block_size
        self.lifecycle_tracker = lifecycle_tracker

    def choose(
        self,
        request_id: str,
        prompts: tuple[PromptFingerprint, ...],
        *,
        candidate_node_ids: tuple[str, ...] | None = None,
        node_loads: dict[str, int] | None = None,
    ) -> PrefixRouteDecision | None:
        """Choose a backend and reserve lifecycle load when enabled."""
        if not prompts:
            return None
        scores_by_prompt = [
            self.prefix_index.score_nodes(
                prompt.block_hashes,
                prompt.num_tokens,
                candidate_node_ids=candidate_node_ids,
            )
            for prompt in prompts
        ]
        if self.policy == "prefix":
            best: PrefixRouteDecision | None = None
            for scores in scores_by_prompt:
                decision = self.prefix_index.choose_scored_node(
                    scores, node_loads=node_loads
                )
                if decision is not None and (
                    best is None or decision.matched_tokens > best.matched_tokens
                ):
                    best = decision
            return best

        tracker = self.lifecycle_tracker
        if tracker is None:
            raise RuntimeError("lifecycle tracker is not configured")
        work_by_target: dict[tuple[str, int | None], _AggregatedWork] = {}
        prompt_keys = tuple(
            self._prompt_block_keys(request_id, prompt_index, prompt)
            for prompt_index, prompt in enumerate(prompts)
        )
        for prompt, scores in zip(prompts, scores_by_prompt, strict=True):
            for score in scores:
                key = (score.node_id, score.data_parallel_rank)
                work = work_by_target.setdefault(key, _AggregatedWork())
                work.prompt_tokens += prompt.num_tokens
                work.matched_tokens += score.matched_tokens
                work.cache_size_blocks = score.cache_size_blocks
                work.prompt_work.append((prompt.num_tokens, score.matched_tokens))

        block_keys = tuple(key for keys in prompt_keys for key in keys)
        candidates = [
            RequestLoad(
                node_id=node_id,
                data_parallel_rank=rank,
                prompt_tokens=work.prompt_tokens,
                matched_tokens=work.matched_tokens,
                prompt_count=len(prompts),
                block_keys=block_keys,
                cache_size_blocks=work.cache_size_blocks,
                prompt_loads=tuple(
                    PromptLoad(
                        prompt_tokens=prompt_tokens,
                        matched_tokens=matched_tokens,
                        block_keys=keys,
                    )
                    for (prompt_tokens, matched_tokens), keys in zip(
                        work.prompt_work, prompt_keys, strict=True
                    )
                ),
            )
            for (node_id, rank), work in work_by_target.items()
        ]
        if not candidates:
            return None
        selection = tracker.select_and_reserve(request_id, candidates)
        selected = selection.selected.load
        return PrefixRouteDecision(
            node_id=selected.node_id,
            data_parallel_rank=selected.data_parallel_rank,
            matched_tokens=selected.matched_tokens,
            cache_size_blocks=selected.cache_size_blocks,
            decision_id=request_id,
        )

    def release(self, request_id: str) -> bool:
        """Idempotently release a lifecycle reservation."""
        if self.lifecycle_tracker is None:
            return False
        return self.lifecycle_tracker.free(request_id)

    def _prompt_block_keys(
        self,
        request_id: str,
        prompt_index: int,
        prompt: PromptFingerprint,
    ) -> tuple[bytes, ...]:
        keys = [b"hash:" + block_hash for block_hash in prompt.block_hashes]
        if prompt.num_tokens % self.block_size:
            tail = hashlib.sha256(f"{request_id}:{prompt_index}".encode()).digest()
            keys.append(b"tail:" + tail)
        if not keys:
            unique = hashlib.sha256(f"{request_id}:{prompt_index}".encode()).digest()
            keys.append(b"request:" + unique)
        return tuple(keys)
