"""Bounded integer replay for a managed, initially empty native backend.

END_SEQ proves the reply stream ended; it is not an engine cache watermark.
Consequently this profile exposes observed-contiguous routing hints only.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.managed_identity import ManagedBackendLease
from vllm_hust_prefix_router.protocols.cache_events import CacheEventBatch

END_SEQ = b"\xff" * 8


class BatchCodec(Protocol):
    def decode_batch(self, payload: bytes) -> CacheEventBatch: ...


class ReplayGap(ValueError):
    """A native replay cannot rebuild the known cache history."""


@dataclass(frozen=True)
class NativeReplayConfig:
    node_id: str
    url: str
    lease_path: Path
    model: str
    hash_block_size: int
    identity_digest: str
    data_parallel_rank: int = 0
    timeout_s: float = 2.0
    interval_s: float = 0.2
    view_ttl_s: float = 3.0
    max_batches: int = 10000
    max_bytes: int = 16 * 1024 * 1024
    observe_replay: bool = False
    reply_delimiter: bool = False

    def __post_init__(self) -> None:
        if type(self.observe_replay) is not bool:
            raise ValueError("observe_replay must be boolean")
        if type(self.reply_delimiter) is not bool:
            raise ValueError("reply_delimiter must be boolean")
        for value in (self.timeout_s, self.interval_s, self.view_ttl_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("native replay timings must be finite and positive")
        if self.view_ttl_s <= self.interval_s:
            raise ValueError("view TTL must exceed the poll interval")
        if any(
            type(v) is not int or v <= 0 for v in (self.max_batches, self.max_bytes)
        ):
            raise ValueError("native replay resource limits must be positive integers")


async def request_native_replay(
    endpoint: str, start_seq: int, config: NativeReplayConfig
) -> tuple[tuple[int, bytes], ...]:
    """DEALER preserves the native ROUTER's multiple replies per request."""
    import zmq
    import zmq.asyncio

    socket = zmq.asyncio.Context.instance().socket(zmq.DEALER)
    socket.setsockopt(zmq.MAXMSGSIZE, config.max_bytes)
    socket.setsockopt(zmq.RCVHWM, config.max_batches + 1)
    socket.connect(endpoint)

    async def transaction():
        await socket.send_multipart((b"", start_seq.to_bytes(8, "big")))
        batches, size = [], 0
        expected = start_seq
        while True:
            frames = await socket.recv_multipart()
            sequence_bytes, payload = decode_native_reply(
                frames, reply_delimiter=config.reply_delimiter
            )
            if sequence_bytes == END_SEQ:
                return tuple(batches)
            sequence = int.from_bytes(sequence_bytes, "big")
            if sequence != expected:
                raise ReplayGap(f"native replay expected {expected}, got {sequence}")
            size += len(payload)
            if len(batches) >= config.max_batches or size > config.max_bytes:
                raise ReplayGap("native replay exceeds configured resource limits")
            batches.append((sequence, payload))
            expected += 1

    try:
        return await asyncio.wait_for(transaction(), timeout=config.timeout_s)
    finally:
        socket.close(linger=0)


def decode_native_reply(
    frames: list[bytes], *, reply_delimiter: bool = False
) -> tuple[bytes, bytes]:
    """Decode one explicitly selected framing; never infer a host profile."""
    offset = int(reply_delimiter)
    if (
        len(frames) != 3 + offset
        or any(frame != b"" for frame in frames[: 1 + offset])
        or len(frames[1 + offset]) != 8
    ):
        raise ValueError("malformed native replay frames")
    sequence, payload = frames[1 + offset :]
    if sequence == END_SEQ and payload:
        raise ValueError("native replay END_SEQ must have an empty payload")
    return sequence, payload


Replay = Callable[
    [str, int, NativeReplayConfig], Awaitable[tuple[tuple[int, bytes], ...]]
]


class NativeReplaySubscriber:
    """Poll replay without applying untagged PUB frames across incarnations."""

    def __init__(
        self,
        config: NativeReplayConfig,
        *,
        index: GlobalPrefixIndex,
        codec: BatchCodec,
        replay: Replay = request_native_replay,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config, self.index, self.codec = config, index, codec
        self._replay, self._clock = replay, clock
        self._lease: ManagedBackendLease | None = None
        self._next_seq = 0
        self._last_success: float | None = None
        self._failed_generation: str | None = None
        self._seen_generations: set[str] = set()
        self._used_endpoints: set[str] = set()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._instance_id = uuid4().hex
        self._sync_attempts = 0
        self._committed_attempt: int | None = None
        self.last_error: str | None = None
        self._attempts = 0
        self._completed_attempts = 0
        self._failed_attempts = 0
        self._diagnostics: deque[dict[str, object]] = deque(maxlen=64)
        self._invalidate()

    def _invalidate(self) -> None:
        self._last_success = None
        self.index.invalidate_node(
            self.config.node_id, self.config.data_parallel_rank, clear=False
        )

    def _read_lease(self) -> ManagedBackendLease:
        lease = ManagedBackendLease.read(self.config.lease_path)
        if (
            lease.node_id != self.config.node_id
            or lease.url != self.config.url
            or lease.model != self.config.model
            or lease.hash_block_size != self.config.hash_block_size
            or lease.identity_digest != self.config.identity_digest
        ):
            raise ValueError("managed lease does not match Router configuration")
        if not lease.is_current():
            raise ProcessLookupError(
                "managed API/engine incarnation is no longer alive"
            )
        return lease

    @property
    def ready(self) -> bool:
        if self._closed:
            return False
        try:
            valid = (
                self._last_success is not None
                and self._clock() - self._last_success < self.config.view_ttl_s
                and self._read_lease() == self._lease
            )
        except (OSError, ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            self._invalidate()
        return valid

    def status(self) -> dict[str, object]:
        ready = self.ready
        return {
            "node_id": self.config.node_id,
            "ready": ready,
            "last_error": self.last_error,
            "generation": self._lease.generation if self._lease else None,
            "next_seq": self._next_seq,
            "sync_attempts": self._sync_attempts,
            "last_committed_replay_attempt_id": (
                f"{self._instance_id}:{self._committed_attempt}"
                if self._committed_attempt is not None
                else None
            ),
            "identity_digest": self.config.identity_digest,
            "data_parallel_rank": self.config.data_parallel_rank,
            "view_age_s": None
            if self._last_success is None
            else (self._clock() - self._last_success),
            "capability": "managed-native-contiguous-replay-hint",
            "authoritative_cache_snapshot": False,
            "atomic_pre_reuse_fence": False,
            "replay_diagnostics": {
                "enabled": self.config.observe_replay,
                "total_attempts": self._attempts,
                "completed_attempts": self._completed_attempts,
                "inflight_attempt": self._attempts
                if self._attempts != self._completed_attempts
                else None,
                "failed_attempts": self._failed_attempts,
                "recent_attempts": list(self._diagnostics),
            },
        }

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("native replay subscriber is closed")
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name=f"native-{self.config.node_id}"
            )

    async def close(self) -> None:
        # Also fence synchronize_once() calls not owned by the polling task.
        self._closed = True
        self._invalidate()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._invalidate()

    async def _run(self) -> None:
        while True:
            try:
                await self.synchronize_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(self.config.interval_s)

    async def synchronize_once(self) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("native replay subscriber is closed")
            self._sync_attempts += 1
            diagnostic = None
            if self.config.observe_replay:
                self._attempts += 1
                diagnostic = {
                    "attempt": self._attempts,
                    "attempt_id": f"{self._instance_id}:{self._sync_attempts}",
                    "generation": None,
                    "start_seq": self._next_seq,
                    "complete_reply_batches": 0,
                    "complete_reply_payload_bytes": 0,
                    "outcome": "failed",
                    "elapsed_ns": {"start": 0},
                }
                started = time.perf_counter_ns()
            try:
                lease = self._read_lease()
                if diagnostic is not None:
                    diagnostic["generation"] = lease.generation
                new_generation = (
                    self._lease is None or lease.generation != self._lease.generation
                )
                if new_generation:
                    if (
                        lease.generation in self._seen_generations
                        or lease.replay_endpoint in self._used_endpoints
                    ):
                        raise ReplayGap("generation or replay endpoint was reused")
                    if len(self._seen_generations) >= 1024:
                        raise ReplayGap("managed incarnation budget exhausted")
                    self._seen_generations.add(lease.generation)
                    self._used_endpoints.add(lease.replay_endpoint)
                    self._lease = lease
                    self._next_seq = 0
                    self._failed_generation = None
                    self.index.invalidate_node(
                        self.config.node_id, self.config.data_parallel_rank
                    )
                elif lease != self._lease:
                    raise ReplayGap("lease changed without a new generation")
                if self._failed_generation == lease.generation:
                    raise ReplayGap(
                        "incomplete history requires a new managed incarnation"
                    )
                if diagnostic is not None:
                    diagnostic["start_seq"] = self._next_seq
                    diagnostic["elapsed_ns"]["request_start"] = (
                        time.perf_counter_ns() - started
                    )
                batches = await self._replay(
                    lease.replay_endpoint, self._next_seq, self.config
                )
                if self._closed:
                    raise RuntimeError("native replay subscriber is closed")
                if diagnostic is not None:
                    diagnostic["complete_reply_batches"] = len(batches)
                    diagnostic["complete_reply_payload_bytes"] = sum(
                        len(payload) for _, payload in batches
                    )
                    diagnostic["elapsed_ns"]["reply_end"] = (
                        time.perf_counter_ns() - started
                    )
                if self._read_lease() != lease:
                    raise ProcessLookupError("incarnation changed during native replay")
                decoded = []
                expected = self._next_seq
                for sequence, payload in batches:
                    if type(sequence) is not int or sequence != expected:
                        raise ReplayGap("native replay sequence is not contiguous")
                    batch = self.codec.decode_batch(payload)
                    if batch.data_parallel_rank != self.config.data_parallel_rank:
                        raise ReplayGap("native batch rank does not match backend")
                    decoded.append(batch)
                    expected += 1
                if diagnostic is not None:
                    diagnostic["elapsed_ns"]["decode_end"] = (
                        time.perf_counter_ns() - started
                    )
                self.index.apply_recovery_batches(
                    self.config.node_id,
                    decoded,
                    data_parallel_rank=self.config.data_parallel_rank,
                    replace_cache=self._next_seq == 0,
                )
                self._next_seq = expected
                self._committed_attempt = self._sync_attempts
                self._last_success = self._clock()
                self.last_error = None
                if diagnostic is not None:
                    diagnostic["outcome"] = "committed"
                    diagnostic["end_seq"] = expected
                    diagnostic["elapsed_ns"]["commit_end"] = (
                        time.perf_counter_ns() - started
                    )
            except BaseException as exc:
                if isinstance(exc, ReplayGap) and self._lease is not None:
                    self._failed_generation = self._lease.generation
                self._invalidate()
                raise
            finally:
                if diagnostic is not None:
                    diagnostic["elapsed_ns"]["finish"] = (
                        time.perf_counter_ns() - started
                    )
                    if diagnostic["outcome"] != "committed":
                        self._failed_attempts += 1
                    self._diagnostics.append(diagnostic)
                    self._completed_attempts = self._attempts
