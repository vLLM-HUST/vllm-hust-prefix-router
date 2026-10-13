"""ZMQ subscription and replay transport for one remote backend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.recovery import CacheEventRecovery, RecoveryState
from vllm_hust_prefix_router.protocols.cache_events import CacheReplayResponse


class ZmqEventCodec(Protocol):
    """Wire codec needed by the ZMQ event transport."""

    snapshot_replay: bool

    def decode_batch(self, payload: bytes): ...

    def encode_replay_request(
        self,
        *,
        publisher_epoch: str | None,
        start_seq: int,
        force_snapshot: bool,
    ) -> bytes: ...

    def decode_replay_response(self, payload: bytes) -> CacheReplayResponse: ...


@dataclass(frozen=True)
class ZmqEventSourceConfig:
    """Endpoints and recovery timing for one backend event source."""

    node_id: str
    event_endpoint: str
    replay_endpoint: str
    data_parallel_rank: int | None = None
    topic: str = ""
    replay_timeout_s: float = 2.0
    sync_interval_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.node_id or not self.event_endpoint or not self.replay_endpoint:
            raise ValueError("node and ZMQ endpoints must be non-empty")
        if self.replay_timeout_s <= 0 or self.sync_interval_s <= 0:
            raise ValueError("ZMQ recovery timeouts must be positive")


class ZmqEventSubscriber:
    """Maintain a trusted Prefix cache view across gaps and restarts."""

    def __init__(
        self,
        config: ZmqEventSourceConfig,
        *,
        index: GlobalPrefixIndex,
        codec: ZmqEventCodec,
    ) -> None:
        self.config = config
        self.index = index
        self.codec = codec
        self.ready = False
        self.last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._recovery_lock = asyncio.Lock()
        self._recovery = CacheEventRecovery(
            node_id=config.node_id,
            data_parallel_rank=config.data_parallel_rank,
            index=index,
            codec=codec,
            replay=self._request_replay,
            snapshot_replay=codec.snapshot_replay,
        )

    async def start(self) -> None:
        """Start the reconnecting subscriber task."""
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name=f"kv-events-{self.config.node_id}"
            )

    async def close(self) -> None:
        """Stop the subscriber and invalidate its cache claims."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self.ready = False
        self.index.invalidate_node(self.config.node_id, self.config.data_parallel_rank)

    def status(self) -> dict[str, object]:
        """Return health and sequence state for readiness evidence."""
        return {
            "ready": self.ready,
            "last_error": self.last_error,
            "publisher_epoch": self._recovery.state.publisher_epoch,
            "next_seq": self._recovery.state.next_seq,
        }

    async def _run(self) -> None:
        import zmq
        import zmq.asyncio

        context = zmq.asyncio.Context.instance()
        reconnect_delay = 0.1
        while True:
            socket = context.socket(zmq.SUB)
            sync_task: asyncio.Task[None] | None = None
            try:
                socket.setsockopt(zmq.SUBSCRIBE, self.config.topic.encode())
                socket.connect(self.config.event_endpoint)
                async with self._recovery_lock:
                    await self._recovery.initialize()
                self.ready = True
                self.last_error = None
                reconnect_delay = 0.1
                sync_task = asyncio.create_task(self._periodic_sync())
                while True:
                    receive_task = asyncio.ensure_future(socket.recv_multipart())
                    try:
                        done, _ = await asyncio.wait(
                            (receive_task, sync_task),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if sync_task in done:
                            sync_task.result()
                            raise RuntimeError("periodic KV recovery stopped")
                        frames = receive_task.result()
                    finally:
                        if not receive_task.done():
                            receive_task.cancel()
                            await asyncio.gather(receive_task, return_exceptions=True)
                    if (
                        len(frames) != 3
                        or frames[0] != self.config.topic.encode()
                        or len(frames[1]) != 8
                    ):
                        raise ValueError("malformed ZMQ KV event frames")
                    sequence = int.from_bytes(frames[1], "big")
                    async with self._recovery_lock:
                        await self._recovery.apply_live(sequence, frames[2])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.ready = False
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.index.invalidate_node(
                    self.config.node_id, self.config.data_parallel_rank
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 5.0)
            finally:
                if sync_task is not None:
                    sync_task.cancel()
                    await asyncio.gather(sync_task, return_exceptions=True)
                socket.close(linger=0)

    async def _periodic_sync(self) -> None:
        while True:
            await asyncio.sleep(self.config.sync_interval_s)
            async with self._recovery_lock:
                response = await self._request_replay(self._recovery.state, False)
                self._recovery.apply_replay(response)

    async def _request_replay(
        self, state: RecoveryState, force_snapshot: bool
    ) -> CacheReplayResponse:
        import zmq
        import zmq.asyncio

        context = zmq.asyncio.Context.instance()
        if not self.codec.snapshot_replay:
            return await self._request_sequence_replay(context, state)
        socket: Any = context.socket(zmq.REQ)
        socket.connect(self.config.replay_endpoint)
        try:
            payload = self.codec.encode_replay_request(
                publisher_epoch=state.publisher_epoch,
                start_seq=state.next_seq or 0,
                force_snapshot=force_snapshot,
            )
            await socket.send(payload)
            response = await asyncio.wait_for(
                socket.recv(), timeout=self.config.replay_timeout_s
            )
            return self.codec.decode_replay_response(response)
        finally:
            socket.close(linger=0)

    async def _request_sequence_replay(
        self, context: Any, state: RecoveryState
    ) -> CacheReplayResponse:
        """Read the bounded sequence-frame stream used by newer hosts."""
        import zmq

        start_seq = state.next_seq or 0
        socket: Any = context.socket(zmq.DEALER)
        socket.connect(self.config.replay_endpoint)
        batches: list[tuple[int, bytes]] = []
        try:
            await socket.send_multipart((b"", start_seq.to_bytes(8, "big")))
            while True:
                frames = await asyncio.wait_for(
                    socket.recv_multipart(), timeout=self.config.replay_timeout_s
                )
                if len(frames) == 3:
                    delimiter, sequence_frame, payload = frames
                    topic = None
                elif len(frames) == 4:
                    delimiter, topic, sequence_frame, payload = frames
                else:
                    raise ValueError(
                        "malformed sequence replay frames: "
                        f"frame lengths {[len(frame) for frame in frames]}"
                    )
                if delimiter != b"" or len(sequence_frame) != 8:
                    raise ValueError("malformed sequence replay frame header")
                if sequence_frame == b"\xff" * 8:
                    if topic not in (None, b"") or payload != b"":
                        raise ValueError("malformed sequence replay terminator")
                    break
                if topic is not None and topic != self.config.topic.encode():
                    raise ValueError("sequence replay topic does not match")
                sequence = int.from_bytes(sequence_frame, "big")
                batches.append((sequence, payload))
            next_seq = batches[-1][0] + 1 if batches else start_seq
            return CacheReplayResponse(
                publisher_epoch="unversioned",
                next_seq=next_seq,
                replayed_batches=tuple(batches),
            )
        finally:
            socket.close(linger=0)
