import asyncio
import sys
import uuid

import zmq
import zmq.asyncio

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.recovery import RecoveryState
from vllm_hust_prefix_router.events.zmq_recovery import (
    ZmqEventSourceConfig,
    ZmqEventSubscriber,
)
from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    CacheEventBatch,
    CacheReplayResponse,
)


class SequenceCodec:
    snapshot_replay = False


class SnapshotCodec:
    snapshot_replay = True

    def decode_batch(self, _payload):
        return CacheEventBatch((AllBlocksCleared(),), data_parallel_rank=0)


def _run(coroutine):
    if sys.platform != "win32":
        return asyncio.run(coroutine)
    loop = asyncio.SelectorEventLoop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


def test_sequence_frame_replay_uses_dealer_and_reads_until_marker() -> None:
    _run(_sequence_frame_replay())


def test_legacy_sequence_frame_replay_without_topic() -> None:
    _run(_legacy_sequence_frame_replay())


async def _legacy_sequence_frame_replay() -> None:
    context = zmq.asyncio.Context.instance()
    endpoint = f"inproc://router-replay-{uuid.uuid4().hex}"
    publisher = context.socket(zmq.ROUTER)
    publisher.bind(endpoint)
    index = GlobalPrefixIndex()
    index.register_node("tp2", hash_block_size=16, data_parallel_rank=0)
    subscriber = ZmqEventSubscriber(
        ZmqEventSourceConfig(
            node_id="tp2",
            event_endpoint="inproc://unused",
            replay_endpoint=endpoint,
            data_parallel_rank=0,
            topic="tp2",
        ),
        index=index,
        codec=SequenceCodec(),
    )

    async def publish_replay():
        identity, delimiter, start = await publisher.recv_multipart()
        assert delimiter == b""
        assert start == (7).to_bytes(8, "big")
        await publisher.send_multipart(
            (identity, b"", (7).to_bytes(8, "big"), b"first")
        )
        await publisher.send_multipart((identity, b"", b"\xff" * 8, b""))

    task = asyncio.create_task(publish_replay())
    try:
        response = await subscriber._request_replay(RecoveryState(next_seq=7), False)
        await task
        assert response.next_seq == 8
        assert response.replayed_batches == ((7, b"first"),)
    finally:
        publisher.close(linger=0)


async def _sequence_frame_replay() -> None:
    context = zmq.asyncio.Context.instance()
    endpoint = f"inproc://router-replay-{uuid.uuid4().hex}"
    publisher = context.socket(zmq.ROUTER)
    publisher.bind(endpoint)
    index = GlobalPrefixIndex()
    index.register_node("tp2", hash_block_size=16, data_parallel_rank=0)
    subscriber = ZmqEventSubscriber(
        ZmqEventSourceConfig(
            node_id="tp2",
            event_endpoint="inproc://unused",
            replay_endpoint=endpoint,
            data_parallel_rank=0,
            topic="tp2",
        ),
        index=index,
        codec=SequenceCodec(),
    )

    async def publish_replay():
        identity, delimiter, start = await publisher.recv_multipart()
        assert delimiter == b""
        assert start == (7).to_bytes(8, "big")
        await publisher.send_multipart(
            (identity, b"", b"tp2", (7).to_bytes(8, "big"), b"first")
        )
        await publisher.send_multipart(
            (identity, b"", b"tp2", (8).to_bytes(8, "big"), b"second")
        )
        await publisher.send_multipart((identity, b"", b"", b"\xff" * 8, b""))

    task = asyncio.create_task(publish_replay())
    try:
        response = await subscriber._request_replay(RecoveryState(next_seq=7), False)
        await task
        assert response.next_seq == 9
        assert response.replayed_batches == ((7, b"first"), (8, b"second"))
    finally:
        publisher.close(linger=0)


def test_periodic_recovery_failure_invalidates_readiness() -> None:
    _run(_periodic_failure())


async def _periodic_failure() -> None:
    index = GlobalPrefixIndex()
    index.register_node("tp2", hash_block_size=16, data_parallel_rank=0)
    subscriber = ZmqEventSubscriber(
        ZmqEventSourceConfig(
            node_id="tp2",
            event_endpoint=f"inproc://{uuid.uuid4().hex}",
            replay_endpoint=f"inproc://{uuid.uuid4().hex}",
            data_parallel_rank=0,
        ),
        index=index,
        codec=SnapshotCodec(),
    )

    async def replay(_state, _force_snapshot):
        return CacheReplayResponse("epoch", 0, (), b"snapshot")

    async def failed_sync():
        raise RuntimeError("replay transport failed")

    subscriber._recovery.replay = replay
    subscriber._periodic_sync = failed_sync
    await subscriber.start()
    try:
        for _ in range(100):
            if subscriber.last_error is not None:
                break
            await asyncio.sleep(0.01)
        assert not subscriber.ready
        assert subscriber.last_error is not None
        assert "replay transport failed" in subscriber.last_error
    finally:
        await subscriber.close()
