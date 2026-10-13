"""Check the installed host's KV event/replay wire against this Router wheel."""

from __future__ import annotations

import asyncio
import socket

from vllm.distributed.kv_events import (
    BlockRemoved as HostBlockRemoved,
)
from vllm.distributed.kv_events import (
    BlockStored as HostBlockStored,
)
from vllm.distributed.kv_events import KVEventBatch, ZmqEventPublisher

from vllm_hust_prefix_router.adapters.vllm_hust.kv_events import (
    VllmHustKvEventCodec,
)
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.recovery import (
    CacheEventRecovery,
    RecoveryState,
)
from vllm_hust_prefix_router.events.zmq_recovery import (
    ZmqEventSourceConfig,
    ZmqEventSubscriber,
)


async def main() -> None:
    def free_port() -> int:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return listener.getsockname()[1]

    event_port, replay_port = free_port(), free_port()
    while replay_port == event_port:
        replay_port = free_port()
    event_endpoint = f"tcp://127.0.0.1:{event_port}"
    replay_endpoint = f"tcp://127.0.0.1:{replay_port}"
    publisher = ZmqEventPublisher(
        0,
        endpoint=f"tcp://*:{event_port}",
        replay_endpoint=f"tcp://*:{replay_port}",
        topic="tp2",
    )
    codec = VllmHustKvEventCodec()
    index = GlobalPrefixIndex()
    index.register_node(
        "tp2",
        hash_block_size=16,
        data_parallel_rank=0,
        group_block_sizes={0: 16},
    )
    subscriber = ZmqEventSubscriber(
        ZmqEventSourceConfig(
            "tp2",
            event_endpoint,
            replay_endpoint,
            data_parallel_rank=0,
            topic="tp2",
        ),
        index=index,
        codec=codec,
    )
    try:
        publisher.publish(
            KVEventBatch(
                0.0,
                [HostBlockStored([b"hash"], None, [1] * 16, 16, None, None, None)],
                0,
            )
        )
        await asyncio.sleep(0.2)
        response = await subscriber._request_replay(RecoveryState(), False)
        assert response.next_seq == 1
        assert len(response.replayed_batches) == 1
        batch = codec.decode_batch(response.replayed_batches[0][1])
        assert batch.data_parallel_rank == 0
        assert batch.events[0].block_hashes == (b"hash",)
        recovery = CacheEventRecovery(
            node_id="tp2",
            data_parallel_rank=0,
            index=index,
            codec=codec,
            replay=subscriber._request_replay,
            snapshot_replay=False,
        )
        await recovery.initialize()
        assert index.score_nodes((b"hash",), 17)[0].matched_tokens == 16

        publisher.publish(KVEventBatch(0.1, [HostBlockRemoved([b"hash"], None)], 0))
        publisher.publish(
            KVEventBatch(
                0.2,
                [HostBlockStored([b"next"], None, [2] * 16, 16, None, None, None)],
                0,
            )
        )
        await asyncio.sleep(0.2)
        live = await subscriber._request_replay(RecoveryState(next_seq=2), False)
        assert live.replayed_batches[0][0] == 2
        await recovery.apply_live(2, live.replayed_batches[0][1])
        assert index.score_nodes((b"hash",), 17)[0].matched_tokens == 0
        assert index.score_nodes((b"next",), 17)[0].matched_tokens == 16
        print("host_kv_wire=passed")
        print("event_loss_recovery=passed")
        print(f"snapshot_replay={codec.snapshot_replay}")
    finally:
        publisher.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
