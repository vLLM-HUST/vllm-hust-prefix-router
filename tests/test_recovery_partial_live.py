"""A rejected live batch must never leave a reusable partial recovery base."""

import pytest

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.recovery import CacheEventRecovery
from vllm_hust_prefix_router.protocols.cache_events import (
    BlockStored,
    CacheEventBatch,
    CacheReplayResponse,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("invalid_group", [False, True])
async def test_invalid_live_batch_requires_rebuild(snapshot, invalid_group):
    class Codec:
        def decode_batch(self, payload):
            if payload == b"invalid":
                return CacheEventBatch(
                    (
                        BlockStored((b"ghost",), 16, 0),
                        BlockStored((b"bad",), 64, 9 if invalid_group else 1),
                    ),
                    0,
                )
            return CacheEventBatch(
                (
                    BlockStored((payload,), 16, 0),
                    BlockStored((payload,), 32, 1),
                ),
                0,
            )

    index = GlobalPrefixIndex()
    index.register_node(
        "worker",
        hash_block_size=16,
        data_parallel_rank=0,
        group_block_sizes={0: 16, 1: 32},
    )

    async def unused(*_args):
        raise AssertionError("unexpected transport call")

    recovery = CacheEventRecovery(
        node_id="worker",
        data_parallel_rank=0,
        index=index,
        codec=Codec(),
        replay=unused,
        snapshot_replay=snapshot,
    )
    recovery.apply_replay(
        CacheReplayResponse("epoch", 1, (), b"old")
        if snapshot
        else CacheReplayResponse("unversioned", 1, ((0, b"old"),))
    )
    with pytest.raises(
        ValueError, match="unexpected KV cache group|block size changed"
    ):
        await recovery.apply_live(1, b"invalid")
    assert not index._nodes["worker", 0].trusted
    assert not index._nodes["worker", 0].group_hashes

    calls = []

    async def rebuild(state, force_snapshot):
        calls.append((state.next_seq, force_snapshot))
        return (
            CacheReplayResponse("epoch", 2, (), b"clean")
            if snapshot
            else CacheReplayResponse("unversioned", 2, ((1, b"clean"),))
        )

    if snapshot:
        with pytest.raises(ValueError, match="requires a full snapshot"):
            recovery.apply_replay(CacheReplayResponse("epoch", 1, ()))
    else:
        assert recovery.state.next_seq is None
    recovery.replay = rebuild
    await recovery.synchronize()
    assert calls == [(1 if snapshot else None, snapshot)]
    assert index._nodes["worker", 0].trusted
    assert index._nodes["worker", 0].group_hashes == {0: {b"clean"}, 1: {b"clean"}}
