import pytest

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.recovery import CacheEventRecovery
from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    BlockStored,
    CacheEventBatch,
    CacheReplayResponse,
)


class FakeCodec:
    def __init__(self, batches):
        self.batches = batches

    def decode_batch(self, payload):
        return self.batches[payload]


def _index() -> GlobalPrefixIndex:
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16, group_block_sizes={0: 16})
    return index


@pytest.mark.asyncio
async def test_initial_snapshot_establishes_trusted_prefix_state() -> None:
    index = _index()
    codec = FakeCodec(
        {
            b"snapshot": CacheEventBatch(
                (AllBlocksCleared(), BlockStored((b"a",), block_size=16))
            )
        }
    )

    async def replay(state, force_snapshot):
        assert state.next_seq is None
        assert force_snapshot
        return CacheReplayResponse("epoch-1", 3, (), b"snapshot")

    recovery = CacheEventRecovery(
        node_id="node-0",
        data_parallel_rank=None,
        index=index,
        codec=codec,
        replay=replay,
    )
    await recovery.initialize()

    assert recovery.state.publisher_epoch == "epoch-1"
    assert recovery.state.next_seq == 3
    assert index.score_nodes((b"a",), 17)[0].matched_tokens == 16


@pytest.mark.asyncio
async def test_live_gap_replays_missing_batch_before_new_event() -> None:
    index = _index()
    codec = FakeCodec(
        {
            b"snapshot": CacheEventBatch((AllBlocksCleared(),)),
            b"missing": CacheEventBatch((BlockStored((b"a",), block_size=16),)),
            b"live": CacheEventBatch((BlockStored((b"b",), block_size=16),)),
        }
    )
    responses = [
        CacheReplayResponse("epoch-1", 1, (), b"snapshot"),
        CacheReplayResponse("epoch-1", 2, ((1, b"missing"),)),
    ]

    async def replay(_state, _force_snapshot):
        return responses.pop(0)

    recovery = CacheEventRecovery(
        node_id="node-0",
        data_parallel_rank=None,
        index=index,
        codec=codec,
        replay=replay,
    )
    await recovery.initialize()
    await recovery.apply_live(2, b"live")

    assert recovery.state.next_seq == 3
    assert index.score_nodes((b"a", b"b"), 33)[0].matched_tokens == 32


def test_generation_change_without_snapshot_stays_untrusted() -> None:
    index = _index()
    codec = FakeCodec({})

    async def replay(_state, _force_snapshot):
        raise AssertionError("not called")

    recovery = CacheEventRecovery(
        node_id="node-0",
        data_parallel_rank=None,
        index=index,
        codec=codec,
        replay=replay,
    )
    recovery.state.publisher_epoch = "epoch-1"
    recovery.state.next_seq = 4
    index.apply_events("node-0", [BlockStored((b"old",), block_size=16)])
    index.invalidate_node("node-0")

    with pytest.raises(ValueError, match="requires a full snapshot"):
        recovery.apply_replay(CacheReplayResponse("epoch-2", 4, ()))

    assert index.score_nodes((b"old",), 17)[0].matched_tokens == 0
