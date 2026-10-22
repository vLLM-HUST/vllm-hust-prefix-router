"""Deterministic recovery regressions, independent of an installed host."""

import asyncio
from uuid import uuid4

import pytest

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.recovery import CacheEventRecovery
from vllm_hust_prefix_router.events.zmq_recovery import (
    ZmqEventSourceConfig,
    ZmqEventSubscriber,
)
from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    CacheEventBatch,
    CacheReplayResponse,
)


class Codec:
    snapshot_replay = True

    batches = {
        b"snapshot": CacheEventBatch(
            (AllBlocksCleared(), BlockStored((b"a", b"b"), 16))
        ),
        b"remove": CacheEventBatch((BlockRemoved((b"a",)),)),
        b"missing": CacheEventBatch((BlockStored((b"c",), 16),)),
        b"live": CacheEventBatch((BlockStored((b"d",), 16),)),
        b"rank0": CacheEventBatch((BlockStored((b"x",), 16),), 0),
        b"rank1": CacheEventBatch((BlockStored((b"y",), 16),), 1),
    }

    def decode_batch(self, payload):
        return self.batches[payload]


def make_recovery(responses=(), *, rank=None):
    index = GlobalPrefixIndex()
    index.register_node(
        "worker",
        hash_block_size=16,
        group_block_sizes={0: 16},
        data_parallel_rank=rank,
    )
    pending = list(responses)

    async def replay(_state, _force_snapshot):
        return pending.pop(0)

    recovery = CacheEventRecovery(
        node_id="worker",
        data_parallel_rank=rank,
        index=index,
        codec=Codec(),
        replay=replay,
    )
    return index, recovery


def matched(index, hashes=(b"a", b"b")):
    return index.score_nodes(hashes, 33)[0].matched_tokens


@pytest.mark.asyncio
async def test_gap_repair_preserves_unchanged_prefix():
    index, recovery = make_recovery(
        [
            CacheReplayResponse("epoch", 3, (), b"snapshot"),
            CacheReplayResponse("epoch", 4, ((3, b"missing"),)),
        ]
    )
    await recovery.initialize()
    await recovery.apply_live(4, b"live")
    assert recovery.state.next_seq == 5
    assert matched(index) == 32
    assert matched(index, (b"c", b"d")) == 32


@pytest.mark.asyncio
async def test_delayed_snapshot_cannot_restore_removed_prefix():
    index, recovery = make_recovery([CacheReplayResponse("epoch", 3, (), b"snapshot")])
    await recovery.initialize()
    await recovery.apply_live(3, b"remove")
    with pytest.raises(ValueError, match="stale replay"):
        recovery.apply_replay(CacheReplayResponse("epoch", 3, (), b"snapshot"))
    assert recovery.state.next_seq == 4
    assert matched(index) == 0
    recovery.apply_replay(CacheReplayResponse("epoch", 4, ()))
    assert matched(index) == 0


@pytest.mark.asyncio
async def test_duplicate_live_is_ignored_after_replay_coverage():
    index, recovery = make_recovery(
        [
            CacheReplayResponse("epoch", 3, (), b"snapshot"),
            CacheReplayResponse("epoch", 5, ((3, b"missing"), (4, b"live"))),
        ]
    )
    await recovery.initialize()
    await recovery.apply_live(4, b"live")
    await recovery.apply_live(4, b"remove")
    assert matched(index) == 32
    assert recovery.state.next_seq == 5


@pytest.mark.asyncio
async def test_mixed_snapshot_delta_rejected_before_mutation():
    index, recovery = make_recovery([CacheReplayResponse("epoch", 3, (), b"snapshot")])
    await recovery.initialize()
    with pytest.raises(ValueError, match="must not mix"):
        recovery.apply_replay(
            CacheReplayResponse("epoch", 4, ((3, b"missing"),), b"remove")
        )
    assert recovery.state.next_seq == 3
    assert matched(index) == 0  # Invalid response revokes trust.
    recovery.apply_replay(CacheReplayResponse("epoch", 3, ()))
    assert matched(index) == 32  # But the unchanged recovery base survived.


@pytest.mark.asyncio
async def test_rank_error_cannot_partially_apply_replay():
    index, recovery = make_recovery(
        [CacheReplayResponse("epoch", 3, (), b"snapshot")], rank=0
    )
    await recovery.initialize()
    with pytest.raises(ValueError, match="rank"):
        recovery.apply_replay(
            CacheReplayResponse("epoch", 5, ((3, b"rank0"), (4, b"rank1")))
        )
    assert recovery.state.next_seq == 3
    recovery.apply_replay(CacheReplayResponse("epoch", 3, ()))
    assert matched(index, (b"x",)) == 0
    assert matched(index) == 32


def test_index_staging_rolls_back_all_batches_on_rank_error():
    index, _ = make_recovery(rank=0)
    with pytest.raises(ValueError, match="configured for rank"):
        index.apply_recovery_batches(
            "worker",
            [Codec.batches[b"rank0"], Codec.batches[b"rank1"]],
            data_parallel_rank=0,
            replace_cache=False,
        )
    assert matched(index, (b"x",)) == 0


def test_snapshot_binds_unconfigured_source_to_observed_rank():
    index, recovery = make_recovery()
    recovery.apply_replay(CacheReplayResponse("epoch", 3, (), b"rank0"))
    assert index.score_nodes((b"x",), 17)[0].data_parallel_rank == 0
    with pytest.raises(ValueError, match="rank"):
        recovery.apply_replay(CacheReplayResponse("epoch", 4, ((3, b"rank1"),)))
    assert recovery.state.next_seq == 3
    recovery.apply_replay(CacheReplayResponse("epoch", 3, ()))
    assert len(index.score_nodes((b"x",), 17)) == 1
    assert matched(index, (b"x",)) == 16


@pytest.mark.parametrize("sequence", [-1, True, 1.5])
def test_invalid_replay_watermark_cannot_mutate(sequence):
    index, recovery = make_recovery()
    with pytest.raises(ValueError, match="non-negative"):
        recovery.apply_replay(CacheReplayResponse("epoch", sequence, (), b"snapshot"))
    assert recovery.state.next_seq is None
    assert matched(index) == 0


def test_retired_epoch_cannot_return():
    index, recovery = make_recovery()
    recovery.apply_replay(CacheReplayResponse("old", 10, (), b"snapshot"))
    recovery.apply_replay(CacheReplayResponse("new", 1, (), b"remove"))
    with pytest.raises(ValueError, match="retired"):
        recovery.apply_replay(CacheReplayResponse("old", 11, (), b"snapshot"))
    assert recovery.state.publisher_epoch == "new"
    assert recovery.state.next_seq == 1
    assert matched(index) == 0


@pytest.mark.asyncio
async def test_sync_and_live_commit_are_serialized():
    index, recovery = make_recovery()
    recovery.apply_replay(CacheReplayResponse("epoch", 3, (), b"snapshot"))
    requested, release = asyncio.Event(), asyncio.Event()

    async def replay(state, _force_snapshot):
        assert state is not recovery.state
        requested.set()
        await release.wait()
        return CacheReplayResponse("epoch", 3, (), b"snapshot")

    recovery.replay = replay
    sync_task = asyncio.create_task(recovery.synchronize())
    live_task = None
    try:
        await asyncio.wait_for(requested.wait(), 1)
        live_task = asyncio.create_task(recovery.apply_live(3, b"remove"))
        release.set()
        await asyncio.wait_for(asyncio.gather(sync_task, live_task), 1)
        assert recovery.state.next_seq == 4
        assert matched(index) == 0
    finally:
        tasks = [task for task in (sync_task, live_task) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("failure", ["timeout", "cancel", "return"])
def test_idle_subscriber_observes_periodic_task_failure(failure):
    # pyzmq requires add_reader; Windows' default Proactor loop lacks it.
    # An explicit selector also works on Python 3.10 without Runner/loop_factory.
    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(_idle_subscriber_failure(failure))
    finally:
        loop.close()


async def _idle_subscriber_failure(failure):
    index, _ = make_recovery()
    entered, release = asyncio.Event(), asyncio.Event()

    class Subscriber(ZmqEventSubscriber):
        async def _request_replay(self, _state, _force_snapshot):
            return CacheReplayResponse("epoch", 3, (), b"snapshot")

        async def _periodic_sync(self):
            entered.set()
            await release.wait()
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "return":
                return
            raise TimeoutError("injected periodic failure")

    identity = uuid4().hex
    source = Subscriber(
        ZmqEventSourceConfig(
            "worker", f"inproc://events-{identity}", f"inproc://replay-{identity}"
        ),
        index=index,
        codec=Codec(),
    )

    async def failure_observed():
        while source.last_error is None:
            await asyncio.sleep(0)

    await source.start()
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert source.ready and matched(index) == 32
        release.set()
        await asyncio.wait_for(failure_observed(), 1)
        assert not source.ready
        assert matched(index) == 0
        assert "periodic" in source.last_error
    finally:
        await source.close()
    assert source._task is None
    assert not source.ready
    assert matched(index) == 0


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_recovery_timeouts_are_finite_positive(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        ZmqEventSourceConfig(
            "worker", "inproc://events", "inproc://replay", replay_timeout_s=timeout
        )


@pytest.mark.asyncio
async def test_gap_is_untrusted_while_replay_waits():
    index, recovery = make_recovery()
    recovery.apply_replay(CacheReplayResponse("epoch", 3, (), b"snapshot"))
    requested, release = asyncio.Event(), asyncio.Event()

    async def replay(_state, _force_snapshot):
        requested.set()
        await release.wait()
        return CacheReplayResponse("epoch", 4, ((3, b"missing"),))

    recovery.replay = replay
    task = asyncio.create_task(recovery.apply_live(4, b"live"))
    try:
        await asyncio.wait_for(requested.wait(), 1)
        assert matched(index) == 0
        release.set()
        await asyncio.wait_for(task, 1)
        assert matched(index) == 32
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_replay_revokes_trust_without_losing_base():
    index, recovery = make_recovery()
    recovery.apply_replay(CacheReplayResponse("epoch", 3, (), b"snapshot"))
    requested = asyncio.Event()

    async def replay(_state, _force_snapshot):
        requested.set()
        await asyncio.Event().wait()

    recovery.replay = replay
    task = asyncio.create_task(recovery.synchronize())
    try:
        await asyncio.wait_for(requested.wait(), 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert matched(index) == 0
    assert recovery.state.next_seq == 3
    recovery.apply_replay(CacheReplayResponse("epoch", 3, ()))
    assert matched(index) == 32


@pytest.mark.asyncio
async def test_failed_reinitialization_cannot_trust_an_empty_delta():
    index, recovery = make_recovery()
    recovery.apply_replay(CacheReplayResponse("epoch", 3, (), b"snapshot"))

    async def failed_replay(_state, force_snapshot):
        assert force_snapshot
        raise TimeoutError("snapshot unavailable")

    recovery.replay = failed_replay
    with pytest.raises(TimeoutError):
        await recovery.initialize()
    with pytest.raises(ValueError, match="requires a full snapshot"):
        recovery.apply_replay(CacheReplayResponse("epoch", 3, ()))
    assert matched(index) == 0
    recovery.apply_replay(CacheReplayResponse("epoch", 3, (), b"snapshot"))
    assert matched(index) == 32
