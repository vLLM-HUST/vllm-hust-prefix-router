"""Managed native replay: identity, bounded wire framing, and state isolation."""

import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.managed_identity import (
    ManagedBackendLease,
    ProcessIdentity,
)
from vllm_hust_prefix_router.events.native_replay import (
    END_SEQ,
    NativeReplayConfig,
    NativeReplaySubscriber,
    ReplayGap,
    request_native_replay,
)
from vllm_hust_prefix_router.protocols.cache_events import (
    BlockRemoved,
    BlockStored,
    CacheEventBatch,
)


class Codec:
    def decode_batch(self, payload):
        return {
            b"a": CacheEventBatch((BlockStored((b"a",), 128),), 0),
            b"b": CacheEventBatch((BlockStored((b"b",), 128),), 0),
            b"remove": CacheEventBatch((BlockRemoved((b"a",)),), 0),
            b"rank": CacheEventBatch((BlockStored((b"wrong",), 128),), 1),
        }[payload]


def lease(port=19000):
    return ManagedBackendLease(
        "worker",
        str(uuid4()),
        ProcessIdentity(10, "boot", "namespace", 100),
        ProcessIdentity(11, "boot", "namespace", 101),
        "http://127.0.0.1:18000",
        f"tcp://127.0.0.1:{port}",
        f"tcp://127.0.0.1:{port + 1}",
        "/model",
        128,
        "sha256",
        True,
        "a" * 64,
    )


def setup_source(monkeypatch, replay):
    current = {"lease": lease(), "alive": True, "time": 0.0}
    monkeypatch.setattr(ManagedBackendLease, "read", lambda _path: current["lease"])
    monkeypatch.setattr(
        ManagedBackendLease, "is_current", lambda _self: current["alive"]
    )
    index = GlobalPrefixIndex()
    index.register_node("worker", hash_block_size=128, data_parallel_rank=0)
    config = NativeReplayConfig(
        "worker", "http://127.0.0.1:18000", Path("/lease"), "/model", 128, "a" * 64
    )
    source = NativeReplaySubscriber(
        config, index=index, codec=Codec(), replay=replay, clock=lambda: current["time"]
    )
    return source, index, current


def match(index, value=b"a"):
    return index.score_nodes((value,), 129)[0].matched_tokens


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "http://127.0.0.1:18000/other"),
        ("replay_endpoint", "tcp://127.0.0.1:19001/"),
        ("event_endpoint", "tcp://127.0.0.1:19000/path"),
    ],
)
def test_managed_lease_rejects_endpoint_paths(field, value):
    with pytest.raises(ValueError, match="loopback"):
        replace(lease(), **{field: value})


@pytest.mark.asyncio
async def test_replay_bootstraps_and_observes_a_terminal_removal(monkeypatch):
    responses = [((0, b"a"),), ((1, b"remove"),), ()]
    starts = []

    async def replay(_endpoint, sequence, _config):
        starts.append(sequence)
        return responses.pop(0)

    source, index, _ = setup_source(monkeypatch, replay)
    await source.synchronize_once()
    assert source.ready and match(index) == 128
    # No subsequent live event is needed: polling retrieves the last lost removal.
    await source.synchronize_once()
    assert source.ready and match(index) == 0
    await source.synchronize_once()
    assert starts == [0, 1, 2]


@pytest.mark.asyncio
async def test_expired_view_and_dead_process_cannot_score(monkeypatch):
    async def replay(_endpoint, _sequence, _config):
        return ((0, b"a"),)

    source, index, current = setup_source(monkeypatch, replay)
    await source.synchronize_once()
    current["time"] = 4
    assert not source.ready and match(index) == 0
    current["time"] = 0
    source._last_success = 0
    current["alive"] = False
    assert not source.ready and match(index) == 0


@pytest.mark.asyncio
async def test_gap_requires_new_incarnation_then_positive_recovery(monkeypatch):
    responses = [((5, b"a"),), ((0, b"b"),)]

    async def replay(_endpoint, _sequence, _config):
        return responses.pop(0)

    source, index, current = setup_source(monkeypatch, replay)
    with pytest.raises(ReplayGap):
        await source.synchronize_once()
    with pytest.raises(ReplayGap, match="new managed incarnation"):
        await source.synchronize_once()
    assert not source.ready and match(index) == 0
    current["lease"] = lease(19010)
    await source.synchronize_once()
    assert source.ready and match(index, b"b") == 128
    assert match(index) == 0


@pytest.mark.asyncio
async def test_delayed_old_replay_cannot_cross_new_incarnation(monkeypatch):
    requested, release = asyncio.Event(), asyncio.Event()

    async def replay(_endpoint, _sequence, _config):
        requested.set()
        await release.wait()
        return ((0, b"a"),)

    source, index, current = setup_source(monkeypatch, replay)
    task = asyncio.create_task(source.synchronize_once())
    try:
        await asyncio.wait_for(requested.wait(), 1)
        current["lease"] = lease(19010)
        release.set()
        with pytest.raises(ProcessLookupError, match="changed during"):
            await asyncio.wait_for(task, 1)
        assert not source.ready and match(index) == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_rank_mismatch_cannot_publish_partial_replay(monkeypatch):
    async def replay(_endpoint, _sequence, _config):
        return ((0, b"a"), (1, b"rank"))

    source, index, _ = setup_source(monkeypatch, replay)
    with pytest.raises(ReplayGap, match="rank"):
        await source.synchronize_once()
    assert not source.ready and match(index) == 0
    assert source.status()["next_seq"] == 0


@pytest.mark.asyncio
async def test_lease_endpoint_reuse_is_rejected(monkeypatch):
    async def replay(_endpoint, _sequence, _config):
        return ()

    source, _, current = setup_source(monkeypatch, replay)
    await source.synchronize_once()
    current["lease"] = replace(current["lease"], generation=str(uuid4()))
    with pytest.raises(ReplayGap, match="reused"):
        await source.synchronize_once()
    assert not source.ready


@pytest.mark.parametrize(
    "frames,error",
    [
        ([(0, b"a"), (1, b"b"), (END_SEQ, b"")], None),
        ([(1, b"a"), (END_SEQ, b"")], ReplayGap),
        ([(END_SEQ, b"not-empty")], ValueError),
        ([(0, b"a"), (0, b"b")], ReplayGap),
        ([(0, b"a"), (1, b"b")], ReplayGap),
        ([], asyncio.TimeoutError),
        ([(0, b"a")], asyncio.TimeoutError),
    ],
)
@pytest.mark.parametrize("reply_delimiter", [False, True])
def test_native_router_dealer_wire_contract(frames, error, reply_delimiter):
    loop = asyncio.SelectorEventLoop()
    try:
        loop.run_until_complete(_wire_contract(frames, error, reply_delimiter))
    finally:
        loop.close()


async def _wire_contract(frames, error, reply_delimiter=False):
    import zmq
    import zmq.asyncio

    context = zmq.asyncio.Context.instance()
    server = context.socket(zmq.ROUTER)
    endpoint = f"inproc://native-wire-{uuid4().hex}"
    server.bind(endpoint)
    config = NativeReplayConfig(
        "w",
        "url",
        Path("/lease"),
        "/m",
        128,
        "a" * 64,
        timeout_s=0.2,
        max_batches=1 if len(frames) == 2 else 10,
        reply_delimiter=reply_delimiter,
    )

    async def respond():
        identity, empty, start = await server.recv_multipart()
        assert empty == b"" and start == b"\0" * 8
        for sequence, payload in frames:
            raw = (
                sequence if isinstance(sequence, bytes) else sequence.to_bytes(8, "big")
            )
            prefix = (identity, b"", b"") if reply_delimiter else (identity, b"")
            await server.send_multipart((*prefix, raw, payload))

    task = asyncio.create_task(respond())
    try:
        if error is None:
            result = await request_native_replay(endpoint, 0, config)
            assert result == ((0, b"a"), (1, b"b"))
        else:
            with pytest.raises(error):
                await request_native_replay(endpoint, 0, config)
        await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        server.close(linger=0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern,arrivals,prefixes",
    [
        ("steady", list(range(8)), [b"a", b"b"] * 4),
        ("microburst", [0, 0, 0.01, 0.01, 4, 4.01, 6, 6.01], [b"a", b"b"] * 4),
        ("hotspot", list(range(8)), [b"a"] * 8),
        ("phase-shift", list(range(8)), [b"a"] * 4 + [b"b"] * 4),
    ],
)
async def test_deterministic_arrivals_reject_dead_generation_then_recover(
    monkeypatch, pattern, arrivals, prefixes
):
    async def replay(_endpoint, _sequence, _config):
        return ((0, b"a"), (1, b"b"))

    source, index, current = setup_source(monkeypatch, replay)
    await source.synchronize_once()
    original_generation = source.status()["generation"]
    recovered = False
    receipts = []
    for request_id, (arrival, prefix) in enumerate(
        zip(arrivals, prefixes, strict=True)
    ):
        current["time"] = arrival
        if 2.5 <= arrival < 6:
            current["alive"] = False
        elif arrival >= 6 and not recovered:
            current["alive"] = True
            current["lease"] = lease(19010)
            await source.synchronize_once()
            recovered = True
        # Check trust immediately before scoring, as the HTTP service does.
        ready = source.ready
        score = match(index, prefix)
        dead = 2.5 <= arrival < 6
        receipts.append(
            {
                "pattern": pattern,
                "request_id": request_id,
                "arrival": arrival,
                "generation": source.status()["generation"],
                "ready": ready,
                "hint_tokens": score,
                # Unsafe shadow oracle, NOT a production policy or engine hit.
                "unsafe_retained_hint": 128,
                "known_dead": dead,
            }
        )
        assert ready is not dead
        assert score == (0 if dead else 128)
    assert any(r["known_dead"] and r["unsafe_retained_hint"] for r in receipts)
    assert all(r["hint_tokens"] == 0 for r in receipts if r["known_dead"])
    assert receipts[-1]["generation"] != original_generation
    assert receipts[-1]["hint_tokens"] == 128
