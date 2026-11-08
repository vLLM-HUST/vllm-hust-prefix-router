"""Structural-copy isolation and deterministic guard-count regression tests."""

import pytest
from aiohttp import web
from test_service_reliability import setup

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    CacheEventBatch,
)
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint
from vllm_hust_prefix_router.service.app import RouterServiceConfig


@pytest.mark.parametrize("replace_cache", [False, True])
def test_replay_shadow_keeps_old_mutable_containers_isolated(replace_cache):
    index = GlobalPrefixIndex()
    previous = index.register_node(
        "worker",
        hash_block_size=16,
        data_parallel_rank=0,
        group_block_sizes={0: 16, 1: 32},
    )
    peer = index.register_node("peer", hash_block_size=16)
    index.apply_events("worker", (BlockStored((b"a",), 16),), 0)
    previous.trusted = False
    index.apply_recovery_batches(
        "worker",
        (
            CacheEventBatch((BlockStored((b"b",), 32, 1),), 0),
            CacheEventBatch((BlockRemoved((b"a",)),), 0),
        ),
        data_parallel_rank=0,
        replace_cache=replace_cache,
    )
    current = index._nodes["worker", 0]
    assert current is not previous and current.trusted
    assert current.group_block_sizes is not previous.group_block_sizes
    assert current.group_hashes is not previous.group_hashes
    assert current.group_hashes[1] == {b"b"}
    current.group_hashes[2].add(b"new-group")
    assert dict(previous.group_hashes) == {0: {b"a"}}
    assert previous.group_block_sizes == {0: 16, 1: 32}
    assert not previous.trusted
    assert index._nodes["peer", None] is peer


def test_invalid_late_rank_rolls_back_removal_clear_and_group_metadata():
    index = GlobalPrefixIndex()
    previous = index.register_node(
        "worker",
        hash_block_size=16,
        data_parallel_rank=0,
        group_block_sizes={0: 16, 1: 32},
    )
    index.apply_events("worker", (BlockStored((b"a", b"b"), 16),), 0)
    with pytest.raises(ValueError, match="configured for rank"):
        index.apply_recovery_batches(
            "worker",
            (
                CacheEventBatch((BlockRemoved((b"a",)),), 0),
                CacheEventBatch((AllBlocksCleared(), BlockStored((b"x",), 32, 1)), 0),
                CacheEventBatch((BlockStored((b"wrong-rank",), 16),), 1),
            ),
            data_parallel_rank=0,
            replace_cache=False,
        )
    assert index._nodes["worker", 0] is previous
    assert dict(previous.group_hashes) == {0: {b"a", b"b"}}
    assert previous.group_block_sizes == {0: 16, 1: 32}
    # Group 1 is empty, so a multi-group request has no reusable full prefix.
    assert index.score_nodes((b"a", b"b"), 33)[0].matched_tokens == 0


def test_replay_preserves_non_target_rank_and_empty_fast_path():
    index = GlobalPrefixIndex()
    index.register_node("worker", hash_block_size=16)
    index.apply_events("worker", (BlockStored((b"rank-zero",), 16),), 0)
    index.apply_events("worker", (BlockStored((b"rank-one",), 16),), 1)
    rank_one = index._nodes["worker", 1]
    index.apply_recovery_batches(
        "worker",
        (CacheEventBatch((BlockRemoved((b"rank-zero",)),), 0),),
        data_parallel_rank=0,
        replace_cache=False,
    )
    assert index._nodes["worker", 1].group_hashes == rank_one.group_hashes
    assert index._nodes["worker", 1].group_hashes[0] is not rank_one.group_hashes[0]
    states = dict(index._nodes)
    index.apply_recovery_batches(
        "worker", (), data_parallel_rank=0, replace_cache=False
    )
    assert all(index._nodes[key] is state for key, state in states.items())


class CountingSource:
    def __init__(self):
        from types import SimpleNamespace

        self.config = SimpleNamespace(node_id="worker")
        self.checks = 0
        self.current = True

    @property
    def ready(self):
        self.checks += 1
        return self.current

    def status(self):
        return {"ready": self.ready, "generation": "fixture-generation"}


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_soft_guard_is_deferred_but_strict_preflight_is_preserved(strict):
    async def handler(_request):
        return web.json_response({"ok": True})

    service, tracker, client, upstream = await setup(handler)
    source = CountingSource()
    service.event_sources = (source,)
    service.config = RouterServiceConfig(
        service.config.backends, "worker", require_trusted_cache_view=strict
    )
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 200
        await response.read()
        assert source.checks == (5 if strict else 4)
        assert tracker.active_request_ids() == ()
        assert service._receipts[-1]["generation_current_at_finish"] is True
    finally:
        service.event_sources = ()
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_strict_untrusted_preflight_never_fingerprints_or_dispatches():
    async def handler(_request):
        raise AssertionError("untrusted backend must not be contacted")

    class UnexpectedFingerprinter:
        async def fingerprint(self, _path, _payload):
            raise AssertionError("strict preflight must reject before fingerprint")

    service, tracker, client, upstream = await setup(handler)
    source = CountingSource()
    source.current = False
    service.event_sources = (source,)
    service.fingerprinter = UnexpectedFingerprinter()
    service.config = RouterServiceConfig(
        service.config.backends, "worker", require_trusted_cache_view=True
    )
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 503
        await response.read()
        assert source.checks == 1
        assert tracker.active_request_ids() == ()
        assert len(service._receipts) == 1
        assert service._receipts[0]["terminal"] == "http_503"
        assert service._receipts[0]["node_id"] is None
        assert service._receipts[0]["engine_cached_tokens"] is None
    finally:
        service.event_sources = ()
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_soft_expiry_during_fingerprint_discards_preexisting_cache_hint():
    import asyncio

    async def handler(_request):
        return web.json_response({"ok": True})

    service, tracker, client, upstream = await setup(handler)
    source = CountingSource()
    service.event_sources = (source,)

    class ExpiringFingerprinter:
        async def fingerprint(self, _path, _payload):
            await asyncio.sleep(0)
            source.current = False
            return (PromptFingerprint(32, (b"a", b"b")),)

    service.fingerprinter = ExpiringFingerprinter()
    try:
        service.planner.prefix_index.apply_events(
            "worker", (BlockStored((b"a", b"b"), 16),)
        )
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 200  # Ordinary HTTP fallback is still available.
        await response.read()
        receipt = service._receipts[-1]
        assert receipt["predicted_cached_tokens"] == 0
        assert receipt["engine_cached_tokens"] is None
        assert receipt["generation_current_at_finish"] is False
        assert source.checks == 4
        assert tracker.active_request_ids() == ()
    finally:
        service.event_sources = ()
        await client.close()
        await upstream.close()
