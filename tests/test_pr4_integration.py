"""Public runtime compatibility must not weaken reliability contracts."""

import asyncio
import json
from dataclasses import replace

import pytest
from aiohttp import web
from test_service_reliability import setup

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.recovery import CacheEventRecovery
from vllm_hust_prefix_router.protocols.cache_events import (
    BlockStored,
    CacheEventBatch,
    CacheReplayResponse,
)
from vllm_hust_prefix_router.service import config_loader


@pytest.fixture
def config_probe(monkeypatch, tmp_path):
    captured = []

    class Codec:
        snapshot_replay = True

        def __init__(self, _version_range):
            pass

    monkeypatch.setattr(config_loader, "VllmHustKvEventCodec", Codec)
    monkeypatch.setattr(
        config_loader,
        "VllmHustTextFingerprinter",
        lambda config: captured.append(config) or object(),
    )
    raw = {
        "hash_block_size": 16,
        "fingerprint": {"tokenizer": "fixture"},
        "backends": [
            {
                "id": "worker",
                "url": "http://127.0.0.1:8100",
                "data_parallel_rank": 0,
                "group_block_sizes": {"0": 16, "1": 32},
                "event_endpoint": "tcp://127.0.0.1:5557",
                "replay_endpoint": "tcp://127.0.0.1:5558",
            }
        ],
    }

    def load():
        path = tmp_path / "config.json"
        path.write_text(json.dumps(raw))
        return config_loader.load_runtime_config(path).service

    return raw, Codec, captured, load


@pytest.mark.parametrize("legacy", [False, True])
def test_default_public_and_explicit_legacy_profiles(config_probe, legacy):
    raw, _, captured, load = config_probe
    if legacy:
        raw["runtime_profile"] = "pr2-enhanced"
    service = load()
    assert captured[0].host_version_range == (
        ">=0.23.1,<0.24" if legacy else ">=0.25.1,<0.30"
    )
    assert captured[0].hash_algorithm == ("xxhash" if legacy else "sha256")
    state = service.planner.prefix_index._nodes["worker", 0]
    assert state.group_block_sizes == {0: 16, 1: 32}


def test_unversioned_requires_opt_in_and_reports_weaker_basis(config_probe):
    raw, codec, _, load = config_probe
    codec.snapshot_replay = False
    with pytest.raises(ValueError, match="no snapshot/epoch"):
        load()
    raw["allow_unversioned_replay"] = True
    service = load()
    assert (
        service.event_sources[0].status()["recovery_basis"] == "unversioned_best_effort"
    )
    raw["require_trusted_cache_view"] = True
    with pytest.raises(ValueError, match="non-strict admission"):
        load()


@pytest.mark.parametrize("profile", ["pr2-enhanced", "managed-hust-cedb477-v1"])
def test_unversioned_opt_in_cannot_override_other_profile(config_probe, profile):
    raw, _, captured, load = config_probe
    raw.update(runtime_profile=profile, allow_unversioned_replay=True)
    with pytest.raises(ValueError, match="best-effort replay"):
        load()
    assert captured == []


@pytest.mark.parametrize("missing", ["data_parallel_rank", "group_block_sizes"])
def test_unversioned_requires_complete_layout_config(config_probe, missing):
    raw, codec, _, load = config_probe
    codec.snapshot_replay = False
    raw["allow_unversioned_replay"] = True
    raw["backends"][0].pop(missing)
    with pytest.raises(ValueError, match="requires group_block_sizes"):
        load()


class GroupCodec:
    def decode_batch(self, payload):
        if payload == b"bad":
            raise ValueError("invalid second group payload")
        return CacheEventBatch(
            (
                BlockStored((payload,), 16, 0),
                BlockStored((payload,), 32, 1),
            ),
            0,
        )


def group_recovery(*, snapshot=True):
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
        codec=GroupCodec(),
        replay=unused,
        snapshot_replay=snapshot,
    )
    return index, recovery


def test_multi_group_snapshot_and_failed_delta_are_atomic():
    index, recovery = group_recovery()
    recovery.apply_replay(CacheReplayResponse("epoch", 1, (), b"old"))
    previous = index._nodes["worker", 0]
    assert previous.group_block_sizes == {0: 16, 1: 32}
    with pytest.raises(ValueError, match="invalid second group"):
        recovery.apply_replay(
            CacheReplayResponse("epoch", 3, ((1, b"new"), (2, b"bad")))
        )
    assert index._nodes["worker", 0] is previous
    assert previous.group_hashes == {0: {b"old"}, 1: {b"old"}}
    assert not previous.trusted and recovery.state.next_seq == 1
    recovery.apply_replay(CacheReplayResponse("epoch", 1, ()))
    assert previous.trusted


def test_unversioned_truncation_stages_replacement_before_publish():
    index, recovery = group_recovery(snapshot=False)
    recovery.apply_replay(CacheReplayResponse("unversioned", 2, ((1, b"old"),)))
    previous = index._nodes["worker", 0]
    with pytest.raises(ValueError, match="invalid second group"):
        recovery.apply_replay(
            CacheReplayResponse("unversioned", 10, ((8, b"new"), (9, b"bad")))
        )
    assert index._nodes["worker", 0] is previous
    assert previous.group_hashes == {0: {b"old"}, 1: {b"old"}}
    assert recovery.state.next_seq == 2 and not previous.trusted
    recovery.apply_replay(CacheReplayResponse("unversioned", 9, ((8, b"new"),)))
    assert index._nodes["worker", 0].group_hashes == {0: {b"new"}, 1: {b"new"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_headers",
    [
        {},
        {"x-data-parallel-rank": "98"},
        {"X-Data-Parallel-Rank": "98"},
        {"x-data-parallel-rank": "98", "X-Data-Parallel-Rank": "97"},
    ],
)
async def test_selected_rank_matches_forwarding_accounting_and_receipt(backend_headers):
    entered, release = asyncio.Event(), asyncio.Event()
    ranks = []

    async def handler(request):
        ranks.append(request.headers.getall("x-data-parallel-rank", []))
        assert request.headers["x-test-backend"] == "configured"
        entered.set()
        await release.wait()
        return web.json_response({"choices": []})

    service, tracker, client, upstream = await setup(handler)
    pool = service.pools.pool("worker")
    pool.config = replace(
        pool.config, headers={"x-test-backend": "configured", **backend_headers}
    )
    service.planner.prefix_index.apply_event_batch(
        "worker", CacheEventBatch((BlockStored((b"a",), 16),), 2)
    )
    request = asyncio.create_task(
        client.post(
            "/v1/completions",
            json={"prompt": "x"},
            headers={"X-Data-Parallel-Rank": "99", "x-test-backend": "untrusted"},
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert ranks == [["2"]]
        assert tracker.worker_load(("worker", 2)).active_requests == 1
        assert tracker.worker_load(("worker", None)).active_requests == 0
        release.set()
        response = await request
        assert response.status == 200
        await response.read()
        assert service.metrics()["recent_receipts"][-1]["data_parallel_rank"] == 2
        assert tracker.active_request_ids() == ()
    finally:
        release.set()
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        await client.close()
        await upstream.close()
