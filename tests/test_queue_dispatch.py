"""Observed pool queue boundaries must not outlive strict view admission."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vllm_hust_prefix_router.config import BackendConfig, BackendPoolConfig
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.cache_events import BlockStored
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint
from vllm_hust_prefix_router.service.app import (
    RouterService,
    RouterServiceConfig,
    create_app,
)


class Source:
    def __init__(self):
        self.config = type("Config", (), {"node_id": "worker"})()
        self.ready = True
        self.generation = "original"
        self.identity_digest = "model-original"
        self.next_seq = 1

    async def start(self):
        pass

    async def close(self):
        pass

    def status(self):
        return {
            "ready": self.ready,
            "generation": self.generation,
            "identity_digest": self.identity_digest,
            "next_seq": self.next_seq,
        }


class Fingerprinter:
    async def fingerprint(self, _path, _payload):
        return (PromptFingerprint(33, (b"a", b"b")),)


@asynccontextmanager
async def fixture(strict, *, queue_timeout_s=2):
    source = Source()
    entered, release = asyncio.Event(), asyncio.Event()
    hits = []

    async def backend(request):
        label = (await request.json())["prompt"]
        hits.append((label, source.ready, source.generation))
        if label == "occupier":
            entered.set()
            await release.wait()
        return web.json_response({"id": label, "choices": []})

    app = web.Application()
    app.router.add_post("/v1/completions", backend)
    upstream = TestServer(app)
    await upstream.start_server()
    index = GlobalPrefixIndex()
    index.register_node("worker", hash_block_size=16)
    index.apply_events("worker", (BlockStored((b"a", b"b"), 16),))
    service = RouterService(
        RouterServiceConfig(
            (
                BackendConfig(
                    "worker",
                    str(upstream.make_url("/")),
                    pool=BackendPoolConfig(
                        max_connections=1,
                        queue_timeout_s=queue_timeout_s,
                        request_timeout_s=5,
                    ),
                ),
            ),
            "worker",
            require_trusted_cache_view=strict,
        ),
        planner=RoutePlanner(index, policy="lifecycle", block_size=16),
        fingerprinter=Fingerprinter(),
        event_sources=(source,),
    )
    client = TestClient(TestServer(create_app(service), handler_cancellation=True))
    await client.start_server()
    tasks = []
    try:
        occupier = asyncio.create_task(
            client.post(
                "/v1/completions",
                json={"prompt": "occupier"},
            )
        )
        tasks.append(occupier)
        await asyncio.wait_for(entered.wait(), 2)
        yield service, source, client, release, hits, tasks
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()
        await upstream.close()


async def wait_queued(service):
    async def queued():
        while service.pools.pool("worker").stats().queued_requests != 1:
            await asyncio.sleep(0)

    await asyncio.wait_for(queued(), 2)


async def wait_clean(service):
    async def clean():
        while service.planner.lifecycle_tracker.active_request_ids():
            await asyncio.sleep(0)

    await asyncio.wait_for(clean(), 2)
    stats = service.pools.pool("worker").stats()
    assert stats.active_requests == stats.queued_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("change", ["unready", "generation", "identity", "watermark"])
async def test_dispatch_rechecks_selected_view_after_real_http_queue(strict, change):
    async with fixture(strict) as (service, source, client, release, hits, tasks):
        queued = asyncio.create_task(
            client.post(
                "/v1/completions",
                json={"prompt": "queued"},
            )
        )
        tasks.append(queued)
        await wait_queued(service)
        if change == "unready":
            source.ready = False
            service.planner.prefix_index.invalidate_node("worker")
        elif change == "generation":
            source.generation = "replacement"  # New generation may already be ready.
        elif change == "identity":
            source.identity_digest = "different-model"
        else:
            source.next_seq += 1  # Ordinary same-generation replay is allowed.
        release.set()
        responses = await asyncio.wait_for(asyncio.gather(*tasks), 3)
        for response in responses:
            await response.read()
        await wait_clean(service)
        rejected = strict and change != "watermark"
        assert responses[1].status == (503 if rejected else 200)
        assert len(hits) == (1 if rejected else 2)
        assert len(service._receipts) == 2
        receipt = next(
            r for r in service._receipts if r["backend_response_id"] != "occupier"
        )
        assert receipt["selection_generation"] == "original"
        assert receipt["generation"] == source.generation
        assert receipt["cache_view_current_at_dispatch"] == (change == "watermark")
        assert receipt["dispatch_attempted"] is not rejected
        assert receipt["response_committed"] is not rejected
        assert receipt["terminal"] == ("http_503" if rejected else "completed")
        assert receipt["predicted_cached_tokens"] == (
            32 if change == "watermark" else 0
        )
        if not rejected:
            assert receipt["backend_response_id"] == "queued"
        else:
            assert receipt["engine_cached_tokens"] is None


@pytest.mark.asyncio
async def test_queue_timeout_does_not_dispatch_and_cleans_reservation():
    async with fixture(True, queue_timeout_s=0.02) as (
        service,
        _source,
        client,
        release,
        hits,
        tasks,
    ):
        response = await client.post("/v1/completions", json={"prompt": "queued"})
        assert response.status == 503
        await response.read()
        receipt = service._receipts[-1]
        assert not receipt["dispatch_attempted"]
        assert receipt["cache_view_current_at_dispatch"] is None
        assert receipt["terminal"] == "admission_rejected"
        assert len(hits) == 1
        release.set()
        await (await tasks[0]).read()
        await wait_clean(service)


@pytest.mark.asyncio
async def test_disconnect_in_queue_does_not_dispatch_or_leak_reservation():
    async with fixture(True) as (service, _source, client, release, hits, tasks):
        queued = asyncio.create_task(
            client.post(
                "/v1/completions",
                json={"prompt": "queued"},
            )
        )
        tasks.append(queued)
        await wait_queued(service)
        queued.cancel()
        await asyncio.gather(queued, return_exceptions=True)

        async def cancelled():
            while not any(r["terminal"] == "cancelled" for r in service._receipts):
                await asyncio.sleep(0)

        await asyncio.wait_for(cancelled(), 2)
        receipt = next(r for r in service._receipts if r["terminal"] == "cancelled")
        assert not receipt["dispatch_attempted"]
        assert receipt["cache_view_current_at_dispatch"] is None
        assert len(hits) == 1
        release.set()
        await (await tasks[0]).read()
        await wait_clean(service)
