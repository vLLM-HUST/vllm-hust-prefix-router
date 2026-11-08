"""Observable HTTP failures and lifecycle cleanup under concurrency."""

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest
from aiohttp import ClientPayloadError, web
from aiohttp.test_utils import TestClient, TestServer

from vllm_hust_prefix_router.config import BackendConfig
from vllm_hust_prefix_router.core.lifecycle import LifecycleLoadTracker
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint
from vllm_hust_prefix_router.service.app import (
    RouterService,
    RouterServiceConfig,
    create_app,
)


class Fingerprinter:
    async def fingerprint(self, _path, _payload):
        return (PromptFingerprint(32, (b"a", b"b")),)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, asyncio.TimeoutError])
async def test_precommit_timeout_is_gateway_error_and_releases_load(monkeypatch, error):
    async def handler(_request):
        raise AssertionError("injected timeout must not reach backend")

    service, tracker, client, upstream = await setup(handler)

    def timeout(*_args, **_kwargs):
        raise error("injected transport timeout")

    monkeypatch.setattr(service.pools.pool("worker").session, "request", timeout)
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 502
        assert tracker.active_request_ids() == ()
        assert service.pools.pool("worker").stats().active_requests == 0
        assert service._receipts[-1]["terminal"] == "transport_error"
    finally:
        await client.close()
        await upstream.close()


async def setup(handler):
    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/completions", handler)
    upstream = TestServer(upstream_app)
    await upstream.start_server()
    index = GlobalPrefixIndex()
    index.register_node("worker", hash_block_size=16)
    tracker = LifecycleLoadTracker(block_size=16)
    service = RouterService(
        RouterServiceConfig(
            (BackendConfig("worker", str(upstream.make_url("/"))),), "worker"
        ),
        planner=RoutePlanner(
            index, policy="lifecycle", block_size=16, lifecycle_tracker=tracker
        ),
        fingerprinter=Fingerprinter(),
    )
    client = TestClient(TestServer(create_app(service), handler_cancellation=True))
    await client.start_server()
    return service, tracker, client, upstream


@pytest.mark.asyncio
async def test_repeated_external_ids_have_independent_reservations():
    entered, release = asyncio.Event(), asyncio.Event()
    arrived = 0

    async def handler(_request):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            entered.set()
        await release.wait()
        return web.json_response(
            {
                "usage": {
                    "prompt_tokens": 32,
                    "prompt_tokens_details": {"cached_tokens": 16},
                }
            }
        )

    service, tracker, client, upstream = await setup(handler)
    requests = [
        asyncio.create_task(
            client.post(
                "/v1/completions",
                json={"prompt": "x"},
                headers={"x-request-id": "same"},
            )
        )
        for _ in range(2)
    ]
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert len(tracker.active_request_ids()) == 2
        release.set()
        responses = await asyncio.wait_for(asyncio.gather(*requests), 1)
        for response in responses:
            assert response.headers["x-request-id"] == "same"
            await response.read()
        assert (
            len(
                {
                    response.headers["x-vllm-hust-router-request-id"]
                    for response in responses
                }
            )
            == 2
        )
        assert tracker.active_request_ids() == ()
        assert len(service.metrics()["recent_receipts"]) == 2
        assert all(
            receipt["engine_cached_tokens"] == 16
            for receipt in service.metrics()["recent_receipts"]
        )
    finally:
        release.set()
        for task in requests:
            task.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_observer_dispatch_error_still_releases_reservation(monkeypatch):
    async def handler(_request):
        raise AssertionError("must not reach backend")

    service, tracker, client, upstream = await setup(handler)

    class BrokenObserver:
        def dispatch(self, _worker):
            raise ValueError("injected dispatch failure")

        def finish(self):
            raise ValueError("injected cleanup failure")

    monkeypatch.setattr(service, "_lifecycle_observer", lambda *_args: BrokenObserver())
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 500
        await response.read()
        assert tracker.active_request_ids() == ()
        assert service.metrics()["observer_errors"] == 1
        assert service.metrics()["recent_receipts"][0]["terminal"] == "router_error"
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_upstream_break_after_headers_cannot_generate_second_response():
    async def handler(request):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"id":"broken-upstream"}\n\n')
        request.transport.close()
        return response

    service, tracker, client, upstream = await setup(handler)
    try:
        response = await client.post(
            "/v1/completions", json={"prompt": "x", "stream": True}
        )
        assert response.status == 200
        with pytest.raises(ClientPayloadError):
            await response.read()
        assert tracker.active_request_ids() == ()
        receipts = service.metrics()["recent_receipts"]
        assert len(receipts) == 1
        assert receipts[0]["terminal"] == "transport_error"
        assert receipts[0]["backend_response_id"] == "broken-upstream"
        assert receipts[0]["engine_cached_tokens"] is None
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_client_disconnect_cancels_handler_and_releases_load():
    release = asyncio.Event()

    async def handler(request):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"id":"cancelled-upstream"}\n\n')
        await release.wait()
        return response

    service, tracker, client, upstream = await setup(handler)

    async def cleaned():
        while (
            tracker.active_request_ids()
            or service.pools.pool("worker").stats().active_requests
        ):
            await asyncio.sleep(0)

    try:
        response = await client.post(
            "/v1/completions", json={"prompt": "x", "stream": True}
        )
        await response.content.readuntil(b"\n\n")
        response.close()
        await asyncio.wait_for(cleaned(), 1)
        assert len(service.metrics()["recent_receipts"]) == 1
        assert service.metrics()["recent_receipts"][0]["terminal"] == "cancelled"
        assert (
            service.metrics()["recent_receipts"][0]["backend_response_id"]
            == "cancelled-upstream"
        )
        assert service.metrics()["recent_receipts"][0]["engine_cached_tokens"] is None
    finally:
        release.set()
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_cancellation_at_downstream_write_retains_upstream_identity(monkeypatch):
    release = asyncio.Event()

    async def handler(request):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"id":"observed-before-write"}\n\n')
        await release.wait()
        return response

    original_write = web.StreamResponse.write

    async def cancel_after_write(response, data):
        await original_write(response, data)
        if "x-vllm-hust-router-request-id" in response.headers:
            raise asyncio.CancelledError

    monkeypatch.setattr(web.StreamResponse, "write", cancel_after_write)
    service, tracker, client, upstream = await setup(handler)
    try:
        response = await client.post(
            "/v1/completions", json={"prompt": "x", "stream": True}
        )
        with suppress(ClientPayloadError, TimeoutError):
            await asyncio.wait_for(response.read(), 1)
        receipt = service.metrics()["recent_receipts"][0]
        assert receipt["terminal"] == "cancelled"
        assert receipt["backend_response_id"] == "observed-before-write"
        assert receipt["engine_cached_tokens"] is None
        assert tracker.active_request_ids() == ()
        assert service.pools.pool("worker").stats().active_requests == 0
    finally:
        release.set()
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_view_expires_during_fingerprinting_before_strict_selection():
    async def handler(_request):
        raise AssertionError("untrusted backend must not be contacted")

    service, tracker, client, upstream = await setup(handler)
    source = SimpleNamespace(config=SimpleNamespace(node_id="worker"), ready=True)
    service.event_sources = (source,)
    service.config = RouterServiceConfig(
        service.config.backends, "worker", require_trusted_cache_view=True
    )

    class ExpiringFingerprinter:
        async def fingerprint(self, _path, _payload):
            await asyncio.sleep(0)
            source.ready = False
            return (PromptFingerprint(32, (b"a", b"b")),)

    service.fingerprinter = ExpiringFingerprinter()
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 503
        assert tracker.active_request_ids() == ()
    finally:
        service.event_sources = ()
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_diagnostic_failure_does_not_leak_request_or_replace_response():
    async def handler(_request):
        return web.json_response({"ok": True})

    service, tracker, client, upstream = await setup(handler)

    def broken_status():
        raise ValueError("injected diagnostic failure")

    service.event_sources = (
        SimpleNamespace(
            config=SimpleNamespace(node_id="worker"), ready=True, status=broken_status
        ),
    )
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 200
        assert await response.json() == {"ok": True}
        assert tracker.active_request_ids() == ()
        assert service._receipts[-1]["terminal"] == "completed"
        assert service._receipts[-1]["generation_current_at_finish"] is None
    finally:
        service.event_sources = ()
        await client.close()
        await upstream.close()
