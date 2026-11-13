import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vllm_hust_prefix_router._version import __version__
from vllm_hust_prefix_router.config import BackendConfig, BackendPoolConfig
from vllm_hust_prefix_router.core.lifecycle import (
    LifecycleLoadTracker,
    LifecycleRoutingConfig,
)
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.request_fingerprint import (
    PromptFingerprint,
    UnsupportedFingerprintRequest,
)
from vllm_hust_prefix_router.service.app import (
    RouterService,
    RouterServiceConfig,
    create_app,
)


class FakeFingerprinter:
    async def fingerprint(self, path, _payload):
        assert path in ("/v1/completions", "/v1/chat/completions")
        return (PromptFingerprint(32, (b"first", b"second")),)


class UnsupportedFingerprinter:
    async def fingerprint(self, _path, _payload):
        raise UnsupportedFingerprintRequest("unsupported")


class BrokenFingerprinter:
    async def fingerprint(self, _path, _payload):
        raise ValueError("unexpected fingerprint failure")


async def _upstream() -> tuple[TestServer, list[dict]]:
    received = []

    async def completion(request: web.Request) -> web.StreamResponse:
        received.append(await request.json())
        response = web.StreamResponse(
            status=200, headers={"content-type": "text/event-stream"}
        )
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"index":0,"text":"ok","finish_reason":"stop"}]}\n\n'
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/completions", completion)
    server = TestServer(app)
    await server.start_server()
    return server, received


@pytest.mark.asyncio
async def test_all_nodes_use_the_same_remote_streaming_path() -> None:
    upstreams = [await _upstream() for _ in range(4)]
    index = GlobalPrefixIndex()
    backends = []
    for node_index, (server, _) in enumerate(upstreams):
        node_id = f"node-{node_index}"
        index.register_node(node_id, hash_block_size=16)
        backends.append(
            BackendConfig(
                node_id,
                str(server.make_url("/")),
                pool=BackendPoolConfig(max_connections=128),
            )
        )
    service = RouterService(
        RouterServiceConfig(tuple(backends), default_backend="node-0"),
        planner=RoutePlanner(index, policy="prefix", block_size=16),
        fingerprinter=FakeFingerprinter(),
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    try:
        health_response = await client.get("/healthz")
        assert await health_response.json() == {
            "status": "ok",
            "extension_id": "org.vllm-hust.prefix-router",
            "version": __version__,
            "config_sha256": None,
        }
        selected = []
        for _ in range(4):
            response = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "stream": True},
            )
            assert response.status == 200
            selected.append(response.headers["x-vllm-hust-router-node"])
            assert b"[DONE]" in await response.read()
        assert set(selected) == {"node-0", "node-1", "node-2", "node-3"}
        assert [len(received) for _, received in upstreams] == [1, 1, 1, 1]
        assert [
            service.pools.pool(f"node-{index}").stats().max_connections
            for index in range(4)
        ] == [128] * 4
    finally:
        await client.close()
        for server, _ in upstreams:
            await server.close()


@pytest.mark.asyncio
async def test_lifecycle_reservation_is_released_after_stream_completion() -> None:
    upstream, _ = await _upstream()
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16)
    tracker = LifecycleLoadTracker(block_size=16)
    service = RouterService(
        RouterServiceConfig(
            (BackendConfig("node-0", str(upstream.make_url("/"))),),
            default_backend="node-0",
        ),
        planner=RoutePlanner(
            index,
            policy="lifecycle",
            block_size=16,
            lifecycle_tracker=tracker,
        ),
        fingerprinter=FakeFingerprinter(),
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/completions", json={"prompt": "hello", "stream": True}
        )
        assert response.status == 200
        await response.read()
        assert tracker.active_request_ids() == ()
        metrics_response = await client.get("/metrics")
        metrics = json.loads(await metrics_response.text())
        assert metrics["routing"] == {
            "policy": "lifecycle",
            "lifecycle": {
                "node-0": {
                    "active_prefill_tokens": 0,
                    "active_decode_blocks": 0,
                    "active_requests": 0,
                    "tracked_prefill_tokens": 0,
                    "tracked_decode_blocks": 0,
                    "tracked_output_blocks": 0,
                    "tracked_requests": 0,
                    "pending_requests": 0,
                    "accepted_requests": 0,
                    "active_decode_sequences": 0,
                }
            },
        }
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_lifecycle_dispatch_uses_selected_dp_rank(monkeypatch) -> None:
    upstream, _ = await _upstream()
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16, data_parallel_rank=0)
    tracker = LifecycleLoadTracker(block_size=16)
    dispatched = []
    original_dispatch = tracker.mark_dispatched

    def record_dispatch(request_id, worker):
        dispatched.append(worker)
        return original_dispatch(request_id, worker)

    monkeypatch.setattr(tracker, "mark_dispatched", record_dispatch)
    service = RouterService(
        RouterServiceConfig(
            (BackendConfig("node-0", str(upstream.make_url("/"))),),
            default_backend="node-0",
        ),
        planner=RoutePlanner(
            index, policy="lifecycle", block_size=16, lifecycle_tracker=tracker
        ),
        fingerprinter=FakeFingerprinter(),
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/completions", json={"prompt": "hello", "stream": True}
        )
        assert response.status == 200
        await response.read()
        assert dispatched == [("node-0", 0)]
        assert tracker.active_request_ids() == ()
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_unsupported_fingerprint_uses_explicit_default_backend() -> None:
    upstream, received = await _upstream()
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16)
    service = RouterService(
        RouterServiceConfig(
            (BackendConfig("node-0", str(upstream.make_url("/"))),),
            default_backend="node-0",
        ),
        planner=RoutePlanner(index, policy="prefix", block_size=16),
        fingerprinter=UnsupportedFingerprinter(),
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/completions", json={"prompt": "hello", "stream": True}
        )
        assert response.status == 200
        await response.read()
        assert len(received) == 1
        metrics_response = await client.get("/metrics")
        metrics = json.loads(await metrics_response.text())
        assert metrics["fingerprint_fallbacks"] == 1
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_unexpected_fingerprint_failure_does_not_silently_route() -> None:
    upstream, received = await _upstream()
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16)
    service = RouterService(
        RouterServiceConfig(
            (BackendConfig("node-0", str(upstream.make_url("/"))),),
            default_backend="node-0",
        ),
        planner=RoutePlanner(index, policy="prefix", block_size=16),
        fingerprinter=BrokenFingerprinter(),
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/completions", json={"prompt": "hello", "stream": True}
        )
        assert response.status == 500
        assert received == []
    finally:
        await client.close()
        await upstream.close()


def test_downstream_headers_preserve_content_encoding() -> None:
    headers = {
        "Content-Encoding": "gzip",
        "Content-Length": "123",
        "Connection": "close",
    }
    assert RouterService._downstream_headers(headers) == {"Content-Encoding": "gzip"}


@pytest.mark.asyncio
async def test_cancelled_dispatch_releases_lifecycle_reservation(monkeypatch) -> None:
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16)
    tracker = LifecycleLoadTracker(block_size=16)
    service = RouterService(
        RouterServiceConfig((BackendConfig("node-0", "http://127.0.0.1:1"),), "node-0"),
        planner=RoutePlanner(
            index, policy="lifecycle", block_size=16, lifecycle_tracker=tracker
        ),
        fingerprinter=FakeFingerprinter(),
    )
    entered = asyncio.Event()

    class BlockingPool:
        @asynccontextmanager
        async def request(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()
            yield

    monkeypatch.setattr(service.pools, "pool", lambda _node_id: BlockingPool())

    async def read_body():
        return b'{"prompt":"hello","stream":true}'

    request = SimpleNamespace(
        content_length=None,
        read=read_body,
        headers={},
        path="/v1/completions",
        method="POST",
        path_qs="/v1/completions",
    )
    task = asyncio.create_task(service.handle_generation(request))
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert len(tracker.active_request_ids()) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tracker.active_request_ids() == ()
    load = tracker.worker_load(("node-0", None))
    assert load.active_requests == load.tracked_requests == 0


async def _wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout=3)


@pytest.mark.asyncio
async def test_lifecycle_routes_away_from_a_real_router_admission_queue():
    upstreams = [await _upstream() for _ in range(2)]
    index = GlobalPrefixIndex()
    backends = []
    for node, (server, _) in enumerate(upstreams):
        node_id = f"node-{node}"
        index.register_node(node_id, hash_block_size=16)
        backends.append(
            BackendConfig(
                node_id,
                str(server.make_url("/")),
                pool=BackendPoolConfig(max_connections=1, queue_timeout_s=5),
            )
        )
    tracker = LifecycleLoadTracker(
        block_size=16,
        config=LifecycleRoutingConfig(router_queue_weight=1),
        random_sample=lambda: 0,
    )
    service = RouterService(
        RouterServiceConfig(tuple(backends), "node-0"),
        planner=RoutePlanner(
            index, policy="lifecycle", block_size=16, lifecycle_tracker=tracker
        ),
        fingerprinter=FakeFingerprinter(),
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    busy_pool = service.pools.pool("node-0")
    waiter = None

    async def wait_for_slot():
        async with busy_pool._admit():
            pass

    try:
        # Occupy a real pool slot and put a second task in its real admission queue.
        async with busy_pool._admit():
            waiter = asyncio.create_task(wait_for_slot())
            await _wait_until(lambda: busy_pool.stats().queued_requests == 1)
            response = await client.post(
                "/v1/completions", json={"prompt": "hello", "stream": True}
            )
            assert response.headers["x-vllm-hust-router-node"] == "node-1"
            await response.read()
            assert [len(received) for _, received in upstreams] == [0, 1]
            metrics = await (await client.get("/metrics")).json()
            assert metrics["backends"]["node-0"]["queued_requests"] == 1
        await waiter
        assert busy_pool.stats().queued_requests == 0
        assert tracker.active_request_ids() == ()
    finally:
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        await client.close()
        for server, _ in upstreams:
            await server.close()


@pytest.mark.asyncio
async def test_live_decode_choices_affect_routing_and_metrics():
    finish_first = asyncio.Event()
    finish_all = asyncio.Event()

    async def stream(request):
        assert (await request.json())["n"] == 2
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"index":0,"text":"a"},{"index":1,"text":"b"}]}\n\n'
        )
        await finish_first.wait()
        await response.write(
            b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n'
        )
        await finish_all.wait()
        await response.write(
            b'data: {"choices":[{"index":1,"finish_reason":"stop"}]}\n\n'
        )
        await response.write_eof()
        return response

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/completions", stream)
    slow = TestServer(upstream_app)
    await slow.start_server()
    fast, _ = await _upstream()
    index = GlobalPrefixIndex()
    for node in ("node-0", "node-1"):
        index.register_node(node, hash_block_size=16)
    tracker = LifecycleLoadTracker(
        block_size=16,
        config=LifecycleRoutingConfig(
            prefill_load_weight=0,
            decode_block_weight=0,
            active_decode_sequence_weight=1,
        ),
        random_sample=lambda: 0,
    )
    service = RouterService(
        RouterServiceConfig(
            (
                BackendConfig("node-0", str(slow.make_url("/"))),
                BackendConfig("node-1", str(fast.make_url("/"))),
            ),
            "node-0",
        ),
        planner=RoutePlanner(
            index, policy="lifecycle", block_size=16, lifecycle_tracker=tracker
        ),
        fingerprinter=FakeFingerprinter(),
    )
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    try:
        first = await client.post(
            "/v1/completions", json={"prompt": "hello", "stream": True, "n": 2}
        )
        assert first.headers["x-vllm-hust-router-node"] == "node-0"
        await _wait_until(
            lambda: tracker.worker_load(("node-0", None)).active_decode_sequences == 2
        )
        metrics = await (await client.get("/metrics")).json()
        assert metrics["routing"]["lifecycle"]["node-0"]["active_decode_sequences"] == 2
        finish_first.set()
        await _wait_until(
            lambda: tracker.worker_load(("node-0", None)).active_decode_sequences == 1
        )
        second = await client.post(
            "/v1/completions", json={"prompt": "hello", "stream": True}
        )
        assert second.headers["x-vllm-hust-router-node"] == "node-1"
        await second.read()
        finish_all.set()
        await first.read()
        await _wait_until(lambda: tracker.active_request_ids() == ())
        assert tracker.worker_load(("node-0", None)).active_decode_sequences == 0
    finally:
        finish_first.set()
        finish_all.set()
        await client.close()
        await slow.close()
        await fast.close()
