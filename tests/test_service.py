import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vllm_hust_prefix_router.config import BackendConfig, BackendPoolConfig
from vllm_hust_prefix_router.core.lifecycle import LifecycleLoadTracker
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
            b'data: {"choices":[{"index":0,"text":"ok",'
            b'"finish_reason":"stop"}]}\n\n'
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
    assert RouterService._downstream_headers(headers) == {
        "Content-Encoding": "gzip"
    }
