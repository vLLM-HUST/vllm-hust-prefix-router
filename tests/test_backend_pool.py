import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm_hust_prefix_router.config import BackendConfig, BackendPoolConfig
from vllm_hust_prefix_router.service.backend_pool import (
    BackendConnectionPool,
    BackendPoolRegistry,
    BackendPoolSaturated,
)


@pytest.mark.asyncio
async def test_registry_uses_an_independent_connector_per_backend() -> None:
    backends = [
        BackendConfig(node_id=f"node-{index}", url=f"http://127.0.0.1:{8100 + index}")
        for index in range(4)
    ]
    registry = BackendPoolRegistry(backends)
    await registry.start()
    try:
        sessions = [registry.pool(backend.node_id).session for backend in backends]
        assert len({id(session) for session in sessions}) == 4
        assert len({id(session.connector) for session in sessions}) == 4
        assert [session.connector.limit for session in sessions] == [512] * 4
        assert [session.connector.limit_per_host for session in sessions] == [512] * 4
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_busy_backend_does_not_consume_another_backends_capacity() -> None:
    pool_config = BackendPoolConfig(
        max_connections=1,
        max_pending_requests=0,
        queue_timeout_s=0.05,
    )
    registry = BackendPoolRegistry(
        [
            BackendConfig("node-0", "http://127.0.0.1:8100", pool=pool_config),
            BackendConfig("node-1", "http://127.0.0.1:8101", pool=pool_config),
        ]
    )

    node_0 = registry.pool("node-0")
    node_1 = registry.pool("node-1")
    async with node_0._admit():
        with pytest.raises(BackendPoolSaturated, match="queue is full"):
            async with node_0._admit():
                pass
        async with node_1._admit():
            assert node_0.stats().active_requests == 1
            assert node_1.stats().active_requests == 1


@pytest.mark.asyncio
async def test_admission_timeout_bounds_queue_delay() -> None:
    pool = BackendConnectionPool(
        BackendConfig(
            "node-0",
            "http://127.0.0.1:8100",
            pool=BackendPoolConfig(
                max_connections=1,
                max_pending_requests=1,
                queue_timeout_s=0.02,
            ),
        )
    )

    async with pool._admit():
        with pytest.raises(BackendPoolSaturated, match="admission timed out"):
            async with pool._admit():
                pass

    stats = pool.stats()
    assert stats.active_requests == 0
    assert stats.queued_requests == 0
    assert stats.admitted_requests == 1
    assert stats.rejected_requests == 1


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    pool = BackendConnectionPool(BackendConfig("node-0", "http://127.0.0.1:8100"))
    await pool.start()
    await pool.close()


@pytest.mark.asyncio
async def test_single_backend_can_hold_more_than_aiohttp_default_100() -> None:
    request_count = 110
    arrived = 0
    all_arrived = asyncio.Event()
    release = asyncio.Event()

    async def hold(_: web.Request) -> web.Response:
        nonlocal arrived
        arrived += 1
        if arrived == request_count:
            all_arrived.set()
        await release.wait()
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/hold", hold)
    server = TestServer(app)
    await server.start_server()
    pool = BackendConnectionPool(
        BackendConfig(
            "node-0",
            str(server.make_url("/")),
            pool=BackendPoolConfig(
                max_connections=128,
                max_pending_requests=0,
                queue_timeout_s=2,
            ),
        )
    )
    await pool.start()

    async def call() -> None:
        async with pool.request("GET", "/hold") as response:
            assert await response.text() == "ok"

    tasks = [asyncio.create_task(call()) for _ in range(request_count)]
    try:
        await asyncio.wait_for(all_arrived.wait(), timeout=10)
        assert pool.stats().active_requests == request_count
        assert pool.session.connector.limit == 128
    finally:
        release.set()
        await asyncio.gather(*tasks)
        await pool.close()
        await server.close()
    await pool.close()


def test_backend_pool_config_rejects_boolean_and_nonpositive_limits() -> None:
    with pytest.raises(ValueError, match="max_connections"):
        BackendPoolConfig(max_connections=True)
    with pytest.raises(ValueError, match="greater than zero"):
        BackendPoolConfig(max_connections=0)


def test_backend_rejects_local_and_non_http_targets() -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        BackendConfig("node-0", "local")
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        BackendConfig("node-0", "zmq://127.0.0.1:5557")
