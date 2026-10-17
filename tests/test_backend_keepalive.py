"""An explicit upstream idle budget expires pooled sockets without POST retries."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm_hust_prefix_router.config import BackendConfig, BackendPoolConfig
from vllm_hust_prefix_router.service.backend_pool import BackendConnectionPool


@pytest.mark.asyncio
async def test_expired_idle_connections_are_replaced_without_duplicate_dispatch():
    transports, ids = [], []

    async def backend(request):
        transports.append(request.transport)
        ids.append(request.headers["x-request-id"])
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/completions", backend)
    server = TestServer(app)
    await server.start_server()
    pool = BackendConnectionPool(
        BackendConfig(
            "worker",
            str(server.make_url("/")),
            pool=BackendPoolConfig(max_connections=2, keepalive_timeout_s=0.05),
        )
    )
    await pool.start()
    try:

        async def send(request_id):
            async with pool.request(
                "POST",
                "/v1/completions",
                json={"prompt": "synthetic"},
                headers={"x-request-id": request_id},
            ) as response:
                assert response.status == 200
                assert await response.json() == {"ok": True}

        await asyncio.gather(send("first"), send("second"))
        old = tuple(transports)
        await asyncio.sleep(0.15)
        await send("after-idle")
        assert all(transports[-1] is not transport for transport in old)
        assert sorted(ids) == ["after-idle", "first", "second"]
        assert pool.stats().admitted_requests == 3
        assert pool.stats().active_requests == pool.stats().queued_requests == 0
    finally:
        await pool.close()
        await server.close()
