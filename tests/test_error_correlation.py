"""Rejected attempts remain individually identifiable without a backend dispatch."""

import asyncio
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
from aiohttp import web
from test_service_reliability import setup


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["json", "unicode", "large", "strict", "internal"])
async def test_early_error_has_one_receipt_and_internal_id(case, monkeypatch):
    async def backend(_request):
        raise AssertionError("rejected request reached backend")

    service, tracker, client, upstream = await setup(backend)
    body, expected = b"invalid", 400
    service.config = replace(service.config, observe_request_timings=True)
    if case == "unicode":
        body = b'"\xff"'
    elif case == "large":
        service.config = replace(service.config, max_request_body_size=4)
        body, expected = b'{"prompt":"too large"}', 413
    elif case == "strict":
        service.config = replace(service.config, require_trusted_cache_view=True)
        service.event_sources = (
            SimpleNamespace(config=SimpleNamespace(node_id="worker"), ready=False),
        )
        body, expected = b'{"prompt":"x"}', 503
    elif case == "internal":

        async def broken(*_args):
            raise RuntimeError("private implementation detail")

        monkeypatch.setattr(service.fingerprinter, "fingerprint", broken)
        body, expected = b'{"prompt":"x"}', 500
    try:
        response = await client.post(
            "/v1/completions", data=body, headers={"x-request-id": "external"}
        )
        assert response.status == expected
        assert "private implementation detail" not in await response.text()
        internal = response.headers["x-vllm-hust-router-request-id"]
        assert internal != "external" and response.headers["x-request-id"] == "external"
        metrics = service.metrics() if case != "strict" else None
        assert service._receipt_count == 1
        receipt = service._receipts[0]
        assert receipt["request_id"] == internal and receipt["receipt_seq"] == 1
        assert (
            receipt["client_request_id_hash"] == hashlib.sha256(b"external").hexdigest()
        )
        assert receipt["node_id"] is None and receipt["engine_cached_tokens"] is None
        assert not receipt["response_committed"]
        assert service._timings[0]["request_id"] == internal
        assert tracker.active_request_ids() == ()
        if metrics:
            assert metrics["total_receipts"] == 1
    finally:
        service.event_sources = ()
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_cancel_before_headers_is_correlatable_without_returned_id(monkeypatch):
    entered = asyncio.Event()

    async def backend(_request):
        raise AssertionError("pre-dispatch cancellation reached backend")

    async def fingerprint(*_args):
        entered.set()
        await asyncio.Event().wait()

    service, tracker, client, upstream = await setup(backend)
    monkeypatch.setattr(service.fingerprinter, "fingerprint", fingerprint)
    task = asyncio.create_task(
        client.post(
            "/v1/completions",
            json={"prompt": "x"},
            headers={"x-request-id": "cancel-attempt"},
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        async def cleaned():
            while not service._receipts:
                await asyncio.sleep(0)

        await asyncio.wait_for(cleaned(), 1)
        assert len(service._receipts) == 1 and tracker.active_request_ids() == ()
        receipt = service._receipts[0]
        assert receipt["terminal"] == "cancelled" and receipt["node_id"] is None
        assert (
            receipt["client_request_id_hash"]
            == hashlib.sha256(b"cancel-attempt").hexdigest()
        )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_concurrent_errors_and_successes_are_unique_and_bounded():
    async def backend(_request):
        return web.json_response({"ok": True})

    service, tracker, client, upstream = await setup(backend)
    try:

        async def send(i):
            response = await client.post(
                "/v1/completions",
                data=b"bad" if i % 2 else b'{"prompt":"x"}',
                headers={"x-request-id": "repeated"},
            )
            await response.read()
            assert response.status == (400 if i % 2 else 200)
            return response.headers["x-vllm-hust-router-request-id"]

        ids = []
        for start in range(0, 272, 16):
            ids.extend(
                await asyncio.gather(*(send(i) for i in range(start, start + 16)))
            )
        metrics = service.metrics()
        assert len(set(ids)) == 272 and metrics["total_receipts"] == 272
        receipts = metrics["recent_receipts"]
        assert len(receipts) == 256
        assert [r["receipt_seq"] for r in receipts] == list(range(17, 273))
        assert len({r["request_id"] for r in receipts}) == 256
        assert tracker.active_request_ids() == ()
    finally:
        await client.close()
        await upstream.close()
