"""Diagnostics cannot replace correctness checks or client token timestamps."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from aiohttp import web
from test_service_reliability import setup

from vllm_hust_prefix_router.config import BackendConfig
from vllm_hust_prefix_router.service.app import RouterServiceConfig
from vllm_hust_prefix_router.service.timing import LoopLagProbe, RequestTiming


def test_timestamps_are_relative_and_first_mark_is_retained():
    ticks = iter((100, 120, 140))
    timing = RequestTiming(clock=lambda: next(ticks))
    timing.mark("first_upstream_body")
    timing.mark("first_upstream_body")
    timing.mark("cleanup_end")
    assert timing.snapshot()["elapsed_ns"] == {
        "entry": 0,
        "first_upstream_body": 20,
        "cleanup_end": 40,
    }
    assert timing.snapshot()["first_body_is_first_token"] is False


@pytest.mark.parametrize("value", [1, "true", None])
def test_observation_config_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="observe_request_timings"):
        RouterServiceConfig(
            (BackendConfig("worker", "http://127.0.0.1:1234"),),
            "worker",
            observe_request_timings=value,
        )


@pytest.mark.asyncio
async def test_loop_probe_is_idempotent_and_closes():
    probe = LoopLagProbe()
    probe.start()
    task = probe._task
    probe.start()
    assert task is probe._task
    await probe.close()
    await probe.close()
    assert task.done() and not probe.snapshot()["running"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_http_observation_is_opt_in_and_keeps_response(enabled):
    async def handler(_request):
        return web.json_response({"ok": True})

    service, tracker, client, upstream = await setup(handler)
    service.config = replace(service.config, observe_request_timings=enabled)
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 200 and await response.json() == {"ok": True}
        assert tracker.active_request_ids() == ()
        diagnostic = service.metrics()["timing_diagnostics"]
        assert service._timings.maxlen == 256
        assert diagnostic["total_records"] == int(enabled)
        if enabled:
            record = diagnostic["recent_requests"][0]
            assert (
                record["request_id"]
                == response.headers["x-vllm-hust-router-request-id"]
            )
            assert record["terminal"] == "completed"
            events = record["elapsed_ns"]
            assert list(events.values()) == sorted(events.values())
            assert events["admission_start"] <= events["pool_admitted"]
            assert events["first_downstream_body"] <= events["cleanup_end"]
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_bad_input_and_fingerprint_error_are_not_dropped(monkeypatch):
    async def handler(_request):
        raise AssertionError("must not dispatch")

    service, tracker, client, upstream = await setup(handler)
    service.config = replace(service.config, observe_request_timings=True)

    async def broken(*_args):
        raise RuntimeError("injected fingerprint error")

    monkeypatch.setattr(service.fingerprinter, "fingerprint", broken)
    try:
        for body, status in (("bad", 400), ('{"prompt":"x"}', 500)):
            response = await client.post("/v1/completions", data=body)
            assert response.status == status
            await response.read()
        assert tracker.active_request_ids() == ()
        records = service.metrics()["timing_diagnostics"]["recent_requests"]
        assert [r["terminal"] for r in records] == ["http_400", "router_error"]
        assert all("handler_end" in r["elapsed_ns"] for r in records)
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_disabled_observation_does_not_allocate_timer(monkeypatch):
    async def handler(_request):
        return web.json_response({"ok": True})

    service, _tracker, client, upstream = await setup(handler)

    def forbidden():
        raise AssertionError("disabled observation allocated a clock")

    monkeypatch.setattr("vllm_hust_prefix_router.service.app.RequestTiming", forbidden)
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 200
        await response.read()
        assert not service._loop_lag.snapshot()["running"]
    finally:
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_strict_rejection_retains_diagnostic_without_dispatch():
    async def handler(_request):
        raise AssertionError("untrusted view must not dispatch")

    service, tracker, client, upstream = await setup(handler)
    service.config = replace(
        service.config, observe_request_timings=True, require_trusted_cache_view=True
    )
    service.event_sources = (
        SimpleNamespace(
            config=SimpleNamespace(node_id="worker"),
            ready=False,
        ),
    )
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        assert response.status == 503
        await response.read()
        record = service._timings[-1]
        assert record["terminal"] == "http_503"
        assert "strict_eligibility_end" in record["elapsed_ns"]
        assert "pool_admitted" not in record["elapsed_ns"]
        assert tracker.active_request_ids() == ()
    finally:
        service.event_sources = ()
        await client.close()
        await upstream.close()


@pytest.mark.asyncio
async def test_stream_cancel_retains_timing_and_releases_resources():
    release = asyncio.Event()

    async def handler(request):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        await release.wait()
        return response

    service, tracker, client, upstream = await setup(handler)
    service.config = replace(service.config, observe_request_timings=True)
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        await response.content.readuntil(b"\n\n")
        response.close()

        async def cleaned():
            while not service._timings:
                await asyncio.sleep(0)

        await asyncio.wait_for(cleaned(), 1)
        assert service._timings[-1]["terminal"] == "cancelled"
        assert "cleanup_end" in service._timings[-1]["elapsed_ns"]
        assert tracker.active_request_ids() == ()
        assert service.pools.pool("worker").stats().active_requests == 0
    finally:
        release.set()
        await client.close()
        await upstream.close()
