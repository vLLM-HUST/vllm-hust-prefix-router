"""Regressions for lifecycle ownership, observation and portable count costs."""

import asyncio
import gc
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vllm_hust_prefix_router.config import BackendConfig
from vllm_hust_prefix_router.core.lifecycle import (
    LifecycleLoadTracker,
    LifecycleRoutingConfig,
    RequestLifecycleObserver,
    RequestLoad,
)
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint
from vllm_hust_prefix_router.service.app import (
    RouterService,
    RouterServiceConfig,
    create_app,
)


def load(node="a", blocks=1, size=1152):
    return RequestLoad(
        node, 0, blocks * size, 0, 1, tuple(str(i).encode() for i in range(blocks))
    )


def service_for(tracker, backends=None):
    index = GlobalPrefixIndex()
    backends = backends or (
        BackendConfig("a", "http://localhost:1", data_parallel_rank=0),
    )
    for backend in backends:
        index.register_node(
            backend.node_id, hash_block_size=tracker.block_size, data_parallel_rank=0
        )
    return RouterService(
        RouterServiceConfig(tuple(backends), "a"),
        planner=RoutePlanner(
            index,
            policy="lifecycle",
            block_size=tracker.block_size,
            lifecycle_tracker=tracker,
        ),
        fingerprinter=object(),
    )


@pytest.mark.parametrize("polls", [0, 1, 10])
def test_service_owned_request_survives_stall_and_other_selection(polls):
    now = [0.0]
    tracker = LifecycleLoadTracker(
        1152, LifecycleRoutingConfig(request_expiry_s=5), clock=lambda: now[0]
    )
    service = service_for(tracker)
    tracker.select_and_reserve("live", [load()])
    observer = RequestLifecycleObserver(tracker)
    observer.bind("live", owns_request=True)
    now[0] = 6
    for _ in range(polls):
        service.metrics()
    assert tracker.cleanup_expired() == ()
    selected = tracker.select_and_reserve("next", [load(), load("b")])
    assert selected.selected.load.node_id == "b"
    assert tracker.touch("live")
    assert observer.finish()
    assert not observer.finish()
    tracker.free("next")
    assert tracker.active_request_ids() == ()


def test_metrics_are_read_only_and_abandoned_owner_can_expire():
    now = [0.0]
    tracker = LifecycleLoadTracker(
        1152, LifecycleRoutingConfig(request_expiry_s=5), clock=lambda: now[0]
    )
    tracker.select_and_reserve("abandoned", [load()])
    observer = RequestLifecycleObserver(tracker)
    observer.bind("abandoned", owns_request=True)
    del observer
    gc.collect()
    now[0] = 6
    assert (
        service_for(tracker).metrics()["routing"]["lifecycle"]["a"]["active_requests"]
        == 1
    )
    assert tracker.cleanup_expired() == ("abandoned",)


@pytest.mark.parametrize("size", [128, 256, 1152])
def test_token_costs_preserve_equivalent_work_ranking(size):
    config = LifecycleRoutingConfig(
        active_request_weight=4608,
        active_decode_sequence_weight=2304,
        router_queue_weight=1152,
        count_weight_unit="tokens",
        decision_history_size=2,
    )
    tracker = LifecycleLoadTracker(size, config)
    tracker.select_and_reserve("background", [load(blocks=4608 // size, size=size)])
    tracker.mark_decode_started("background", 0)
    a = load(blocks=4608 // size, size=size)
    b = load("b", blocks=4608 // size, size=size)
    result = tracker.select_and_reserve("target", [a, b])
    assert result.selected.load.node_id == "b"
    for score in result.candidates:
        expected = 2 if score.load.node_id == "a" else 1
        assert score.active_request_cost * size == 4608 * expected
        assert score.active_decode_sequence_cost * size == (
            2304 if expected == 2 else 0
        )
    assert tracker.effective_count_weights()["router_queue_weight"] == 1152 / size


def test_decision_record_is_bounded_private_and_does_not_change_sampling():
    records = []
    for retention in (0, 2):
        calls = []

        def sample(calls=calls):
            calls.append(1)
            return 0.9

        tracker = LifecycleLoadTracker(
            1152,
            LifecycleRoutingConfig(decision_history_size=retention),
            random_sample=sample,
        )
        choices = []
        for index in range(3):
            selected = tracker.select_and_reserve(str(index), [load(), load("b")])
            choices.append(selected.selected.load.node_id)
            tracker.free(str(index))
        records.append((choices, len(calls)))
        history = tracker.decision_history()
        assert len(history) == retention
        assert "block_keys" not in json.dumps(history)
        if history:
            assert history[-1]["random_sample"] == 0.9
            history[-1]["candidates"].clear()
            assert tracker.decision_history()[-1]["candidates"]
    assert records[0] == records[1]


@pytest.mark.parametrize("value", ["", "bytes", None])
def test_invalid_count_unit_rejected(value):
    with pytest.raises(ValueError, match="count_weight_unit"):
        LifecycleRoutingConfig(count_weight_unit=value)


def test_tool_function_output_starts_decode_but_role_does_not():
    tracker = LifecycleLoadTracker(1152)
    tracker.select_and_reserve("tool", [load()])
    observer = RequestLifecycleObserver(tracker)
    observer.bind("tool")
    observer.observe(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    for delta, expected in (
        ({"role": "assistant"}, 0),
        ({"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}, 1),
    ):
        payload = {"choices": [{"index": 0, "delta": delta}]}
        observer.observe(
            {
                "type": "http.response.body",
                "more_body": True,
                "body": b"data: " + json.dumps(payload).encode() + b"\n\n",
            }
        )
        assert (
            tracker.worker_load_snapshot(("a", 0)).active_decode_sequences == expected
        )
    observer.finish()


@pytest.mark.asyncio
async def test_upstream_decode_is_visible_while_downstream_write_is_blocked(
    monkeypatch,
):
    blocked, release = asyncio.Event(), asyncio.Event()
    original_write = web.StreamResponse.write

    async def slow_write(response, data):
        if response.headers.get("x-vllm-hust-router-request-id") and b'"text"' in data:
            blocked.set()
            await release.wait()
        return await original_write(response, data)

    monkeypatch.setattr(web.StreamResponse, "write", slow_write)

    async def completion(_request):
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(_request)
        await response.write(b'data: {"choices":[{"index":0,"text":"x"}]}\n\n')
        await release.wait()
        await response.write(
            b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n'
        )
        return response

    app = web.Application()
    app.router.add_post("/v1/completions", completion)
    upstream = TestServer(app)
    await upstream.start_server()
    tracker = LifecycleLoadTracker(1152)
    backend = BackendConfig("a", str(upstream.make_url("/")), data_parallel_rank=0)
    service = service_for(tracker, (backend,))

    class Fingerprinter:
        async def fingerprint(self, _path, _payload):
            return (PromptFingerprint(1152, (b"hash",)),)

    service.fingerprinter = Fingerprinter()
    client = TestClient(TestServer(create_app(service)))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/completions", json={"prompt": "x", "stream": True}
        )
        await asyncio.wait_for(blocked.wait(), 3)
        assert tracker.worker_load_snapshot(("a", 0)).active_decode_sequences == 1
        release.set()
        await response.read()
        assert tracker.active_request_ids() == ()
    finally:
        release.set()
        await client.close()
        await upstream.close()
