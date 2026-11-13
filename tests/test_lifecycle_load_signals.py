import json

import pytest

from vllm_hust_prefix_router.core.lifecycle import (
    LifecycleLoadTracker,
    LifecycleRoutingConfig,
    PromptLoad,
    RequestLifecycleObserver,
    RequestLoad,
)
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint


def _load(node_id, *, matched_tokens=64, queue=0):
    return RequestLoad(
        node_id=node_id,
        data_parallel_rank=0,
        prompt_tokens=64,
        matched_tokens=matched_tokens,
        prompt_count=1,
        block_keys=(b"a", b"b", b"c", b"d"),
        router_queue_requests=queue,
    )


def _observer(tracker, *, choices=1, streaming=True):
    observer = RequestLifecycleObserver(tracker)
    observer.bind("request", choices_per_prompt=choices)
    observer.observe(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")] if streaming else [],
        }
    )
    return observer


def _emit(observer, *choices):
    body = b"data: " + json.dumps({"choices": choices}).encode() + b"\n\n"
    # Network fragments need not align with SSE frames.
    for chunk in (body[:9], body[9:]):
        observer.observe(
            {"type": "http.response.body", "body": chunk, "more_body": True}
        )


def test_decode_choices_start_and_finish_independently():
    tracker = LifecycleLoadTracker(block_size=16)
    worker = ("a", 0)
    tracker.select_and_reserve("request", [_load("a")])
    observer = _observer(tracker, choices=2)
    _emit(observer, {"index": 0, "delta": {"role": "assistant"}})
    assert tracker.worker_load(worker).active_decode_sequences == 0
    _emit(observer, {"index": 0, "delta": {"reasoning_content": "thinking"}})
    _emit(observer, {"index": 0, "delta": {"content": "again"}})
    assert tracker.worker_load(worker).active_decode_sequences == 1
    _emit(observer, {"index": 1, "token_ids": [42]})
    assert tracker.worker_load(worker).active_decode_sequences == 2
    _emit(observer, {"index": 0, "finish_reason": "stop"})
    assert tracker.worker_load(worker).active_decode_sequences == 1
    assert tracker.worker_load(worker).active_requests == 1
    # Late and duplicate frames must not resurrect a finished choice.
    _emit(observer, {"index": 0, "text": "late"})
    _emit(observer, {"index": 0, "finish_reason": "stop"})
    assert not tracker.mark_decode_started("request", 0, 0)
    assert not tracker.record_output_tokens("request", 0, 2, 0)
    assert tracker.worker_load(worker).active_decode_sequences == 1
    _emit(observer, {"index": 1, "text": "last", "finish_reason": "length"})
    assert tracker.worker_load(worker).active_decode_sequences == 0
    assert tracker.active_request_ids() == ()
    assert not observer.finish()


def test_decode_counts_are_independent_for_batch_prompts():
    prompts = (PromptLoad(16, 0, (b"a",)), PromptLoad(16, 0, (b"b",)))
    tracker = LifecycleLoadTracker(block_size=16)
    tracker.select_and_reserve(
        "request",
        [RequestLoad("a", 0, 32, 0, 2, (b"a", b"b"), prompt_loads=prompts)],
    )
    observer = _observer(tracker, choices=2)
    _emit(observer, {"index": 0, "text": "a"}, {"index": 3, "text": "b"})
    assert tracker.worker_load(("a", 0)).active_decode_sequences == 2
    assert tracker.free_prompt("request", 0)
    assert tracker.worker_load(("a", 0)).active_decode_sequences == 1
    assert tracker.worker_load(("a", 0)).active_requests == 1
    observer.finish()
    assert tracker.worker_load(("a", 0)).active_decode_sequences == 0


def test_invalid_choices_and_nonstreaming_responses_do_not_invent_decode_load():
    tracker = LifecycleLoadTracker(block_size=16)
    tracker.select_and_reserve("request", [_load("a")])
    observer = _observer(tracker)
    for index in (-1, 1, True, "0"):
        _emit(observer, {"index": index, "text": "invalid"})
    assert tracker.worker_load(("a", 0)).active_decode_sequences == 0
    assert tracker.worker_load(("a", 0)).active_requests == 1
    observer.finish()

    tracker.select_and_reserve("request", [_load("a")])
    observer = _observer(tracker, streaming=False)
    observer.observe(
        {"type": "http.response.body", "body": b'{"choices":[]}', "more_body": False}
    )
    assert tracker.worker_load(("a", 0)).active_decode_sequences == 0
    observer.finish()


def test_decode_load_is_released_on_abort_and_expiry():
    now = [0.0]
    tracker = LifecycleLoadTracker(
        block_size=16,
        config=LifecycleRoutingConfig(request_expiry_s=5),
        clock=lambda: now[0],
    )
    tracker.select_and_reserve("request", [_load("a")])
    observer = _observer(tracker)
    _emit(observer, {"index": 0, "text": "a"})
    assert tracker.worker_load(("a", 0)).active_decode_sequences == 1
    assert observer.finish()
    assert tracker.worker_load(("a", 0)).active_decode_sequences == 0
    tracker.select_and_reserve("request", [_load("a")])
    assert tracker.mark_decode_started("request", 0)
    now[0] = 5
    assert tracker.cleanup_expired() == ("request",)
    assert tracker.worker_load(("a", 0)).active_decode_sequences == 0


@pytest.mark.parametrize("weight,expected", [(0.0, "a"), (1.0, "b")])
def test_decode_cost_distinguishes_sequences_sharing_the_same_blocks(weight, expected):
    tracker = LifecycleLoadTracker(
        block_size=16,
        config=LifecycleRoutingConfig(active_decode_sequence_weight=weight),
    )
    for index in range(10):
        request_id = f"background-{index}"
        tracker.select_and_reserve(request_id, [_load("a")])
        tracker.mark_decode_started(request_id, 0)
    assert tracker.worker_load(("a", 0)).active_decode_blocks == 4
    selection = tracker.select_and_reserve(
        "new", [_load("a"), _load("b", matched_tokens=0)]
    )
    assert selection.selected.load.node_id == expected
    scores = {score.load.node_id: score for score in selection.candidates}
    assert scores["a"].active_decode_sequences == 10
    assert scores["a"].active_decode_sequence_cost == 10 * weight
    assert scores["a"].cost == 4 + 10 * weight
    assert scores["b"].cost == 8


@pytest.mark.parametrize("weight,expected", [(0.0, "a"), (2.0, "b")])
def test_queue_cost_can_outweigh_prefix_benefit(weight, expected):
    tracker = LifecycleLoadTracker(
        block_size=16, config=LifecycleRoutingConfig(router_queue_weight=weight)
    )
    selection = tracker.select_and_reserve(
        "new", [_load("a", queue=3), _load("b", matched_tokens=0)]
    )
    assert selection.selected.load.node_id == expected
    score = selection.candidates[0]
    assert score.router_queue_requests == 3
    assert score.router_queue_cost == 3 * weight
    assert score.cost == 4 + 3 * weight


def test_queue_snapshot_is_refreshed_and_scoped_by_backend_id():
    index = GlobalPrefixIndex()
    index.register_node("a", hash_block_size=16, data_parallel_rank=0)
    index.register_node("a", hash_block_size=16, data_parallel_rank=1)
    index.register_node("b", hash_block_size=16)
    tracker = LifecycleLoadTracker(
        block_size=16,
        config=LifecycleRoutingConfig(router_queue_weight=1),
        random_sample=lambda: 0,
    )
    planner = RoutePlanner(
        index, policy="lifecycle", block_size=16, lifecycle_tracker=tracker
    )
    prompts = (PromptFingerprint(32, (b"a", b"b")),)
    first = planner.choose("first", prompts, router_queue_requests={"a": 2})
    assert first.node_id == "b"
    planner.release("first")
    # A cleared queue must not linger in a tracker-owned load snapshot.
    second = planner.choose("second", prompts)
    assert second.node_id == "a"
    planner.release("second")


@pytest.mark.parametrize(
    "name", ["active_decode_sequence_weight", "router_queue_weight"]
)
@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "1"])
def test_new_weights_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        LifecycleRoutingConfig(**{name: value})


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_queue_count_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="router_queue_requests"):
        _load("a", queue=value)
