# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_hust_prefix_router.core.lifecycle import (
    LifecycleLoadTracker,
    LifecycleRoutingConfig,
    PromptLoad,
    RequestLifecycleObserver,
    RequestLoad,
)


def _load(
    node_id: str,
    *,
    prompt_tokens: int = 64,
    matched_tokens: int = 0,
    block_keys: tuple[bytes, ...] = (b"a", b"b", b"c", b"d"),
) -> RequestLoad:
    return RequestLoad(
        node_id=node_id,
        data_parallel_rank=0,
        prompt_tokens=prompt_tokens,
        matched_tokens=matched_tokens,
        prompt_count=1,
        block_keys=block_keys,
    )


def test_cached_worker_wins_when_active_load_is_equal() -> None:
    tracker = LifecycleLoadTracker(block_size=16)

    selection = tracker.select_and_reserve(
        "request-1",
        [
            _load("node-a", matched_tokens=64),
            _load("node-b", matched_tokens=0),
        ],
    )

    assert selection.selected.load.node_id == "node-a"
    scores = {score.load.node_id: score.cost for score in selection.candidates}
    assert scores == {"node-a": 4.0, "node-b": 8.0}


def test_active_load_can_outweigh_a_prefix_hit() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    tracker.select_and_reserve(
        "background",
        [
            _load(
                "node-a",
                prompt_tokens=128,
                block_keys=tuple(bytes([index]) for index in range(8)),
            )
        ],
    )

    selection = tracker.select_and_reserve(
        "request-1",
        [
            _load("node-a", matched_tokens=64),
            _load("node-b", matched_tokens=0),
        ],
    )

    assert selection.selected.load.node_id == "node-b"


def test_lifecycle_transitions_preserve_decode_load_until_free() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    worker = ("node-a", 0)
    tracker.select_and_reserve(
        "request-1",
        [_load("node-a", prompt_tokens=50, matched_tokens=2)],
    )

    assert tracker.worker_load(worker).tracked_prefill_tokens == 48
    assert tracker.worker_load(worker).tracked_decode_blocks == 4
    assert tracker.worker_load(worker).tracked_requests == 1
    assert tracker.worker_load(worker).pending_requests == 1
    assert tracker.mark_dispatched("request-1", worker)
    assert tracker.worker_load(worker).pending_requests == 1
    assert tracker.mark_accepted("request-1")
    assert tracker.worker_load(worker).pending_requests == 0
    assert tracker.worker_load(worker).accepted_requests == 1
    assert tracker.mark_prefill_completed("request-1")
    assert not tracker.mark_prefill_completed("request-1")
    assert tracker.worker_load(worker).tracked_prefill_tokens == 0
    assert tracker.worker_load(worker).tracked_decode_blocks == 4
    assert tracker.free("request-1")
    assert not tracker.free("request-1")
    assert tracker.worker_load(worker).tracked_decode_blocks == 0


def test_shared_prompt_blocks_are_reference_counted() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    worker = ("node-a", 0)
    tracker.select_and_reserve("request-1", [_load("node-a")])
    tracker.select_and_reserve("request-2", [_load("node-a")])

    assert tracker.worker_load(worker).tracked_decode_blocks == 4
    tracker.free("request-1")
    assert tracker.worker_load(worker).tracked_decode_blocks == 4
    tracker.free("request-2")
    assert tracker.worker_load(worker).tracked_decode_blocks == 0


def test_dispatch_failure_can_move_the_same_reservation() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    tracker.select_and_reserve(
        "request-1",
        [_load("node-a"), _load("node-b")],
    )

    assert tracker.mark_dispatched("request-1", ("node-b", 0))
    assert tracker.worker_load(("node-a", 0)).tracked_requests == 0
    assert tracker.worker_load(("node-b", 0)).tracked_requests == 1


def test_stale_requests_expire_without_underflow() -> None:
    now = [10.0]
    tracker = LifecycleLoadTracker(
        block_size=16,
        config=LifecycleRoutingConfig(request_expiry_s=5),
        clock=lambda: now[0],
    )
    tracker.select_and_reserve("request-1", [_load("node-a")])
    now[0] = 15.0

    assert tracker.cleanup_expired() == ("request-1",)
    assert tracker.cleanup_expired() == ()
    assert tracker.worker_load(("node-a", 0)).tracked_requests == 0


def test_stream_observer_ends_prefill_on_first_nonempty_content() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    tracker.select_and_reserve("request-1", [_load("node-a")])
    observer = RequestLifecycleObserver(tracker)
    observer.bind("request-1")
    observer.dispatch(("node-a", 0))
    assert tracker.worker_load(("node-a", 0)).pending_requests == 1
    observer.observe(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    assert tracker.worker_load(("node-a", 0)).accepted_requests == 1
    observer.observe(
        {
            "type": "http.response.body",
            "body": b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            "more_body": True,
        }
    )
    assert tracker.worker_load(("node-a", 0)).tracked_prefill_tokens == 64
    observer.observe(
        {
            "type": "http.response.body",
            "body": b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            "more_body": True,
        }
    )
    assert tracker.worker_load(("node-a", 0)).tracked_prefill_tokens == 0
    assert tracker.worker_load(("node-a", 0)).tracked_decode_blocks == 4
    assert observer.finish()
    assert tracker.worker_load(("node-a", 0)).tracked_requests == 0


def test_equal_cost_prefers_the_smaller_cache_index() -> None:
    tracker = LifecycleLoadTracker(block_size=16, random_sample=lambda: 0.99)

    selection = tracker.select_and_reserve(
        "request-1",
        [
            RequestLoad(
                node_id="node-a",
                data_parallel_rank=0,
                prompt_tokens=64,
                matched_tokens=0,
                prompt_count=1,
                block_keys=(b"a", b"b", b"c", b"d"),
                cache_size_blocks=20,
            ),
            RequestLoad(
                node_id="node-b",
                data_parallel_rank=0,
                prompt_tokens=64,
                matched_tokens=0,
                prompt_count=1,
                block_keys=(b"a", b"b", b"c", b"d"),
                cache_size_blocks=10,
            ),
        ],
    )

    assert selection.selected.load.node_id == "node-b"


def test_temperature_sampling_can_choose_a_nonminimum_cost() -> None:
    tracker = LifecycleLoadTracker(
        block_size=16,
        config=LifecycleRoutingConfig(selection_temperature=1.0),
        random_sample=lambda: 0.99,
    )

    selection = tracker.select_and_reserve(
        "request-1",
        [
            _load("node-a", matched_tokens=64),
            _load("node-b", matched_tokens=0),
        ],
    )

    assert selection.selected.load.node_id == "node-b"


def test_prompt_lifecycle_releases_only_the_completed_prompt() -> None:
    shared = b"shared"
    first = PromptLoad(32, 0, (shared, b"first"))
    second = PromptLoad(48, 16, (shared, b"second", b"tail"))
    tracker = LifecycleLoadTracker(block_size=16)
    worker = ("node-a", 0)
    tracker.select_and_reserve(
        "request-1",
        [
            RequestLoad(
                node_id="node-a",
                data_parallel_rank=0,
                prompt_tokens=80,
                matched_tokens=16,
                prompt_count=2,
                block_keys=first.block_keys + second.block_keys,
                prompt_loads=(first, second),
            )
        ],
    )
    tracker.mark_accepted("request-1")

    assert tracker.mark_prefill_completed("request-1", 0)
    assert tracker.worker_load(worker).tracked_prefill_tokens == 32
    assert tracker.free_prompt("request-1", 0)
    view = tracker.worker_load(worker)
    assert view.tracked_requests == 1
    assert view.tracked_decode_blocks == 3
    assert tracker.active_request_ids() == ("request-1",)
    assert tracker.free_prompt("request-1", 1)
    assert tracker.active_request_ids() == ()
    assert tracker.worker_load(worker).tracked_decode_blocks == 0


def test_exact_output_progress_adds_and_releases_decode_blocks() -> None:
    tracker = LifecycleLoadTracker(
        block_size=4,
        config=LifecycleRoutingConfig(track_output_blocks=True),
    )
    worker = ("node-a", 0)
    tracker.select_and_reserve(
        "request-1",
        [
            RequestLoad(
                node_id="node-a",
                data_parallel_rank=0,
                prompt_tokens=3,
                matched_tokens=0,
                prompt_count=1,
                block_keys=(b"prompt",),
            )
        ],
    )

    assert tracker.record_output_tokens("request-1", 0, 1)
    assert tracker.worker_load(worker).tracked_output_blocks == 0
    assert tracker.record_output_tokens("request-1", 0, 2)
    view = tracker.worker_load(worker)
    assert view.tracked_output_blocks == 1
    assert view.tracked_decode_blocks == 2
    assert tracker.free_prompt("request-1", 0)
    assert tracker.worker_load(worker).tracked_decode_blocks == 0


def test_output_blocks_are_tracked_per_choice() -> None:
    tracker = LifecycleLoadTracker(
        block_size=4,
        config=LifecycleRoutingConfig(track_output_blocks=True),
    )
    worker = ("node-a", 0)
    tracker.select_and_reserve(
        "request-1",
        [
            RequestLoad(
                node_id="node-a",
                data_parallel_rank=0,
                prompt_tokens=4,
                matched_tokens=0,
                prompt_count=1,
                block_keys=(b"prompt",),
            )
        ],
    )

    assert tracker.record_output_tokens("request-1", 0, 1, choice_index=0)
    assert tracker.record_output_tokens("request-1", 0, 1, choice_index=1)
    view = tracker.worker_load(worker)
    assert view.tracked_output_blocks == 2
    assert view.tracked_decode_blocks == 3


def test_stream_observer_tracks_batch_prompts_independently() -> None:
    first = PromptLoad(32, 0, (b"shared", b"first"))
    second = PromptLoad(48, 16, (b"shared", b"second", b"tail"))
    tracker = LifecycleLoadTracker(block_size=16)
    worker = ("node-a", 0)
    tracker.select_and_reserve(
        "request-1",
        [
            RequestLoad(
                node_id="node-a",
                data_parallel_rank=0,
                prompt_tokens=80,
                matched_tokens=16,
                prompt_count=2,
                block_keys=first.block_keys + second.block_keys,
                prompt_loads=(first, second),
            )
        ],
    )
    observer = RequestLifecycleObserver(tracker)
    observer.bind("request-1")
    observer.dispatch(worker)
    observer.observe(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    observer.observe(
        {
            "type": "http.response.body",
            "body": (
                b'data: {"choices":[{"index":0,"text":"a","finish_reason":"stop"}]}\n\n'
            ),
            "more_body": True,
        }
    )

    view = tracker.worker_load(worker)
    assert view.tracked_prefill_tokens == 32
    assert view.tracked_requests == 1
    observer.observe(
        {
            "type": "http.response.body",
            "body": (
                b'data: {"choices":[{"index":1,"text":"b","finish_reason":"stop"}]}\n\n'
            ),
            "more_body": True,
        }
    )
    assert tracker.active_request_ids() == ()
    assert not observer.finish()


def test_stream_observer_uses_exact_delta_token_ids_for_output_blocks() -> None:
    tracker = LifecycleLoadTracker(
        block_size=4,
        config=LifecycleRoutingConfig(track_output_blocks=True),
    )
    worker = ("node-a", 0)
    tracker.select_and_reserve(
        "request-1",
        [
            RequestLoad(
                node_id="node-a",
                data_parallel_rank=0,
                prompt_tokens=4,
                matched_tokens=0,
                prompt_count=1,
                block_keys=(b"prompt",),
            )
        ],
    )
    observer = RequestLifecycleObserver(tracker)
    observer.bind("request-1")
    observer.dispatch(worker)
    observer.observe(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    observer.observe(
        {
            "type": "http.response.body",
            "body": (
                b'data: {"choices":[{"index":0,"text":"four",'
                b'"token_ids":[1,2,3,4],"finish_reason":null}]}\n\n'
            ),
            "more_body": True,
        }
    )

    view = tracker.worker_load(worker)
    assert view.tracked_output_blocks == 1
    assert view.tracked_decode_blocks == 2
    assert observer.finish()


def test_stream_observer_waits_for_every_choice_before_release() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    worker = ("node-a", 0)
    tracker.select_and_reserve("request-1", [_load("node-a")])
    observer = RequestLifecycleObserver(tracker)
    observer.bind("request-1", choices_per_prompt=2)
    observer.dispatch(worker)
    observer.observe(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    observer.observe(
        {
            "type": "http.response.body",
            "body": (
                b'data: {"choices":[{"index":0,"text":"a","finish_reason":"stop"}]}\n\n'
            ),
            "more_body": True,
        }
    )
    assert tracker.active_request_ids() == ("request-1",)

    observer.observe(
        {
            "type": "http.response.body",
            "body": (
                b'data: {"choices":[{"index":1,"text":"b","finish_reason":"stop"}]}\n\n'
            ),
            "more_body": True,
        }
    )
    assert tracker.active_request_ids() == ()


def test_shared_blocks_keep_the_same_cost_through_acceptance_and_release() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    worker = ("node-a", 0)
    for index in range(10):
        request_id = f"request-{index}"
        tracker.select_and_reserve(request_id, [_load("node-a", matched_tokens=64)])
        tracker.mark_dispatched(request_id, worker)
        tracker.mark_accepted(request_id)
        tracker.mark_prefill_completed(request_id)

    load = tracker.worker_load(worker)
    assert load.active_decode_blocks == 4
    assert load.active_requests == 10
    assert load.active_prefill_tokens == 0
    selection = tracker.select_and_reserve(
        "incoming",
        [_load("node-a", matched_tokens=64), _load("node-b")],
    )
    assert selection.selected.load.worker == worker
    assert selection.selected.incoming_decode_blocks == 0
    assert selection.selected.cost == 4
    for index in range(10):
        tracker.free(f"request-{index}")
        assert tracker.worker_load(worker).active_decode_blocks == 4
    tracker.free("incoming")
    assert tracker.worker_load(worker).active_decode_blocks == 0
    assert tracker.worker_load(worker).active_requests == 0


def test_token_output_completes_prefill_before_text_is_available() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    tracker.select_and_reserve("request-1", [_load("node-a")])
    observer = RequestLifecycleObserver(tracker)
    observer.bind("request-1")
    observer.observe(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    observer.observe(
        {
            "type": "http.response.body",
            "body": b'data: {"choices":[{"index":0,"text":"","token_ids":[42]}]}\n\n',
            "more_body": True,
        }
    )
    load = tracker.worker_load(("node-a", 0))
    assert load.active_prefill_tokens == 0
    assert load.active_decode_blocks == 4
    observer.finish()
    assert tracker.worker_load(("node-a", 0)).active_decode_blocks == 0
