from vllm_hust_prefix_router.core.lifecycle import LifecycleLoadTracker
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint


def _index() -> GlobalPrefixIndex:
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16)
    index.register_node("node-1", hash_block_size=16)
    return index


def test_prefix_policy_preserves_longest_prefix_selection() -> None:
    index = _index()
    index.apply_events("node-1", [])
    planner = RoutePlanner(index, policy="prefix", block_size=16)

    decision = planner.choose(
        "request-1", (PromptFingerprint(32, (b"a", b"b")),)
    )

    assert decision is not None
    assert decision.node_id == "node-0"
    assert decision.decision_id is None


def test_lifecycle_policy_reserves_and_releases_request_load() -> None:
    tracker = LifecycleLoadTracker(block_size=16)
    planner = RoutePlanner(
        _index(),
        policy="lifecycle",
        block_size=16,
        lifecycle_tracker=tracker,
    )

    first = planner.choose(
        "request-1", (PromptFingerprint(64, (b"a", b"b", b"c", b"d")),)
    )
    second = planner.choose(
        "request-2", (PromptFingerprint(64, (b"e", b"f", b"g", b"h")),)
    )

    assert first is not None and second is not None
    assert first.node_id != second.node_id
    assert tracker.active_request_ids() == ("request-1", "request-2")
    assert planner.release("request-1")
    assert planner.release("request-2")
    assert tracker.active_request_ids() == ()


def test_lifecycle_does_not_accept_snapshot_load_input() -> None:
    planner = RoutePlanner(_index(), policy="lifecycle", block_size=16)

    assert not hasattr(planner, "routing_snapshot")
    assert not hasattr(planner, "prefill_rate")
