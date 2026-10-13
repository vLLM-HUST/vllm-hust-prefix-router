import pytest

from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)


def test_longest_prefix_match_uses_normalized_events() -> None:
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16, group_block_sizes={0: 16})
    index.register_node("node-1", hash_block_size=16, group_block_sizes={0: 16})
    hashes = (b"first", b"second", b"third")
    index.apply_events("node-1", [BlockStored(hashes[:2], block_size=16)])

    decision = index.choose_node(hashes, prompt_num_tokens=49)

    assert decision is not None
    assert decision.node_id == "node-1"
    assert decision.matched_tokens == 32


def test_hybrid_prefix_requires_every_configured_cache_group() -> None:
    index = GlobalPrefixIndex()
    index.register_node(
        "tp2-instance",
        hash_block_size=16,
        data_parallel_rank=0,
        group_block_sizes={0: 16, 1: 32},
    )
    index.apply_events(
        "tp2-instance",
        [BlockStored((b"a", b"b"), block_size=16, group_idx=0)],
        data_parallel_rank=0,
    )
    assert index.score_nodes((b"a", b"b"), 33)[0].matched_tokens == 0
    index.apply_events(
        "tp2-instance",
        [BlockStored((b"b",), block_size=32, group_idx=1)],
        data_parallel_rank=0,
    )
    assert index.score_nodes((b"a", b"b"), 33)[0].matched_tokens == 32


def test_configured_cache_group_layout_fails_closed_on_mismatch() -> None:
    index = GlobalPrefixIndex()
    index.register_node("tp2", hash_block_size=16, group_block_sizes={0: 16})
    with pytest.raises(ValueError, match="unexpected KV cache group"):
        index.apply_events("tp2", [BlockStored((b"a",), block_size=16, group_idx=1)])
    with pytest.raises(ValueError, match="block size changed"):
        index.apply_events("tp2", [BlockStored((b"a",), block_size=32, group_idx=0)])


def test_event_gap_invalidation_forces_zero_prefix_score() -> None:
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16, group_block_sizes={0: 16})
    index.apply_events("node-0", [BlockStored((b"first", b"second"), block_size=16)])
    assert index.score_nodes((b"first", b"second"), 33)[0].matched_tokens == 32

    index.invalidate_node("node-0")

    assert index.score_nodes((b"first", b"second"), 33)[0].matched_tokens == 0


def test_remove_and_clear_events_are_idempotent() -> None:
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16, group_block_sizes={0: 16})
    index.apply_events("node-0", [BlockStored((b"first",), block_size=16)])
    index.apply_events("node-0", [BlockRemoved((b"first",))])
    index.apply_events("node-0", [BlockRemoved((b"first",))])
    index.apply_events("node-0", [AllBlocksCleared(), AllBlocksCleared()])

    decision = index.choose_node((b"first",), prompt_num_tokens=17)
    assert decision is not None
    assert decision.matched_tokens == 0


def test_load_breaks_equal_prefix_ties() -> None:
    index = GlobalPrefixIndex()
    index.register_node("node-0", hash_block_size=16)
    index.register_node("node-1", hash_block_size=16)

    decision = index.choose_node(
        (),
        prompt_num_tokens=32,
        node_loads={"node-0": 10, "node-1": 2},
    )

    assert decision is not None
    assert decision.node_id == "node-1"
