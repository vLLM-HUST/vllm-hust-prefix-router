import sys
from types import ModuleType, SimpleNamespace

import msgspec
import pytest

from vllm_hust_prefix_router.adapters.vllm_hust.kv_events import (
    VllmHustKvEventCodec,
)
from vllm_hust_prefix_router.protocols.cache_events import AllBlocksCleared


def test_current_batch_adapter_without_snapshot_symbols(monkeypatch) -> None:
    class HostStored(msgspec.Struct, tag=True):
        block_hashes: list[bytes]
        block_size: int
        group_idx: int | None = None

    class HostRemoved(msgspec.Struct, tag=True):
        block_hashes: list[bytes]
        group_idx: int | None = None

    class HostCleared(msgspec.Struct, tag=True):
        pass

    class HostBatch(msgspec.Struct, array_like=True):
        ts: float
        events: list[HostStored | HostRemoved | HostCleared]
        data_parallel_rank: int | None = None

    host = ModuleType("vllm")
    distributed = ModuleType("vllm.distributed")
    events = ModuleType("vllm.distributed.kv_events")
    events.BlockStored = HostStored
    events.BlockRemoved = HostRemoved
    events.AllBlocksCleared = HostCleared
    events.KVEventBatch = HostBatch
    distributed.kv_events = events
    monkeypatch.setitem(sys.modules, "vllm", host)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.distributed.kv_events", events)
    monkeypatch.setattr(
        "vllm_hust_prefix_router.adapters.vllm_hust.kv_events.version",
        lambda _name: "0.29.1",
    )

    codec = VllmHustKvEventCodec()
    assert not codec.snapshot_replay
    payload = msgspec.msgpack.encode(HostBatch(0.0, [HostStored([b"a"], 16, 1)], 0))
    batch = codec.decode_batch(payload)
    assert batch.data_parallel_rank == 0
    assert batch.events[0].group_idx == 1
    with pytest.raises(RuntimeError, match="sequence-frame"):
        codec.encode_replay_request(
            publisher_epoch=None, start_seq=0, force_snapshot=False
        )


class _HostStored:
    pass


class _HostRemoved:
    pass


class _HostCleared:
    pass


class _Decoder:
    def decode(self, payload):
        assert payload == b"snapshot"
        return SimpleNamespace(
            events=[_HostCleared()],
            data_parallel_rank=0,
        )


def test_decode_snapshot_clear_without_group_index() -> None:
    codec = object.__new__(VllmHustKvEventCodec)
    codec._host_stored = _HostStored
    codec._host_removed = _HostRemoved
    codec._host_cleared = _HostCleared
    codec._batch_decoder = _Decoder()

    batch = codec.decode_batch(b"snapshot")

    assert batch.events == (AllBlocksCleared(),)
    assert batch.data_parallel_rank == 0
