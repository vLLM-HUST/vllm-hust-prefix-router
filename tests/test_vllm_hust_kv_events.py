from types import SimpleNamespace

from vllm_hust_prefix_router.adapters.vllm_hust.kv_events import (
    VllmHustKvEventCodec,
)
from vllm_hust_prefix_router.protocols.cache_events import AllBlocksCleared


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
