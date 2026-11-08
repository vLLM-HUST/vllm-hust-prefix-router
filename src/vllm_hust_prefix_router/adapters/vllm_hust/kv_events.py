"""Decode compatible vLLM-HUST KV-event and replay wire formats."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.specifiers import SpecifierSet

from vllm_hust_prefix_router.adapters.vllm_hust.request_fingerprint import (
    normalize_external_block_hash,
)
from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    CacheEventBatch,
    CacheReplayResponse,
)


class VllmHustKvEventCodec:
    """Narrow, version-checked adapter for vLLM-HUST event structs."""

    def __init__(self, host_version_range: str = ">=0.25.1,<0.30") -> None:
        try:
            installed_version = version("vllm")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "vLLM-HUST must be installed to decode its KV events"
            ) from exc
        if installed_version not in SpecifierSet(host_version_range):
            raise RuntimeError(
                f"vLLM-HUST {installed_version} is outside the tested range "
                f"{host_version_range}"
            )

        import msgspec
        from vllm.distributed import kv_events
        from vllm.distributed.kv_events import (
            AllBlocksCleared as HostAllBlocksCleared,
        )
        from vllm.distributed.kv_events import BlockRemoved as HostBlockRemoved
        from vllm.distributed.kv_events import BlockStored as HostBlockStored

        KVEventBatch = kv_events.KVEventBatch
        self._msgspec = msgspec
        self._host_stored = HostBlockStored
        self._host_removed = HostBlockRemoved
        self._host_cleared = HostAllBlocksCleared
        self._batch_decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        replay_request = getattr(kv_events, "ZmqEventReplayRequest", None)
        replay_response = getattr(kv_events, "ZmqEventReplayResponse", None)
        self.snapshot_replay = (
            replay_request is not None and replay_response is not None
        )
        if self.snapshot_replay:
            self._replay_decoder = msgspec.msgpack.Decoder(type=replay_response)
            self._replay_request = replay_request
        self._encoder = msgspec.msgpack.Encoder()

    def decode_batch(self, payload: bytes) -> CacheEventBatch:
        """Decode and normalize one host event batch."""
        batch = self._batch_decoder.decode(payload)
        return decode_host_batch(
            batch,
            stored=self._host_stored,
            removed=self._host_removed,
            cleared=self._host_cleared,
        )

    def encode_replay_request(
        self,
        *,
        publisher_epoch: str | None,
        start_seq: int,
        force_snapshot: bool,
    ) -> bytes:
        """Encode one host replay request."""
        if not self.snapshot_replay:
            raise RuntimeError("this host uses sequence-frame replay")
        request = self._replay_request(
            publisher_epoch=publisher_epoch,
            start_seq=start_seq,
            force_snapshot=force_snapshot,
        )
        return self._encoder.encode(request)

    def decode_replay_response(self, payload: bytes) -> CacheReplayResponse:
        """Decode the host replay response while preserving batch payloads."""
        if not self.snapshot_replay:
            raise RuntimeError("this host uses sequence-frame replay")
        response: Any = self._replay_decoder.decode(payload)
        return CacheReplayResponse(
            publisher_epoch=response.publisher_epoch,
            next_seq=response.next_seq,
            replayed_batches=tuple(response.replayed_batches),
            snapshot=response.snapshot,
        )


def decode_host_batch(
    batch: Any, *, stored: type, removed: type, cleared: type
) -> CacheEventBatch:
    """Normalize the common KV batch schema; transport profiles remain distinct."""
    events = []
    for event in batch.events:
        if isinstance(event, cleared):
            events.append(AllBlocksCleared())
        elif isinstance(event, stored):
            if event.block_size <= 0:
                raise ValueError("native cache block size must be positive")
            events.append(
                BlockStored(
                    tuple(normalize_external_block_hash(v) for v in event.block_hashes),
                    block_size=event.block_size,
                    group_idx=0 if event.group_idx is None else event.group_idx,
                )
            )
        elif isinstance(event, removed):
            events.append(
                BlockRemoved(
                    tuple(normalize_external_block_hash(v) for v in event.block_hashes),
                    group_idx=0 if event.group_idx is None else event.group_idx,
                )
            )
        else:
            raise TypeError(f"unsupported KV event {type(event).__name__}")
    return CacheEventBatch(tuple(events), batch.data_parallel_rank)
