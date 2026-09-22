"""Decode the tested vLLM-HUST KV-event and replay wire formats."""

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

    def __init__(self, host_version_range: str = ">=0.23.1,<0.24") -> None:
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
        from vllm.distributed.kv_events import (
            AllBlocksCleared as HostAllBlocksCleared,
        )
        from vllm.distributed.kv_events import BlockRemoved as HostBlockRemoved
        from vllm.distributed.kv_events import BlockStored as HostBlockStored
        from vllm.distributed.kv_events import (
            KVEventBatch,
            ZmqEventReplayRequest,
            ZmqEventReplayResponse,
        )
        self._msgspec = msgspec
        self._host_stored = HostBlockStored
        self._host_removed = HostBlockRemoved
        self._host_cleared = HostAllBlocksCleared
        self._batch_decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        self._replay_decoder = msgspec.msgpack.Decoder(type=ZmqEventReplayResponse)
        self._replay_request = ZmqEventReplayRequest
        self._encoder = msgspec.msgpack.Encoder()

    def decode_batch(self, payload: bytes) -> CacheEventBatch:
        """Decode and normalize one host event batch."""
        batch = self._batch_decoder.decode(payload)
        events = []
        for event in batch.events:
            group_idx = 0 if event.group_idx is None else event.group_idx
            if isinstance(event, self._host_stored):
                events.append(
                    BlockStored(
                        tuple(
                            normalize_external_block_hash(value)
                            for value in event.block_hashes
                        ),
                        block_size=event.block_size,
                        group_idx=group_idx,
                    )
                )
            elif isinstance(event, self._host_removed):
                events.append(
                    BlockRemoved(
                        tuple(
                            normalize_external_block_hash(value)
                            for value in event.block_hashes
                        ),
                        group_idx=group_idx,
                    )
                )
            elif isinstance(event, self._host_cleared):
                events.append(AllBlocksCleared())
            else:
                raise TypeError(f"unsupported KV event {type(event).__name__}")
        return CacheEventBatch(
            events=tuple(events),
            data_parallel_rank=batch.data_parallel_rank,
        )

    def encode_replay_request(
        self,
        *,
        publisher_epoch: str | None,
        start_seq: int,
        force_snapshot: bool,
    ) -> bytes:
        """Encode one host replay request."""
        request = self._replay_request(
            publisher_epoch=publisher_epoch,
            start_seq=start_seq,
            force_snapshot=force_snapshot,
        )
        return self._encoder.encode(request)

    def decode_replay_response(self, payload: bytes) -> CacheReplayResponse:
        """Decode the host replay response while preserving batch payloads."""
        response: Any = self._replay_decoder.decode(payload)
        return CacheReplayResponse(
            publisher_epoch=response.publisher_epoch,
            next_seq=response.next_seq,
            replayed_batches=tuple(response.replayed_batches),
            snapshot=response.snapshot,
        )
