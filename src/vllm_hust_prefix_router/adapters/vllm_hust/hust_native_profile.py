"""Source-checked integer-replay profile for one frozen HUST host pair.

The operator supplies fresh-engine identity. The publisher supplies neither an
epoch nor an authoritative snapshot; this adapter cannot manufacture either.
"""

from __future__ import annotations

import hashlib
import os
from importlib.metadata import distribution
from pathlib import Path

from vllm_hust_prefix_router.adapters.vllm_hust.native_profile import (
    model_identity as legacy_model_identity,
)

PROFILE = "managed-hust-cedb477-v1"
HOST_VERSION = "0.30.0+empty"
CORE_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
ASCEND_COMMIT = "b47795c63a8e63e7aed4d311f951428d94e045ab"
PACKAGE_VERSIONS = {
    "vllm": HOST_VERSION,
    "vllm-ascend": "0.25.1rc2.dev766+hust.20260903.4.gb47795c63",
}
SOURCE_HASHES = {
    "vllm": {
        "distributed/kv_events.py": (
            "de08f01e8736881560256c350a3b6418999dd4aa30f9c6ec9159527f7ed6899a"
        ),
        "v1/core/kv_cache_utils.py": (
            "2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2"
        ),
        "utils/hashing.py": (
            "89b4bd2bd246a5bfdcf00d766ea38a8182376aecfd6b6b7d5c47d96b1559243e"
        ),
        "tokenizers/registry.py": (
            "077f11649db9c6e086eeb60f9dd81ed5667cadd83a29e66a0870c1ed5b1680a5"
        ),
    },
    "vllm-ascend": {
        "worker/worker.py": (
            "23097be63048125e23d5fe7e7fcd23d7d22040e8728995cb137cf5490b63c53e"
        ),
        "utils.py": "bd7900d551ac4cb432f5794da113b664c98dc4fd771d5de0fb8e371b84ebe29c",
    },
}


def verify_hust_native_profile() -> dict[str, str]:
    """Check metadata and relevant installed bytes before host module imports.

    This detects drift in the audited interface, not the provenance of every byte
    of an engine build. Full wheel/source provenance belongs to environment setup.
    """
    if (
        os.getenv("PYTHONHASHSEED") != "0"
        or os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") != "0"
    ):
        raise RuntimeError("HUST native profile requires seed 0 and full-byte hashes")
    versions = {}
    for name, expected in PACKAGE_VERSIONS.items():
        dist = distribution(name)
        if dist.version != expected:
            raise RuntimeError(f"unqualified HUST package: {name} {dist.version}")
        versions[name] = dist.version
        root = Path(dist.locate_file(name.replace("-", "_"))).resolve()
        for relative, digest in SOURCE_HASHES[name].items():
            path = root / relative
            if (
                not path.resolve().is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise RuntimeError(f"HUST interface source mismatch: {name}/{relative}")
    return versions


def model_identity(model: str) -> str:
    """Bind model/tokenizer identity to this profile to prevent cross-profile use."""
    identity = legacy_model_identity(model)
    return hashlib.sha256(f"{PROFILE}:{identity}".encode()).hexdigest()


def normalize_hust_batch(batch, *, stored: type, removed: type, cleared: type):
    """Admit only the audited dense local cache group and full SHA-256 hashes."""
    from vllm_hust_prefix_router.protocols.cache_events import (
        AllBlocksCleared,
        BlockRemoved,
        BlockStored,
        CacheEventBatch,
    )

    if batch.data_parallel_rank != 0:
        raise ValueError("HUST native profile supports only explicit DP rank 0")
    events = []
    for event in batch.events:
        if isinstance(event, cleared):
            events.append(AllBlocksCleared())
            continue
        if not isinstance(event, (stored, removed)):
            raise TypeError("unsupported HUST cache event")
        if (
            event.medium != "GPU"
            or event.group_idx != 0
            or event.locality not in (None, "LOCAL")
            or event.ownership is not None
        ):
            raise ValueError("unqualified HUST cache tier, locality or group")
        hashes = tuple(event.block_hashes)
        if any(not isinstance(value, bytes) or len(value) != 32 for value in hashes):
            raise ValueError("HUST native profile requires 32-byte event hashes")
        if isinstance(event, stored):
            if (
                event.block_size != 128
                or event.kv_cache_spec_kind != "full_attention"
                or event.kv_cache_spec_sliding_window is not None
                or event.lora_id is not None
                or event.lora_name is not None
                or any(value is not None for value in (event.extra_keys or ()))
            ):
                raise ValueError("unqualified HUST cache spec or hash extra keys")
            events.append(BlockStored(hashes, block_size=128, group_idx=0))
        else:
            events.append(BlockRemoved(hashes, group_idx=0))
    return CacheEventBatch(tuple(events), 0)


class HustNativeKvEventCodec:
    """Decode this host's event batch, with no enhanced replay/snapshot claim."""

    def __init__(self) -> None:
        verify_hust_native_profile()
        import msgspec
        from vllm.distributed import kv_events

        self._stored = kv_events.BlockStored
        self._removed = kv_events.BlockRemoved
        self._cleared = kv_events.AllBlocksCleared
        self._decoder = msgspec.msgpack.Decoder(type=kv_events.KVEventBatch)

    def decode_batch(self, payload: bytes):
        return normalize_hust_batch(
            self._decoder.decode(payload),
            stored=self._stored,
            removed=self._removed,
            cleared=self._cleared,
        )
