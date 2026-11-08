"""Explicit source-pinned native profile, separate from the PR #2 protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

CORE_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
ASCEND_COMMIT = "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd"
PROFILE = "managed-native-0fc695f-v1"
SOURCE_HASHES = {
    "distributed/kv_events.py": (
        "24841685ad6db3543a19b9639eea24e1992ab37c4773c775da9f4140bb813322"
    ),
    "v1/core/kv_cache_utils.py": (
        "b89f9afa7f95fcc30412ae2031d5370a23e95f35d068fa14e5824dce1c900ee5"
    ),
    "utils/hashing.py": (
        "89b4bd2bd246a5bfdcf00d766ea38a8182376aecfd6b6b7d5c47d96b1559243e"
    ),
}


def model_identity(model: str) -> str:
    """Fingerprint the approved dense model's tokenizer/template and config."""
    root = Path(model).resolve()
    expected_config = "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e"
    if (
        hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
        != expected_config
    ):
        raise RuntimeError("native profile model configuration is not qualified")
    names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    if (root / "special_tokens_map.json").exists():
        names.append("special_tokens_map.json")
    if (root / "added_tokens.json").exists():
        names.append("added_tokens.json")
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }
    contract = {
        "model": str(root),
        "files": files,
        "hash": "sha256",
        "block_size": 128,
        "hash_seed": "0",
        "full_byte_events": True,
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def verify_native_profile() -> None:
    """Reject an unknown build before importing its protocol-specific decoder."""
    import vllm
    import vllm_ascend

    if (
        os.getenv("PYTHONHASHSEED") != "0"
        or os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") != "0"
    ):
        raise RuntimeError("native profile requires seed 0 and full-byte event hashes")

    if version("vllm") != "0.23.0+empty" or version("vllm-ascend") != "0.23.0":
        raise RuntimeError("managed native profile requires its exact package versions")
    core_package = Path(vllm.__file__).resolve().parent
    ascend_package = Path(vllm_ascend.__file__).resolve().parent
    for package, commit in (
        (core_package, CORE_COMMIT),
        (ascend_package, ASCEND_COMMIT),
    ):
        root = package.parent
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            timeout=5,
        ).strip()
        if head != commit or dirty:
            raise RuntimeError(
                "managed native profile requires clean pinned host sources"
            )
    for relative, expected in SOURCE_HASHES.items():
        if (
            hashlib.sha256((core_package / relative).read_bytes()).hexdigest()
            != expected
        ):
            raise RuntimeError(f"native profile source mismatch: {relative}")


class NativeKvEventCodec:
    """Reuse normalized decoding while retaining a separate compatibility guard."""

    def __init__(self) -> None:
        verify_native_profile()
        import msgspec
        from vllm.distributed import kv_events

        self._host_stored = kv_events.BlockStored
        self._host_removed = kv_events.BlockRemoved
        self._host_cleared = kv_events.AllBlocksCleared
        self._batch_decoder = msgspec.msgpack.Decoder(type=kv_events.KVEventBatch)

    def decode_batch(self, payload: bytes):
        # The normalized event schema is shared; replay framing is not.
        from vllm_hust_prefix_router.adapters.vllm_hust.kv_events import (
            decode_host_batch,
        )

        return decode_host_batch(
            self._batch_decoder.decode(payload),
            stored=self._host_stored,
            removed=self._host_removed,
            cleared=self._host_cleared,
        )
