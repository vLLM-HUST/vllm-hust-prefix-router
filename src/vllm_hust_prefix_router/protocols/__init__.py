"""Stable data contracts used by the router core."""

from vllm_hust_prefix_router.protocols.cache_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    CacheEventBatch,
)
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint

__all__ = [
    "AllBlocksCleared",
    "BlockRemoved",
    "BlockStored",
    "CacheEventBatch",
    "PromptFingerprint",
]
