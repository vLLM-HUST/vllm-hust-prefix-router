"""Request fingerprint contract shared by policy and framework adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vllm_hust_prefix_router.protocols.cache_events import BlockHash


class UnsupportedFingerprintRequest(ValueError):
    """The request cannot be hashed without changing worker semantics."""


@dataclass(frozen=True)
class PromptFingerprint:
    """Token count and chained full-block hashes for one rendered prompt."""

    num_tokens: int
    block_hashes: tuple[BlockHash, ...]

    def __post_init__(self) -> None:
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")


class RequestFingerprinter(Protocol):
    """Convert a supported OpenAI request into exact worker-compatible hashes."""

    async def fingerprint(
        self, path: str, payload: dict[str, object]
    ) -> tuple[PromptFingerprint, ...]: ...
