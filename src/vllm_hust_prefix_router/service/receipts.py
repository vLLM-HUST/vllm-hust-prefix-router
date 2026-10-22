"""Bounded observation of backend-reported usage; never infer unknown hits."""

from __future__ import annotations

import hashlib
import json

from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint


def fingerprint_digest(prompts: tuple[PromptFingerprint, ...]) -> str:
    """Associate existing normalized hashes with evidence, not engine admission."""
    digest = hashlib.sha256(b"prefix-router-request-fingerprint-v1\0")
    digest.update(len(prompts).to_bytes(8, "big"))
    for prompt in prompts:
        count = str(prompt.num_tokens).encode("ascii")
        digest.update(len(count).to_bytes(8, "big"))
        digest.update(count)
        digest.update(len(prompt.block_hashes).to_bytes(8, "big"))
        for block_hash in prompt.block_hashes:
            digest.update(len(block_hash).to_bytes(8, "big"))
            digest.update(block_hash)
    return digest.hexdigest()


class UsageObserver:
    """Extract usage from JSON or SSE while bounding response buffering."""

    def __init__(self, *, streaming: bool, max_bytes: int = 256 * 1024) -> None:
        self.streaming, self.max_bytes = streaming, max_bytes
        self.cached_tokens: int | None = None
        self.prompt_tokens: int | None = None
        self.backend_response_id: str | None = None
        self._buffer = b""
        self._disabled = False

    def feed(self, chunk: bytes) -> None:
        if self._disabled:
            return
        if len(self._buffer) + len(chunk) > self.max_bytes:
            self._buffer = b""
            self._disabled = True
            return
        self._buffer += chunk
        if self.streaming:
            self._buffer = self._buffer.replace(b"\r\n", b"\n")
            while b"\n\n" in self._buffer:
                frame, self._buffer = self._buffer.split(b"\n\n", 1)
                data = b"\n".join(
                    line[5:].lstrip()
                    for line in frame.split(b"\n")
                    if line.startswith(b"data:")
                )
                self._observe(data)

    def finish(self) -> None:
        if not self.streaming and not self._disabled:
            self._observe(self._buffer)
        self._buffer = b""

    def _observe(self, data: bytes) -> None:
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        # A stream can be interrupted before its final usage event. Response
        # identity and actual cache usage are independent observations.
        response_id = payload.get("id")
        if isinstance(response_id, str) and 0 < len(response_id) <= 256:
            if self.backend_response_id not in (None, response_id):
                # Conflicting response identities cannot support one receipt.
                self.backend_response_id = None
                self.prompt_tokens = self.cached_tokens = None
                self._buffer = b""
                self._disabled = True
                return
            self.backend_response_id = response_id
        if not isinstance(payload.get("usage"), dict):
            return
        usage = payload["usage"]
        prompt = usage.get("prompt_tokens")
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if type(prompt) is int and prompt >= 0:
            self.prompt_tokens = prompt
        if type(cached) is int and 0 <= cached <= (self.prompt_tokens or 0):
            self.cached_tokens = cached
