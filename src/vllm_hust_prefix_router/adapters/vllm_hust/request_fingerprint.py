"""Text request fingerprinting compatible with the tested vLLM-HUST line."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from packaging.specifiers import SpecifierSet

from vllm_hust_prefix_router.protocols.request_fingerprint import (
    PromptFingerprint,
    UnsupportedFingerprintRequest,
)


def normalize_external_block_hash(value: bytes | int) -> bytes:
    """Normalize the two vLLM KV-event hash representations."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return (value & ((1 << 64) - 1)).to_bytes(8, "big")
    raise TypeError("block hash must be bytes or a non-negative integer")


@dataclass(frozen=True)
class VllmHustFingerprintConfig:
    """Configuration that must match every worker in the Router pool."""

    tokenizer: str
    hash_block_size: int
    hash_algorithm: str = "xxhash"
    host_version_range: str = ">=0.23.1,<0.24"
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.tokenizer:
            raise ValueError("tokenizer must be non-empty")
        if self.hash_block_size <= 0:
            raise ValueError("hash_block_size must be positive")
        SpecifierSet(self.host_version_range)


class VllmHustTextFingerprinter:
    """Render and hash supported text-only OpenAI requests without an Engine."""

    def __init__(
        self,
        config: VllmHustFingerprintConfig,
        *,
        _tokenizer: Any | None = None,
        _hash_tokens: Callable[[Sequence[int]], tuple[bytes, ...]] | None = None,
    ) -> None:
        self.config = config
        if _tokenizer is not None and _hash_tokens is not None:
            self._tokenizer = _tokenizer
            self._hash_tokens = _hash_tokens
            return

        if not os.getenv("PYTHONHASHSEED"):
            raise RuntimeError(
                "PYTHONHASHSEED must be fixed and identical on Router and workers"
            )
        try:
            installed_version = version("vllm")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "vLLM-HUST must be installed to use its request fingerprint adapter"
            ) from exc
        if installed_version not in SpecifierSet(config.host_version_range):
            raise RuntimeError(
                f"vLLM-HUST {installed_version} is outside the tested range "
                f"{config.host_version_range}"
            )

        from vllm.tokenizers.registry import get_tokenizer
        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import (
            hash_block_tokens,
            init_none_hash,
            maybe_convert_block_hash,
        )

        self._tokenizer = get_tokenizer(
            config.tokenizer,
            revision=config.tokenizer_revision,
            trust_remote_code=config.trust_remote_code,
        )
        hash_function = get_hash_fn_by_name(config.hash_algorithm)
        init_none_hash(hash_function)

        def hash_tokens(token_ids: Sequence[int]) -> tuple[bytes, ...]:
            parent_hash = None
            hashes: list[bytes] = []
            for start in range(0, len(token_ids), config.hash_block_size):
                block = token_ids[start : start + config.hash_block_size]
                if len(block) < config.hash_block_size:
                    break
                parent_hash = hash_block_tokens(
                    hash_function,
                    parent_hash,
                    block,
                    extra_keys=None,
                )
                hashes.append(
                    normalize_external_block_hash(maybe_convert_block_hash(parent_hash))
                )
            return tuple(hashes)

        self._hash_tokens = hash_tokens

    async def fingerprint(
        self, path: str, payload: dict[str, object]
    ) -> tuple[PromptFingerprint, ...]:
        """Fingerprint supported completion or chat-completion input."""
        self._reject_unsupported(payload)
        if path == "/v1/completions":
            token_batches = self._completion_tokens(payload)
        elif path == "/v1/chat/completions":
            token_batches = (self._chat_tokens(payload),)
        else:
            raise UnsupportedFingerprintRequest(f"unsupported request path {path!r}")
        return tuple(
            PromptFingerprint(len(token_ids), self._hash_tokens(token_ids))
            for token_ids in token_batches
        )

    @staticmethod
    def _reject_unsupported(payload: dict[str, object]) -> None:
        unsupported = (
            "prompt_embeds",
            "multi_modal_data",
            "cache_salt",
            "lora_request",
            "documents",
            "tools",
        )
        present = [name for name in unsupported if payload.get(name) not in (None, [])]
        if present:
            raise UnsupportedFingerprintRequest(
                f"prefix fingerprinting does not support: {', '.join(present)}"
            )

    def _completion_tokens(
        self, payload: dict[str, object]
    ) -> tuple[tuple[int, ...], ...]:
        prompt = payload.get("prompt")
        add_special_tokens = payload.get("add_special_tokens", True)
        if not isinstance(add_special_tokens, bool):
            raise UnsupportedFingerprintRequest("add_special_tokens must be boolean")
        if isinstance(prompt, str):
            batches = (
                tuple(
                    self._tokenizer.encode(
                        prompt, add_special_tokens=add_special_tokens
                    )
                ),
            )
        elif isinstance(prompt, list) and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in prompt
        ):
            batches = (tuple(prompt),)
        elif isinstance(prompt, list) and all(
            isinstance(value, str) for value in prompt
        ):
            batches = tuple(
                tuple(
                    self._tokenizer.encode(value, add_special_tokens=add_special_tokens)
                )
                for value in prompt
            )
        elif isinstance(prompt, list) and all(
            isinstance(value, list)
            and all(
                isinstance(token, int) and not isinstance(token, bool) and token >= 0
                for token in value
            )
            for value in prompt
        ):
            batches = tuple(tuple(value) for value in prompt)
        else:
            raise UnsupportedFingerprintRequest("unsupported completion prompt type")
        return tuple(self._truncate(tokens, payload) for tokens in batches)

    def _chat_tokens(self, payload: dict[str, object]) -> tuple[int, ...]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise UnsupportedFingerprintRequest(
                "chat messages must be a non-empty list"
            )
        for message in messages:
            if not isinstance(message, dict) or not isinstance(
                message.get("content"), str
            ):
                raise UnsupportedFingerprintRequest(
                    "only text chat message content is supported"
                )
        add_generation_prompt = payload.get("add_generation_prompt", True)
        continue_final_message = payload.get("continue_final_message", False)
        add_special_tokens = payload.get("add_special_tokens", False)
        if not all(
            isinstance(value, bool)
            for value in (
                add_generation_prompt,
                continue_final_message,
                add_special_tokens,
            )
        ):
            raise UnsupportedFingerprintRequest("chat template flags must be boolean")
        if add_generation_prompt and continue_final_message:
            raise UnsupportedFingerprintRequest(
                "add_generation_prompt and continue_final_message conflict"
            )
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        if not isinstance(rendered, list) or not all(
            isinstance(token, int) for token in rendered
        ):
            raise UnsupportedFingerprintRequest(
                "tokenizer chat template did not return token IDs"
            )
        tokens = tuple(rendered)
        if add_special_tokens:
            text = self._tokenizer.decode(tokens, skip_special_tokens=False)
            tokens = tuple(self._tokenizer.encode(text, add_special_tokens=True))
        return self._truncate(tokens, payload)

    @staticmethod
    def _truncate(
        token_ids: tuple[int, ...], payload: dict[str, object]
    ) -> tuple[int, ...]:
        limit = payload.get("truncate_prompt_tokens")
        if limit is None or limit == -1:
            return token_ids
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise UnsupportedFingerprintRequest(
                "truncate_prompt_tokens must be positive or -1"
            )
        side = payload.get("truncation_side", "left")
        if side == "left":
            return token_ids[-limit:]
        if side == "right":
            return token_ids[:limit]
        raise UnsupportedFingerprintRequest("truncation_side must be left or right")
