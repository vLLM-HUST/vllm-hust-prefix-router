import pytest

from vllm_hust_prefix_router.adapters.vllm_hust.request_fingerprint import (
    UnsupportedFingerprintRequest,
    VllmHustFingerprintConfig,
    VllmHustTextFingerprinter,
    normalize_external_block_hash,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = [ord(value) for value in text]
        return ([1] + tokens) if add_special_tokens else tokens

    def apply_chat_template(self, _messages, **kwargs):
        assert kwargs["tokenize"]
        return [10, 11, 12, 13]

    def decode(self, _tokens, _skip_special_tokens=False):
        return "chat"


def _fingerprinter() -> VllmHustTextFingerprinter:
    return VllmHustTextFingerprinter(
        VllmHustFingerprintConfig("fake", hash_block_size=2),
        _tokenizer=FakeTokenizer(),
        _hash_tokens=lambda tokens: (
            len(tokens).to_bytes(2, "big"),
        ),
    )


@pytest.mark.asyncio
async def test_completion_fingerprint_supports_text_batches() -> None:
    fingerprints = await _fingerprinter().fingerprint(
        "/v1/completions",
        {"prompt": ["a", "bc"], "add_special_tokens": False},
    )

    assert [fingerprint.num_tokens for fingerprint in fingerprints] == [1, 2]
    assert [fingerprint.block_hashes for fingerprint in fingerprints] == [
        (b"\x00\x01",),
        (b"\x00\x02",),
    ]


@pytest.mark.asyncio
async def test_chat_fingerprint_rejects_non_text_content() -> None:
    with pytest.raises(UnsupportedFingerprintRequest, match="text chat"):
        await _fingerprinter().fingerprint(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
        )


def test_external_integer_hash_is_normalized_to_eight_bytes() -> None:
    assert normalize_external_block_hash(1) == b"\x00" * 7 + b"\x01"
    assert normalize_external_block_hash(b"hash") == b"hash"
