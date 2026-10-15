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
        assert kwargs["return_dict"] is False
        return [10, 11, 12, 13]

    def decode(self, _tokens, _skip_special_tokens=False):
        return "chat"


def _fingerprinter() -> VllmHustTextFingerprinter:
    return VllmHustTextFingerprinter(
        VllmHustFingerprintConfig("fake", hash_block_size=2),
        _tokenizer=FakeTokenizer(),
        _hash_tokens=lambda tokens: (len(tokens).to_bytes(2, "big"),),
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


@pytest.mark.asyncio
async def test_chat_template_override_falls_back_instead_of_misrouting() -> None:
    with pytest.raises(UnsupportedFingerprintRequest, match="chat_template_kwargs"):
        await _fingerprinter().fingerprint(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )


def test_external_integer_hash_is_normalized_to_eight_bytes() -> None:
    assert normalize_external_block_hash(1) == b"\x00" * 7 + b"\x01"
    assert normalize_external_block_hash(b"hash") == b"hash"


@pytest.mark.asyncio
async def test_chat_requests_list_return_on_transformers_v5():
    class V5Tokenizer(FakeTokenizer):
        def apply_chat_template(self, _messages, **kwargs):
            ids = [10, 11, 12, 13]
            return ids if kwargs.get("return_dict") is False else {"input_ids": ids}

    fingerprinter = VllmHustTextFingerprinter(
        VllmHustFingerprintConfig("fake", hash_block_size=2),
        _tokenizer=V5Tokenizer(),
        _hash_tokens=lambda tokens: (len(tokens).to_bytes(2, "big"),),
    )
    result = await fingerprinter.fingerprint(
        "/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}
    )
    assert result[0].num_tokens == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("rendered", [[True], [-1], [[1]], {"input_ids": [1]}])
async def test_chat_rejects_invalid_or_unexpected_token_shapes(rendered):
    class InvalidTokenizer(FakeTokenizer):
        def apply_chat_template(self, _messages, **_kwargs):
            return rendered

    fingerprinter = VllmHustTextFingerprinter(
        VllmHustFingerprintConfig("fake", hash_block_size=2),
        _tokenizer=InvalidTokenizer(),
        _hash_tokens=lambda _tokens: (),
    )
    with pytest.raises(UnsupportedFingerprintRequest, match="token IDs"):
        await fingerprinter.fingerprint(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}
        )
