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


@pytest.mark.asyncio
async def test_salt_is_passed_to_host_hasher_without_changing_tokens():
    calls = []

    def salted(tokens, salt):
        calls.append((tuple(tokens), salt))
        return (salt.encode(),)

    fp = VllmHustTextFingerprinter(
        VllmHustFingerprintConfig("fake", 2),
        _tokenizer=FakeTokenizer(),
        _hash_tokens=lambda _tokens: (b"plain",),
        _hash_salted_tokens=salted,
    )
    for salt in ("A", "B"):
        result = await fp.fingerprint(
            "/v1/completions",
            {
                "prompt": [[1, 2, 3], [4, 5]],
                "cache_salt": salt,
            },
        )
        assert [r.num_tokens for r in result] == [3, 2]
        assert all(r.block_hashes == (salt.encode(),) for r in result)
    assert calls == [((1, 2, 3), "A"), ((4, 5), "A"), ((1, 2, 3), "B"), ((4, 5), "B")]


@pytest.mark.asyncio
@pytest.mark.parametrize("salt", ["", 1, True, [], "x" * 1025])
async def test_invalid_salt_falls_back(salt):
    with pytest.raises(UnsupportedFingerprintRequest, match="cache_salt"):
        await _fingerprinter().fingerprint(
            "/v1/completions",
            {
                "prompt": [1, 2],
                "cache_salt": salt,
            },
        )


@pytest.mark.asyncio
async def test_unknown_runtime_hash_controls_fall_back():
    with pytest.raises(UnsupportedFingerprintRequest, match="vllm_xargs"):
        await _fingerprinter().fingerprint(
            "/v1/completions",
            {
                "prompt": [1, 2],
                "vllm_xargs": {"runtime": "unknown"},
            },
        )
