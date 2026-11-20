"""Exercise the production constructor's host API contract without an NPU."""

import sys
from types import ModuleType

import pytest

from vllm_hust_prefix_router.adapters.vllm_hust import request_fingerprint as adapter


@pytest.mark.asyncio
async def test_normal_constructor_delegates_hashing_and_event_conversion(monkeypatch):
    modules = {}
    for name in (
        "vllm",
        "vllm.tokenizers",
        "vllm.tokenizers.registry",
        "vllm.utils",
        "vllm.utils.hashing",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.kv_cache_utils",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    calls, initialized = [], []
    hash_function = object()

    def factory(size, function):
        assert size == 2 and function is hash_function

        def host_hasher(request):
            assert request.block_hashes == []
            assert request.kv_materialization_runtime_control is None
            assert request.lora_request is None and request.mm_features == []
            assert request.prompt_embeds is None
            calls.append((request.all_token_ids, request.cache_salt))
            return list(range(1, request.num_tokens // size + 1))

        return host_hasher

    modules["vllm.tokenizers.registry"].get_tokenizer = lambda *_a, **_kw: object()
    modules["vllm.utils.hashing"].get_hash_fn_by_name = lambda _name: hash_function
    host = modules["vllm.v1.core.kv_cache_utils"]
    host.get_request_block_hasher = factory
    host.init_none_hash = initialized.append
    host.maybe_convert_block_hash = lambda value: value
    monkeypatch.setattr(adapter, "version", lambda _name: "0.29.1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    fp = adapter.VllmHustTextFingerprinter(adapter.VllmHustFingerprintConfig("fake", 2))
    for salt in (None, "same", "other"):
        result = await fp.fingerprint(
            "/v1/completions",
            {
                "prompt": [1, 2, 3, 4, 5],
                "cache_salt": salt,
            },
        )
        assert result[0].block_hashes == tuple(i.to_bytes(8, "big") for i in (1, 2))
    assert initialized == [hash_function]
    assert calls == [((1, 2, 3, 4, 5), salt) for salt in (None, "same", "other")]


@pytest.mark.asyncio
async def test_explicit_salt_opt_out_retains_fallback():
    fp = adapter.VllmHustTextFingerprinter(
        adapter.VllmHustFingerprintConfig("fake", 2, allow_cache_salt=False),
        _tokenizer=object(),
        _hash_tokens=lambda _tokens: (),
    )
    with pytest.raises(adapter.UnsupportedFingerprintRequest, match="cache_salt"):
        await fp.fingerprint("/v1/completions", {"prompt": [1, 2], "cache_salt": "A"})
