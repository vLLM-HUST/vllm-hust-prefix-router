"""Evidence identity cannot change routing or expose full fingerprint contents."""

from dataclasses import replace

import pytest
from aiohttp import web
from test_service_reliability import setup

from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint
from vllm_hust_prefix_router.service.receipts import fingerprint_digest


def test_fingerprint_identity_encodes_block_lengths_prompt_boundaries_and_count():
    cases = (
        (PromptFingerprint(32, (b"ab", b"c")),),
        (PromptFingerprint(32, (b"a", b"bc")),),
        (PromptFingerprint(33, (b"ab", b"c")),),
        (PromptFingerprint(32, (b"c", b"ab")),),
        (PromptFingerprint(16, (b"ab",)), PromptFingerprint(16, (b"c",))),
    )
    assert len({fingerprint_digest(p) for p in cases}) == len(cases)
    assert fingerprint_digest(cases[0]) == fingerprint_digest(cases[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("observed", [False, True])
async def test_receipt_fingerprint_is_optional_and_never_retokens_request(
    monkeypatch, observed
):
    async def backend(_request):
        return web.json_response({"ok": True})

    service, _, client, upstream = await setup(backend)
    service.config = replace(service.config, observe_request_timings=observed)
    calls = 0
    original = service.fingerprinter.fingerprint

    async def fingerprint(*args):
        nonlocal calls
        calls += 1
        return await original(*args)

    monkeypatch.setattr(service.fingerprinter, "fingerprint", fingerprint)
    try:
        response = await client.post("/v1/completions", json={"prompt": "x"})
        await response.read()
        receipt = service.metrics()["recent_receipts"][-1]
        assert calls == 1
        expected = fingerprint_digest((PromptFingerprint(32, (b"a", b"b")),))
        assert receipt["request_fingerprint_sha256"] == (expected if observed else None)
        assert receipt["replay_attempt_id"] is None
    finally:
        await client.close()
        await upstream.close()
