import json

import pytest

from vllm_hust_prefix_router.service.receipts import UsageObserver


@pytest.mark.parametrize(
    "cached,expected",
    [(0, 0), (128, 128), (None, None), (-1, None), (True, None), (257, None)],
)
def test_json_usage_does_not_infer_unknown_cached_tokens(cached, expected):
    observer = UsageObserver(streaming=False)
    observer.feed(
        json.dumps(
            {
                "usage": {
                    "prompt_tokens": 256,
                    "prompt_tokens_details": {"cached_tokens": cached},
                }
            }
        ).encode()
    )
    observer.finish()
    assert observer.cached_tokens == expected


def test_fragmented_sse_usage_is_observed():
    observer = UsageObserver(streaming=True)
    event = (
        b'data: {"usage":{"prompt_tokens":256,'
        b'"prompt_tokens_details":{"cached_tokens":128}}}\n\n'
    )
    for byte in event:
        observer.feed(bytes([byte]))
    observer.feed(b"data: [DONE]\n\n")
    observer.finish()
    assert observer.cached_tokens == 128


def test_oversized_response_remains_unknown():
    observer = UsageObserver(streaming=False, max_bytes=2)
    observer.feed(b"too long")
    observer.finish()
    assert observer.cached_tokens is None


def test_interrupted_sse_retains_identity_without_inventing_usage():
    observer = UsageObserver(streaming=True)
    frame = b'data: {"id":"cmpl-actual","choices":[{"text":"token"}]}\n\n'
    for byte in frame:
        observer.feed(bytes([byte]))
    observer.finish()
    assert observer.backend_response_id == "cmpl-actual"
    assert observer.cached_tokens is None and observer.prompt_tokens is None


@pytest.mark.parametrize("response_id", ["", "x" * 257, None, 123, True])
def test_sse_response_identity_is_bounded(response_id):
    observer = UsageObserver(streaming=True)
    observer.feed(b"data: " + json.dumps({"id": response_id}).encode() + b"\n\n")
    assert observer.backend_response_id is None


def test_sse_conflicting_id_invalidates_evidence_without_rebinding():
    observer = UsageObserver(streaming=True)
    for response_id in ("first", "other", "first"):
        observer.feed(
            b"data: "
            + json.dumps(
                {
                    "id": response_id,
                    "usage": {
                        "prompt_tokens": 400,
                        "prompt_tokens_details": {"cached_tokens": 384},
                    },
                }
            ).encode()
            + b"\n\n"
        )
    observer.finish()
    assert observer.backend_response_id is None and observer.cached_tokens is None
