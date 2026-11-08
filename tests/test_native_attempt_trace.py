"""Lifecycle retirement and replay identity, independent of host imports."""

import asyncio
from dataclasses import replace

import pytest
from test_native_replay import lease, match, setup_source

from vllm_hust_prefix_router.events.native_replay import ReplayGap


@pytest.mark.asyncio
async def test_close_rejects_an_already_pending_external_completion(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def replay(*_args):
        entered.set()
        await release.wait()
        return ((0, b"a"),)

    source, index, _ = setup_source(monkeypatch, replay)
    pending = asyncio.create_task(source.synchronize_once())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await source.close()
        release.set()
        with pytest.raises(RuntimeError, match="closed"):
            await asyncio.wait_for(pending, 1)
        assert not source.ready and match(index) == 0
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_closed_subscriber_cannot_be_restarted_or_synchronized(monkeypatch):
    called = False

    async def replay(*_args):
        nonlocal called
        called = True
        return ()

    source, index, _ = setup_source(monkeypatch, replay)
    await source.close()
    with pytest.raises(RuntimeError, match="closed"):
        await source.synchronize_once()
    with pytest.raises(RuntimeError, match="closed"):
        await source.start()
    assert not called and not source.ready and match(index) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("observed", [False, True])
async def test_retry_attempts_are_unique_and_commit_trace_survives_failure(
    monkeypatch, observed
):
    responses = [((0, b"a"),), TimeoutError("interrupted"), ()]

    async def replay(*_args):
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    source, index, _ = setup_source(monkeypatch, replay)
    source.config = replace(source.config, observe_replay=observed)
    await source.synchronize_once()
    first = source.status()["last_committed_replay_attempt_id"]
    with pytest.raises(TimeoutError):
        await source.synchronize_once()
    assert not source.ready and match(index) == 0
    failed = source.status()
    assert failed["last_committed_replay_attempt_id"] == first
    assert failed["sync_attempts"] == 2
    await source.synchronize_once()
    final = source.status()
    assert final["last_committed_replay_attempt_id"] != first
    assert final["sync_attempts"] == 3 and source.ready and match(index) == 128
    if observed:
        attempts = final["replay_diagnostics"]["recent_attempts"]
        assert len({a["attempt_id"] for a in attempts}) == 3
        assert attempts[-1]["attempt_id"] == final["last_committed_replay_attempt_id"]


@pytest.mark.asyncio
async def test_generation_aba_cannot_restore_retired_identity(monkeypatch):
    async def replay(*_args):
        return ((0, b"a"),)

    source, index, current = setup_source(monkeypatch, replay)
    original = current["lease"]
    await source.synchronize_once()
    current["lease"] = lease(19010)
    await source.synchronize_once()
    replacement = source.status()["generation"]
    current["lease"] = original
    with pytest.raises(ReplayGap, match="reused"):
        await source.synchronize_once()
    assert source.status()["generation"] == replacement
    assert not source.ready and match(index) == 0


@pytest.mark.asyncio
async def test_block_remove_restore_has_new_replay_identity_and_no_old_watermark(
    monkeypatch,
):
    replies = [((0, b"a"),), ((1, b"remove"),), ((2, b"a"),)]

    async def replay(*_args):
        return replies.pop(0)

    source, index, _ = setup_source(monkeypatch, replay)
    identities = []
    for expected_tokens, expected_seq in ((128, 1), (0, 2), (128, 3)):
        await source.synchronize_once()
        state = source.status()
        identities.append(state["last_committed_replay_attempt_id"])
        assert state["next_seq"] == expected_seq
        assert source.ready and match(index) == expected_tokens
    assert len(set(identities)) == 3
