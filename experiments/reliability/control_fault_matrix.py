"""Virtual-clock HTTP correctness matrix, with explicitly synthetic KV usage.

Uses the real Router/index/replay code, not a fake routing implementation. The
request client is sequential; this cannot measure burst queueing, P99 or capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vllm_hust_prefix_router.config import BackendConfig
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.managed_identity import (
    ManagedBackendLease,
    ProcessIdentity,
)
from vllm_hust_prefix_router.events.native_replay import (
    NativeReplayConfig,
    NativeReplaySubscriber,
    ReplayGap,
)
from vllm_hust_prefix_router.protocols.cache_events import (
    BlockStored,
    CacheEventBatch,
)
from vllm_hust_prefix_router.protocols.request_fingerprint import PromptFingerprint
from vllm_hust_prefix_router.service.app import (
    RouterService,
    RouterServiceConfig,
    create_app,
)
from vllm_hust_prefix_router.service.receipts import fingerprint_digest

PATTERNS = ("steady", "microburst", "hotspot", "phase-shift")
FAULTS = (
    "stale-view",
    "worker-restart",
    "replay-timeout",
    "partial-reply",
    "history-gap",
    "malformed-tail",
)
MODES = ("safe-fallback", "strict-reject")


def trace(pattern, seed=17):
    if pattern not in PATTERNS:
        raise ValueError("unknown arrival pattern")
    arrivals = (
        [0, 0, 0.01, 0.01, 3, 3, 3.01, 6, 6, 6.01, 9, 9.01]
        if pattern == "microburst"
        else list(range(12))
    )
    prefixes = ["a", "b"] * 6
    random.Random(seed).shuffle(prefixes)
    if pattern == "hotspot":
        prefixes = ["a"] * 12
    elif pattern == "phase-shift":
        prefixes = ["a"] * 6 + ["b"] * 6
    return [
        {"ordinal": i, "arrival_s": at, "prefix": prefix}
        for i, (at, prefix) in enumerate(zip(arrivals, prefixes, strict=True))
    ]


def trace_digest(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Codec:
    def decode_batch(self, payload):
        if payload == b"bad-tail":
            raise ValueError("injected malformed tail")
        return CacheEventBatch((BlockStored((payload,), 128),), 0)


class Fingerprinter:
    async def fingerprint(self, _path, payload):
        return (PromptFingerprint(129, (payload["prompt"].encode(),)),)


class BarrierSource(NativeReplaySubscriber):
    """Only launcher identity and clocks are simulated; replay commits are real."""

    def __init__(self, config, *, index, world):
        self.world = world
        super().__init__(
            config,
            index=index,
            codec=Codec(),
            replay=self.reply,
            clock=lambda: world["time"],
        )

    def _read_lease(self):
        if not self.world["alive"]:
            raise ProcessLookupError("injected retired worker")
        return self.world["lease"]

    async def start(self):
        # Explicit barriers replace polling for deterministic CPU validation.
        pass

    async def reply(self, _endpoint, sequence, _config):
        fault = self.world["fault"]
        if fault in ("replay-timeout", "partial-reply"):
            # The wire-level partial-prefix/END_SEQ behavior is tested separately.
            raise TimeoutError("injected incomplete replay transaction")
        if fault == "history-gap":
            return ((sequence + 1, b"a"),)
        if fault == "malformed-tail":
            return ((sequence, b"uncommitted"), (sequence + 1, b"bad-tail"))
        return ((0, b"a"), (1, b"b")) if sequence == 0 else ()


def make_lease(url, identity):
    return ManagedBackendLease(
        "cached",
        str(uuid5(NAMESPACE_URL, "control-matrix/" + identity)),
        ProcessIdentity(10, "synthetic-boot", "synthetic-namespace", 100),
        ProcessIdentity(11, "synthetic-boot", "synthetic-namespace", 101),
        url,
        "tcp://127.0.0.1:19000" if identity == "first" else "tcp://127.0.0.1:19010",
        "tcp://127.0.0.1:19001" if identity == "first" else "tcp://127.0.0.1:19011",
        "/synthetic-model",
        128,
        "sha256",
        True,
        "a" * 64,
    )


async def cell(pattern, fault, mode, *, seed=17):
    if fault not in FAULTS or mode not in MODES:
        raise ValueError("unknown fault or deployment mode")
    requests = trace(pattern, seed)
    dispatched = []

    def backend(node):
        async def handle(request):
            payload = await request.json()
            dispatched.append(
                {"node": node, "external_id": request.headers["x-request-id"]}
            )
            return web.json_response(
                {
                    "id": "synthetic:" + request.headers["x-request-id"],
                    "choices": [{"text": payload["prompt"]}],
                    "usage": {
                        "prompt_tokens": 129,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {
                            "cached_tokens": 128 if node == "cached" else 0
                        },
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/v1/completions", handle)
        return TestServer(app)

    cached, cold = backend("cached"), backend("cold")
    client = None
    source = None
    try:
        await cached.start_server()
        await cold.start_server()
        cached_url, cold_url = str(cached.make_url("")), str(cold.make_url(""))
        world = {
            "time": 0,
            "alive": True,
            "fault": None,
            "lease": make_lease(cached_url, "first"),
        }
        index = GlobalPrefixIndex()
        for node in ("cached", "cold"):
            index.register_node(node, hash_block_size=128, data_parallel_rank=0)
        source = BarrierSource(
            NativeReplayConfig(
                "cached",
                cached_url,
                Path("/synthetic-lease"),
                "/synthetic-model",
                128,
                "a" * 64,
                interval_s=0.1,
                view_ttl_s=0.5,
                observe_replay=True,
            ),
            index=index,
            world=world,
        )
        service = RouterService(
            RouterServiceConfig(
                (
                    BackendConfig("cached", cached_url, data_parallel_rank=0),
                    BackendConfig("cold", cold_url, data_parallel_rank=0),
                ),
                "cold",
                require_trusted_cache_view=mode == "strict-reject",
                observe_request_timings=True,
            ),
            planner=RoutePlanner(index, policy="prefix", block_size=128),
            fingerprinter=Fingerprinter(),
            event_sources=(source,),
        )
        client = TestClient(TestServer(create_app(service), handler_cancellation=True))
        await client.start_server()
        rows, failures = [], []
        restored = False
        for request in requests:
            at = request["arrival_s"]
            world["time"] = at
            affected = 3 <= at < 6
            if affected:
                world["fault"] = fault
                world["alive"] = fault != "worker-restart"
                if fault != "stale-view":
                    try:
                        await source.synchronize_once()
                    except (
                        TimeoutError,
                        ReplayGap,
                        ValueError,
                        ProcessLookupError,
                    ) as exc:
                        failures.append(type(exc).__name__)
            else:
                if at >= 6 and not restored:
                    world.update(
                        alive=True,
                        fault=None,
                        lease=make_lease(cached_url, "replacement"),
                    )
                    restored = True
                await source.synchronize_once()
            view = source.status()
            assert view["ready"] is not affected
            assert index.score_nodes((b"uncommitted",), 129)[0].matched_tokens == 0
            attempt = f"{pattern}/{fault}/{request['ordinal']}"
            response = await client.post(
                "/v1/completions",
                json={"prompt": request["prefix"]},
                headers={"x-request-id": attempt},
            )
            body = await response.read()

            async def drained(expected=request["ordinal"] + 1):
                while service.metrics()["total_receipts"] < expected:
                    await asyncio.sleep(0)

            await asyncio.wait_for(drained(), 1)
            receipt = service.metrics()["recent_receipts"][-1]
            expected_status = 503 if affected and mode == "strict-reject" else 200
            assert response.status == expected_status
            assert (
                receipt["request_id"]
                == response.headers["x-vllm-hust-router-request-id"]
            )
            assert (
                receipt["client_request_id_hash"]
                == hashlib.sha256(attempt.encode()).hexdigest()
            )
            if expected_status == 200:
                expected_node = "cold" if affected else "cached"
                assert receipt["node_id"] == expected_node
                assert receipt["predicted_cached_tokens"] == (0 if affected else 128)
                assert json.loads(body)["choices"][0]["text"] == request["prefix"]
                expected_fingerprint = fingerprint_digest(
                    (PromptFingerprint(129, (request["prefix"].encode(),)),)
                )
                assert receipt["request_fingerprint_sha256"] == expected_fingerprint
                if not affected:
                    assert (
                        receipt["replay_attempt_id"]
                        == view["last_committed_replay_attempt_id"]
                    )
                    assert receipt["generation"] == view["generation"]
                    assert receipt["cache_identity_digest"] == "a" * 64
            else:
                assert receipt["node_id"] is None and not receipt["response_committed"]
            rows.append(
                {
                    **request,
                    "attempt_id": attempt,
                    "affected": affected,
                    "view": view,
                    "http_status": response.status,
                    "receipt": receipt,
                    "unsafe_retained_hint_tokens": 128 if affected else None,
                    "body_utf8": body.decode(),
                }
            )
        metrics = service.metrics()
        assert metrics["total_receipts"] == len(requests)
        assert len({r["receipt"]["request_id"] for r in rows}) == len(requests)
        assert len(dispatched) == sum(r["http_status"] == 200 for r in rows)
        assert len({r["external_id"] for r in dispatched}) == len(dispatched)
        assert all(
            not p["active_requests"] and not p["queued_requests"]
            for p in metrics["backends"].values()
        )
        assert any(r["affected"] for r in rows) and restored
        assert rows[-1]["receipt"]["predicted_cached_tokens"] == 128
        return {
            "pattern": pattern,
            "fault": fault,
            "mode": mode,
            "seed": seed,
            "trace_sha256": trace_digest(requests),
            "rows": rows,
            "replay_failures": failures,
            "backend_dispatches": dispatched,
            "final_metrics": metrics,
            "result": "pass",
            "backend_kind": "synthetic-http",
            "arrival_clock": "virtual-sequential",
            "engine_reuse_qualified": False,
            "performance_qualified": False,
            "kv_transfer_fencing_qualified": False,
        }
    finally:
        try:
            if client is not None:
                await client.close()
        finally:
            if source is not None:
                await source.close()
            await asyncio.gather(cached.close(), cold.close())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_cell(result):
    """Rebuild correctness expectations without sockets or host imports."""
    pattern, fault, mode = result["pattern"], result["fault"], result["mode"]
    require(fault in FAULTS and mode in MODES, "unknown fault or mode")
    requests = trace(pattern, result["seed"])
    require(result["trace_sha256"] == trace_digest(requests), "trace hash mismatch")
    require(result["backend_kind"] == "synthetic-http", "unexpected evidence source")
    require(result["arrival_clock"] == "virtual-sequential", "unexpected arrival clock")
    for key in (
        "engine_reuse_qualified",
        "performance_qualified",
        "kv_transfer_fencing_qualified",
    ):
        require(result[key] is False, "invalid experimental claim")
    rows = result["rows"]
    require(len(rows) == len(requests), "missing attempts")
    ids, dispatches, first_generation = set(), [], None
    for expected, row in zip(requests, rows, strict=True):
        require(
            all(row[key] == value for key, value in expected.items()),
            "request trace changed",
        )
        affected = 3 <= row["arrival_s"] < 6
        require(row["affected"] is affected, "fault phase mismatch")
        view, receipt = row["view"], row["receipt"]
        require(view["ready"] is not affected, "view trust mismatch")
        attempt = f"{pattern}/{fault}/{row['ordinal']}"
        require(row["attempt_id"] == attempt, "attempt identity mismatch")
        require(
            receipt["client_request_id_hash"]
            == hashlib.sha256(attempt.encode()).hexdigest(),
            "receipt correlation mismatch",
        )
        require(receipt["receipt_seq"] == row["ordinal"] + 1, "receipt gap")
        require(receipt["request_id"] not in ids, "duplicate response receipt")
        ids.add(receipt["request_id"])
        status = 503 if affected and mode == "strict-reject" else 200
        require(row["http_status"] == status, "HTTP status mismatch")
        require(
            row["unsafe_retained_hint_tokens"] == (128 if affected else None),
            "unsafe shadow mismatch",
        )
        if status == 503:
            require(
                receipt["node_id"] is None and receipt["response_committed"] is False,
                "rejection dispatched",
            )
            require(receipt["terminal"] == "http_503", "rejection terminal mismatch")
            continue
        node = "cold" if affected else "cached"
        dispatches.append({"node": node, "external_id": attempt})
        require(receipt["node_id"] == node, "unsafe backend selection")
        require(
            receipt["predicted_cached_tokens"] == (0 if affected else 128),
            "unsafe cache hint",
        )
        require(receipt["terminal"] == "completed", "incomplete response")
        body = json.loads(row["body_utf8"])
        require(body["id"] == "synthetic:" + attempt, "backend response mismatch")
        require(body["choices"][0]["text"] == row["prefix"], "output identity mismatch")
        require(
            body["usage"]["prompt_tokens_details"]["cached_tokens"]
            == receipt["engine_cached_tokens"]
            == (0 if affected else 128),
            "synthetic usage mismatch",
        )
        digest = fingerprint_digest(
            (PromptFingerprint(129, (row["prefix"].encode(),)),)
        )
        require(
            receipt["request_fingerprint_sha256"] == digest,
            "fingerprint evidence mismatch",
        )
        if not affected:
            require(
                receipt["replay_attempt_id"] == view["last_committed_replay_attempt_id"]
                and bool(receipt["replay_attempt_id"]),
                "replay attempt mismatch",
            )
            require(receipt["generation"] == view["generation"], "generation mismatch")
            require(
                receipt["cache_identity_digest"] == "a" * 64
                and receipt["data_parallel_rank"] == 0,
                "cache identity mismatch",
            )
            if first_generation is None:
                first_generation = receipt["generation"]
    require(result["backend_dispatches"] == dispatches, "extra or reordered dispatch")
    final = result["final_metrics"]
    require(final["total_receipts"] == len(rows), "receipt counter mismatch")
    require(
        final["recent_receipts"] == [r["receipt"] for r in rows],
        "final receipts mismatch",
    )
    require(
        all(
            not p["active_requests"] and not p["queued_requests"]
            for p in final["backends"].values()
        ),
        "resources not released",
    )
    require(
        rows[-1]["receipt"]["generation"] != first_generation
        and rows[-1]["receipt"]["predicted_cached_tokens"] == 128,
        "recovery positive missing",
    )


def validate_matrix(directory):
    count = 0
    for pattern in PATTERNS:
        for fault in FAULTS:
            hashes = []
            for mode in MODES:
                result = json.loads(
                    (directory / f"{pattern}-{fault}-{mode}.json").read_bytes()
                )
                require(
                    (result["pattern"], result["fault"], result["mode"])
                    == (pattern, fault, mode),
                    "cell label mismatch",
                )
                validate_cell(result)
                count += len(result["rows"])
                hashes.append(result["trace_sha256"])
            require(hashes[0] == hashes[1], "deployment modes did not match requests")
    return {
        "result": "pass",
        "cells": 48,
        "requests": count,
        "engine_reuse_qualified": False,
        "performance_qualified": False,
        "kv_transfer_fencing_qualified": False,
    }


async def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for pattern in PATTERNS:
        for fault in FAULTS:
            matched = []
            for mode in MODES:
                result = await cell(pattern, fault, mode)
                (directory / f"{pattern}-{fault}-{mode}.json").write_text(
                    json.dumps(result, indent=2)
                )
                matched.append(result["trace_sha256"])
            assert matched[0] == matched[1]
    summary = validate_matrix(directory)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_matrix(args.validate)))
    else:
        asyncio.run(run(args.output_dir))


if __name__ == "__main__":
    main()
