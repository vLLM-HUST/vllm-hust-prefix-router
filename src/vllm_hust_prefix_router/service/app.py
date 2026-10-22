"""aiohttp assembly for the external OpenAI-compatible Router."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from aiohttp import ClientError, web

from vllm_hust_prefix_router.config import BackendConfig
from vllm_hust_prefix_router.core.lifecycle import RequestLifecycleObserver
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.protocols.request_fingerprint import (
    RequestFingerprinter,
    UnsupportedFingerprintRequest,
)
from vllm_hust_prefix_router.service.backend_pool import (
    BackendPoolRegistry,
    BackendPoolSaturated,
)
from vllm_hust_prefix_router.service.receipts import UsageObserver, fingerprint_digest
from vllm_hust_prefix_router.service.timing import LoopLagProbe, RequestTiming

logger = logging.getLogger(__name__)

_REQUEST_PATHS = frozenset({"/v1/completions", "/v1/chat/completions"})
# Before Python 3.11 asyncio.TimeoutError was a distinct exception class.
_TRANSPORT_ERRORS = (ClientError, TimeoutError, asyncio.TimeoutError, ConnectionError)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


@dataclass(frozen=True)
class RouterServiceConfig:
    """Runtime configuration after JSON validation."""

    backends: tuple[BackendConfig, ...]
    default_backend: str
    max_request_body_size: int = 16 * 1024 * 1024
    require_trusted_cache_view: bool = False
    observe_request_timings: bool = False

    def __post_init__(self) -> None:
        node_ids = {backend.node_id for backend in self.backends}
        if not node_ids:
            raise ValueError("at least one backend is required")
        if len(node_ids) != len(self.backends):
            raise ValueError("backend node IDs must be unique")
        if self.default_backend not in node_ids:
            raise ValueError("default_backend must reference a configured backend")
        if (
            not isinstance(self.max_request_body_size, int)
            or isinstance(self.max_request_body_size, bool)
            or self.max_request_body_size <= 0
        ):
            raise ValueError("max_request_body_size must be a positive integer")
        if not isinstance(self.require_trusted_cache_view, bool):
            raise ValueError("require_trusted_cache_view must be boolean")
        if not isinstance(self.observe_request_timings, bool):
            raise ValueError("observe_request_timings must be boolean")


class EventSource(Protocol):
    """Lifecycle and status contract for one backend event source."""

    ready: bool
    config: Any

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def status(self) -> dict[str, object]: ...


@dataclass
class _RequestState:
    request_id: str
    client_request_id_hash: str | None = None
    receipt_recorded: bool = False
    fingerprint_sha256: str | None = None


class RouterService:
    """Route, forward, stream, and observe OpenAI generation requests."""

    def __init__(
        self,
        config: RouterServiceConfig,
        *,
        planner: RoutePlanner,
        fingerprinter: RequestFingerprinter,
        event_sources: tuple[EventSource, ...] = (),
    ) -> None:
        self.config = config
        self.planner = planner
        self.fingerprinter = fingerprinter
        self.event_sources = event_sources
        self.pools = BackendPoolRegistry(list(config.backends))
        self._backends = {backend.node_id: backend for backend in config.backends}
        self._fingerprint_fallbacks = 0
        self._receipts: deque[dict[str, object]] = deque(maxlen=256)
        self._receipt_count = 0
        self._observer_errors = 0
        self._timings: deque[dict[str, object]] = deque(maxlen=256)
        self._timing_count = 0
        self._loop_lag = LoopLagProbe()

    async def start(self) -> None:
        """Start outbound transports."""
        await self.pools.start()
        await asyncio.gather(*(source.start() for source in self.event_sources))
        if self.config.observe_request_timings:
            self._loop_lag.start()

    async def close(self) -> None:
        """Close outbound transports."""
        await asyncio.gather(
            *(source.close() for source in self.event_sources),
            return_exceptions=True,
        )
        await self.pools.close()
        await self._loop_lag.close()

    async def handle_generation(self, request: web.Request) -> web.StreamResponse:
        """Record optional diagnostics, including rejection before dispatch."""
        request_id = uuid.uuid4().hex
        state = _RequestState(request_id)
        if "x-request-id" in request.headers:
            state.client_request_id_hash = hashlib.sha256(
                request.headers["x-request-id"].encode()
            ).hexdigest()
        error_headers = {
            "x-vllm-hust-router-request-id": request_id,
            "x-request-id": request.headers.get("x-request-id") or request_id,
        }
        terminal = "router_error"
        timing = RequestTiming() if self.config.observe_request_timings else None
        if timing is not None:
            timing.request_id = request_id
        try:
            return await self._handle_generation(request, timing, state)
        except web.HTTPException as exc:
            terminal = f"http_{exc.status}"
            exc.headers.update(error_headers)
            if timing is not None:
                timing.http_status = exc.status
                timing.terminal = terminal
            raise
        except asyncio.CancelledError:
            terminal = "cancelled"
            if timing is not None:
                timing.terminal = "cancelled"
            raise
        except Exception:
            if timing is not None:
                timing.http_status = 500
            logger.exception("generation request failed before response commitment")
            raise web.HTTPInternalServerError(headers=error_headers) from None
        finally:
            if not state.receipt_recorded:
                self._record_receipt(
                    {
                        "request_id": request_id,
                        "client_request_id_hash": state.client_request_id_hash,
                        "request_fingerprint_sha256": state.fingerprint_sha256,
                        "node_id": None,
                        "generation": None,
                        "generation_current_at_finish": None,
                        "view_next_seq": None,
                        "view_age_s": None,
                        "predicted_cached_tokens": None,
                        "engine_cached_tokens": None,
                        "backend_response_id": None,
                        "engine_usage_source": None,
                        "response_committed": False,
                        "terminal": terminal,
                    }
                )
            if timing is not None:
                timing.mark("handler_end")
                self._timings.append(timing.snapshot())
                self._timing_count += 1

    def _record_receipt(self, record: dict[str, object]) -> None:
        self._receipt_count += 1
        self._receipts.append({"receipt_seq": self._receipt_count, **record})

    async def _handle_generation(
        self, request: web.Request, timing: RequestTiming | None, state: _RequestState
    ) -> web.StreamResponse:
        """Select one remote backend and stream its response to the client."""
        request_id = state.request_id
        if request.content_length is not None and (
            request.content_length > self.config.max_request_body_size
        ):
            raise web.HTTPRequestEntityTooLarge(
                max_size=self.config.max_request_body_size,
                actual_size=request.content_length,
            )
        body = await request.read()
        if timing is not None:
            timing.mark("body_read_end")
        if len(body) > self.config.max_request_body_size:
            raise web.HTTPRequestEntityTooLarge(
                max_size=self.config.max_request_body_size,
                actual_size=len(body),
            )
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise web.HTTPBadRequest(text="request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="request body must be a JSON object")

        if timing is not None:
            timing.mark("payload_validated")
        external_request_id = request.headers.get("x-request-id") or request_id
        decision = None
        candidates = None
        source_by_node = {
            source.config.node_id: source for source in self.event_sources
        }
        if source_by_node and self.config.require_trusted_cache_view:
            # Strict admission retains its early rejection. Soft fallback only
            # needs eligibility after fingerprinting; this result would otherwise
            # be overwritten before selection, duplicating native identity I/O.
            candidates = tuple(
                node for node, source in source_by_node.items() if source.ready
            )
            if timing is not None:
                timing.mark("strict_eligibility_end")
            if not candidates:
                raise web.HTTPServiceUnavailable(
                    text="no trusted cache view is available"
                )
        try:
            prompts = await self.fingerprinter.fingerprint(request.path, payload)
        except UnsupportedFingerprintRequest:
            self._fingerprint_fallbacks += 1
        else:
            if timing is not None:
                state.fingerprint_sha256 = fingerprint_digest(prompts)
            if timing is not None:
                timing.mark("fingerprint_end")
            # Fingerprinting can yield while a lease expires or an engine exits.
            if source_by_node:
                candidates = tuple(
                    node for node, source in source_by_node.items() if source.ready
                )
            if timing is not None:
                timing.mark("eligibility_end")
            decision = self.planner.choose(
                request_id, prompts, candidate_node_ids=candidates
            )
        if timing is not None:
            timing.mark("plan_end")
        node_id = (
            decision.node_id if decision is not None else self.config.default_backend
        )
        if (
            decision is None
            and source_by_node
            and self.config.require_trusted_cache_view
        ):
            raise web.HTTPServiceUnavailable(
                text="request has no eligible prefix decision"
            )
        backend = self._backends[node_id]
        selected_rank = (
            decision.data_parallel_rank
            if decision is not None
            else backend.data_parallel_rank
        )
        if timing is not None:
            timing.node_id = node_id

        observer = None
        response = None
        response_committed = False
        usage = None
        terminal = "router_error"
        source = source_by_node.get(node_id)
        view = {}
        selection_view = {}
        dispatch_attempted = False
        cache_view_current_at_dispatch = None

        def before_dispatch() -> None:
            nonlocal view, dispatch_attempted, cache_view_current_at_dispatch
            if source is not None:
                view = self._source_status(source)
                cache_view_current_at_dispatch = (
                    view.get("ready") is True
                    and view.get("generation", view.get("publisher_epoch"))
                    == selection_view.get(
                        "generation", selection_view.get("publisher_epoch")
                    )
                    and view.get("identity_digest")
                    == selection_view.get("identity_digest")
                )
                if (
                    self.config.require_trusted_cache_view
                    and not cache_view_current_at_dispatch
                ):
                    raise web.HTTPServiceUnavailable(
                        text="selected cache view changed before backend dispatch"
                    )
            dispatch_attempted = True

        try:
            view = self._source_status(source)
            selection_view = view
            observer = self._lifecycle_observer(request_id, payload, decision)
            if observer is not None:
                observer.dispatch((node_id, selected_rank))
            headers = self._upstream_headers(
                request.headers, backend, data_parallel_rank=selected_rank
            )
            if timing is not None:
                timing.mark("dispatch_diagnostics_end")
            async with self.pools.pool(node_id).request(
                request.method,
                request.path_qs,
                data=body,
                headers=headers,
                timing=timing,
                before_dispatch=before_dispatch,
            ) as upstream:
                if timing is not None:
                    timing.http_status = upstream.status
                if observer is not None:
                    observer.observe(
                        {
                            "type": "http.response.start",
                            "status": upstream.status,
                            "headers": [
                                (key.encode(), value.encode())
                                for key, value in upstream.headers.items()
                            ],
                        }
                    )
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=self._downstream_headers(upstream.headers),
                )
                response.headers["x-vllm-hust-router-node"] = node_id
                response.headers["x-request-id"] = external_request_id
                response.headers["x-vllm-hust-router-request-id"] = request_id
                usage = UsageObserver(
                    streaming="text/event-stream"
                    in upstream.headers.get("content-type", "")
                )
                # Prepare may partially submit headers before raising.
                response_committed = True
                await response.prepare(request)
                if timing is not None:
                    timing.mark("downstream_headers")
                async for chunk in upstream.content.iter_any():
                    if timing is not None and chunk:
                        timing.mark("first_upstream_body")
                    # Observe what the backend sent before a cancellable write.
                    # This does not claim the client received or committed it.
                    usage.feed(chunk)
                    await response.write(chunk)
                    if timing is not None and chunk:
                        timing.mark("first_downstream_body")
                    if observer is not None:
                        observer.observe(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": True,
                            }
                        )
                await response.write_eof()
                usage.finish()
                terminal = "completed"
                if timing is not None:
                    timing.mark("response_eof")
                return response
        except BackendPoolSaturated as exc:
            terminal = "admission_rejected"
            raise web.HTTPServiceUnavailable(
                text=str(exc), headers={"retry-after": "1"}
            ) from exc
        except asyncio.CancelledError:
            terminal = "cancelled"
            raise
        except web.HTTPException as exc:
            terminal = f"http_{exc.status}"
            raise
        except Exception as exc:
            terminal = (
                "transport_error"
                if isinstance(exc, _TRANSPORT_ERRORS)
                else "router_error"
            )
            if response_committed:
                if response is not None:
                    response.force_close()
                if request.transport is not None:
                    request.transport.close()
                return response
            if isinstance(exc, _TRANSPORT_ERRORS):
                raise web.HTTPBadGateway(text=f"backend {node_id!r} failed") from exc
            raise
        finally:
            try:
                if observer is not None:
                    observer.finish()
            except Exception:
                self._observer_errors += 1
                logger.exception("request lifecycle observer cleanup failed")
            finally:
                if decision is not None and decision.decision_id is not None:
                    self.planner.release(decision.decision_id)
                finished_view = self._source_status(source)
                self._record_receipt(
                    {
                        "request_id": request_id,
                        "client_request_id_hash": state.client_request_id_hash,
                        "request_fingerprint_sha256": state.fingerprint_sha256,
                        "node_id": node_id,
                        "data_parallel_rank": selected_rank,
                        "cache_identity_digest": view.get("identity_digest"),
                        "selection_generation": selection_view.get(
                            "generation", selection_view.get("publisher_epoch")
                        ),
                        "dispatch_attempted": dispatch_attempted,
                        "cache_view_current_at_dispatch": (
                            cache_view_current_at_dispatch
                        ),
                        "replay_attempt_id": view.get(
                            "last_committed_replay_attempt_id"
                        ),
                        "generation": view.get(
                            "generation", view.get("publisher_epoch")
                        ),
                        "generation_current_at_finish": view.get(
                            "generation", view.get("publisher_epoch")
                        )
                        == finished_view.get(
                            "generation", finished_view.get("publisher_epoch")
                        )
                        and finished_view.get("ready") is True
                        if view.get("generation", view.get("publisher_epoch"))
                        is not None
                        else None,
                        "view_next_seq": view.get("next_seq"),
                        "view_age_s": view.get("view_age_s"),
                        "predicted_cached_tokens": decision.matched_tokens
                        if decision and cache_view_current_at_dispatch is not False
                        else 0,
                        "engine_cached_tokens": usage.cached_tokens
                        if usage is not None
                        else None,
                        "backend_response_id": usage.backend_response_id
                        if usage is not None
                        else None,
                        "engine_usage_source": "backend_http_usage"
                        if usage is not None and usage.cached_tokens is not None
                        else None,
                        "response_committed": response_committed,
                        "terminal": terminal,
                    }
                )
                state.receipt_recorded = True
                if timing is not None:
                    timing.terminal = terminal
                    timing.mark("cleanup_end")

    @staticmethod
    def _source_status(source: EventSource | None) -> dict[str, object]:
        if source is None:
            return {}
        try:
            return source.status()
        except Exception:
            # Diagnostic failures must not mask cleanup or a committed response.
            logger.exception("cache source diagnostics failed")
            return {"ready": False, "last_error": "status unavailable"}

    def metrics(self) -> dict[str, object]:
        """Return transport metrics suitable for health evidence and scraping."""
        routing: dict[str, object] = {"policy": self.planner.policy}
        tracker = self.planner.lifecycle_tracker
        if tracker is not None:
            routing["lifecycle"] = {
                node_id: asdict(
                    tracker.worker_load((node_id, backend.data_parallel_rank))
                )
                for node_id, backend in self._backends.items()
            }
        return {
            "fingerprint_fallbacks": self._fingerprint_fallbacks,
            "routing": routing,
            "backends": {
                node_id: vars(stats) for node_id, stats in self.pools.stats().items()
            },
            "event_sources": [source.status() for source in self.event_sources],
            "recent_receipts": list(self._receipts),
            "total_receipts": self._receipt_count,
            "observer_errors": self._observer_errors,
            "timing_diagnostics": {
                "enabled": self.config.observe_request_timings,
                "total_records": self._timing_count,
                "retention": 256,
                "recent_requests": list(self._timings),
                "loop_lag": self._loop_lag.snapshot(),
            },
            "cache_views": [
                vars(score) for score in self.planner.prefix_index.score_nodes((), 0)
            ],
        }

    def ready(self) -> bool:
        """Require every configured event source to hold reconciled state."""
        return all(source.ready for source in self.event_sources)

    def _lifecycle_observer(
        self,
        request_id: str,
        payload: dict[str, object],
        decision: Any,
    ) -> RequestLifecycleObserver | None:
        tracker = self.planner.lifecycle_tracker
        if tracker is None or decision is None or decision.decision_id is None:
            return None
        choices = payload.get("n", 1)
        if not isinstance(choices, int) or isinstance(choices, bool) or choices <= 0:
            choices = 1
        observer = RequestLifecycleObserver(tracker)
        observer.bind(request_id, choices_per_prompt=choices)
        return observer

    @staticmethod
    def _upstream_headers(
        headers: Mapping[str, str],
        backend: BackendConfig,
        *,
        data_parallel_rank: int | None = None,
    ) -> dict[str, str]:
        result = {
            key: value
            for key, value in headers.items()
            if key.lower()
            not in _HOP_BY_HOP_HEADERS
            | {"host", "content-length", "x-data-parallel-rank"}
        }
        rank = (
            data_parallel_rank
            if data_parallel_rank is not None
            else backend.data_parallel_rank
        )
        if rank is not None:
            result["x-data-parallel-rank"] = str(rank)
        return result

    @staticmethod
    def _downstream_headers(headers: Mapping[str, str]) -> dict[str, str]:
        return {
            key: value
            for key, value in headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS | {"content-length"}
        }


def create_app(service: RouterService) -> web.Application:
    """Create an aiohttp application without starting a process."""
    app = web.Application(client_max_size=service.config.max_request_body_size)

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def ready(_: web.Request) -> web.Response:
        status = 200 if service.ready() else 503
        return web.json_response(
            {"status": "ready" if status == 200 else "degraded"}, status=status
        )

    async def metrics(_: web.Request) -> web.Response:
        return web.json_response(service.metrics())

    async def start(_: web.Application) -> None:
        await service.start()

    async def close(_: web.Application) -> None:
        await service.close()

    app.router.add_get("/healthz", health)
    app.router.add_get("/readyz", ready)
    app.router.add_get("/metrics", metrics)
    for path in _REQUEST_PATHS:
        app.router.add_post(path, service.handle_generation)
    app.on_startup.append(start)
    app.on_cleanup.append(close)
    return app
