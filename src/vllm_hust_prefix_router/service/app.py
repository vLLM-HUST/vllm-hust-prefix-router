"""aiohttp assembly for the external OpenAI-compatible Router."""

from __future__ import annotations

import asyncio
import json
import uuid
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

_REQUEST_PATHS = frozenset({"/v1/completions", "/v1/chat/completions"})
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


class EventSource(Protocol):
    """Lifecycle and status contract for one backend event source."""

    ready: bool

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def status(self) -> dict[str, object]: ...


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

    async def start(self) -> None:
        """Start outbound transports."""
        await self.pools.start()
        await asyncio.gather(*(source.start() for source in self.event_sources))

    async def close(self) -> None:
        """Close outbound transports."""
        await asyncio.gather(
            *(source.close() for source in self.event_sources),
            return_exceptions=True,
        )
        await self.pools.close()

    async def handle_generation(self, request: web.Request) -> web.StreamResponse:
        """Select one remote backend and stream its response to the client."""
        if request.content_length is not None and (
            request.content_length > self.config.max_request_body_size
        ):
            raise web.HTTPRequestEntityTooLarge(
                max_size=self.config.max_request_body_size,
                actual_size=request.content_length,
            )
        body = await request.read()
        if len(body) > self.config.max_request_body_size:
            raise web.HTTPRequestEntityTooLarge(
                max_size=self.config.max_request_body_size,
                actual_size=len(body),
            )
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(text="request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="request body must be a JSON object")

        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        decision = None
        try:
            prompts = await self.fingerprinter.fingerprint(request.path, payload)
        except UnsupportedFingerprintRequest:
            self._fingerprint_fallbacks += 1
        else:
            decision = self.planner.choose(request_id, prompts)
        node_id = (
            decision.node_id if decision is not None else self.config.default_backend
        )
        backend = self._backends[node_id]

        observer = self._lifecycle_observer(request_id, payload, decision)
        if observer is not None:
            observer.dispatch((node_id, backend.data_parallel_rank))

        headers = self._upstream_headers(request.headers, backend)
        try:
            async with self.pools.pool(node_id).request(
                request.method,
                request.path_qs,
                data=body,
                headers=headers,
            ) as upstream:
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
                response.headers["x-request-id"] = request_id
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                    if observer is not None:
                        observer.observe(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": True,
                            }
                        )
                await response.write_eof()
                return response
        except BackendPoolSaturated as exc:
            raise web.HTTPServiceUnavailable(
                text=str(exc), headers={"retry-after": "1"}
            ) from exc
        except (ClientError, asyncio.TimeoutError) as exc:
            raise web.HTTPBadGateway(text=f"backend {node_id!r} failed") from exc
        finally:
            if observer is not None:
                observer.finish()
            elif decision is not None and decision.decision_id is not None:
                self.planner.release(decision.decision_id)

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
        headers: Mapping[str, str], backend: BackendConfig
    ) -> dict[str, str]:
        result = {
            key: value
            for key, value in headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS | {"host", "content-length"}
        }
        if backend.data_parallel_rank is not None:
            result["x-data-parallel-rank"] = str(backend.data_parallel_rank)
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
