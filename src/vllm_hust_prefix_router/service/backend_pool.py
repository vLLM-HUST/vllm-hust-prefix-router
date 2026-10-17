"""Per-backend HTTP connection pools for long-lived generation streams."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp

from vllm_hust_prefix_router.config import BackendConfig
from vllm_hust_prefix_router.service.timing import RequestTiming


class BackendPoolSaturated(RuntimeError):
    """Raised before dispatch when a backend cannot admit another request."""


@dataclass(frozen=True)
class BackendPoolStats:
    """Point-in-time admission and queue statistics."""

    node_id: str
    active_requests: int
    queued_requests: int
    max_connections: int
    max_pending_requests: int
    admitted_requests: int
    rejected_requests: int
    total_queue_wait_s: float
    max_queue_wait_s: float


class BackendConnectionPool:
    """A connection pool and bounded admission queue for exactly one backend.

    Generation responses hold an HTTP/1.1 connection for their full streaming
    lifetime. A single shared aiohttp session therefore imposes its connector
    limit across every node. This class gives each node an independent connector
    and rejects excessive queueing before it can inflate TTFT for tens of seconds.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._slots = asyncio.Semaphore(config.pool.max_connections)
        self._active_requests = 0
        self._queued_requests = 0
        self._admitted_requests = 0
        self._rejected_requests = 0
        self._total_queue_wait_s = 0.0
        self._max_queue_wait_s = 0.0

    @property
    def session(self) -> aiohttp.ClientSession:
        """Return the started client session."""
        if self._session is None or self._session.closed:
            raise RuntimeError(f"backend pool {self.config.node_id!r} is not started")
        return self._session

    async def start(self) -> None:
        """Create this backend's independent keep-alive connector."""
        if self._session is not None and not self._session.closed:
            return
        pool = self.config.pool
        connector = aiohttp.TCPConnector(
            limit=pool.max_connections,
            limit_per_host=pool.max_connections,
            keepalive_timeout=pool.keepalive_timeout_s,
            ttl_dns_cache=pool.dns_cache_ttl_s,
        )
        timeout = aiohttp.ClientTimeout(
            total=pool.request_timeout_s,
            connect=pool.connect_timeout_s,
            sock_connect=pool.connect_timeout_s,
            sock_read=pool.request_timeout_s,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            auto_decompress=False,
        )

    async def close(self) -> None:
        """Close idle and active connector resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def stats(self) -> BackendPoolStats:
        """Return current admission metrics without exposing aiohttp internals."""
        return BackendPoolStats(
            node_id=self.config.node_id,
            active_requests=self._active_requests,
            queued_requests=self._queued_requests,
            max_connections=self.config.pool.max_connections,
            max_pending_requests=self.config.pool.max_pending_requests,
            admitted_requests=self._admitted_requests,
            rejected_requests=self._rejected_requests,
            total_queue_wait_s=self._total_queue_wait_s,
            max_queue_wait_s=self._max_queue_wait_s,
        )

    @asynccontextmanager
    async def _admit(self, timing: RequestTiming | None = None) -> AsyncIterator[None]:
        pool = self.config.pool
        if timing is not None:
            timing.mark("admission_start")
        if self._slots.locked() and self._queued_requests >= pool.max_pending_requests:
            self._rejected_requests += 1
            raise BackendPoolSaturated(
                f"backend {self.config.node_id!r} admission queue is full"
            )

        started_at = time.perf_counter()
        self._queued_requests += 1
        try:
            try:
                await asyncio.wait_for(
                    self._slots.acquire(), timeout=pool.queue_timeout_s
                )
            except asyncio.TimeoutError as exc:
                self._rejected_requests += 1
                raise BackendPoolSaturated(
                    f"backend {self.config.node_id!r} admission timed out"
                ) from exc
        finally:
            self._queued_requests -= 1

        queue_wait_s = time.perf_counter() - started_at
        self._total_queue_wait_s += queue_wait_s
        self._max_queue_wait_s = max(self._max_queue_wait_s, queue_wait_s)
        self._active_requests += 1
        self._admitted_requests += 1
        if timing is not None:
            timing.mark("pool_admitted")
        try:
            yield
        finally:
            self._active_requests -= 1
            self._slots.release()

    def _target_url(self, path: str) -> str:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("upstream path must be an absolute path, not a URL")
        return f"{self.config.url.rstrip('/')}{path}"

    @asynccontextmanager
    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        timing: RequestTiming | None = None,
        before_dispatch: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """Dispatch one request while holding an admission slot through streaming."""
        merged_headers = dict(headers or {})
        if self.config.headers:
            rank_header = "x-data-parallel-rank"
            selected_rank = next(
                (
                    value
                    for key, value in merged_headers.items()
                    if key.lower() == rank_header
                ),
                None,
            )
            merged_headers.update(self.config.headers)
            if selected_rank is not None:
                # Backend credentials still override client headers, but a static
                # header must not replace the Router's selected rank or add a
                # second differently cased copy of that routing header.
                merged_headers = {
                    key: value
                    for key, value in merged_headers.items()
                    if key.lower() != rank_header
                }
                merged_headers[rank_header] = selected_rank
        async with self._admit(timing):
            # Admission can yield while an incarnation or cache view changes.
            # The synchronous guard runs with the slot held and may refuse the
            # request before constructing its HTTP transport. It cannot retry.
            if before_dispatch is not None:
                before_dispatch()
            async with self.session.request(
                method,
                self._target_url(path),
                headers=merged_headers,
                **kwargs,
            ) as response:
                if timing is not None:
                    timing.mark("upstream_headers")
                yield response


class BackendPoolRegistry:
    """Own one isolated connection pool per configured remote backend."""

    def __init__(self, backends: list[BackendConfig]) -> None:
        node_ids = [backend.node_id for backend in backends]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("backend node IDs must be unique")
        if not backends:
            raise ValueError("at least one backend is required")
        self._pools = {
            backend.node_id: BackendConnectionPool(backend) for backend in backends
        }

    async def start(self) -> None:
        """Start every backend pool."""
        await asyncio.gather(*(pool.start() for pool in self._pools.values()))

    async def close(self) -> None:
        """Close every backend pool."""
        await asyncio.gather(
            *(pool.close() for pool in self._pools.values()),
            return_exceptions=True,
        )

    def pool(self, node_id: str) -> BackendConnectionPool:
        """Return a backend pool by node ID."""
        try:
            return self._pools[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown backend {node_id!r}") from exc

    def stats(self) -> dict[str, BackendPoolStats]:
        """Return point-in-time statistics for every backend."""
        return {node_id: pool.stats() for node_id, pool in self._pools.items()}
