"""Validated configuration for the external router service."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class BackendPoolConfig:
    """Connection and admission limits applied independently per backend."""

    max_connections: int = 512
    max_pending_requests: int = 512
    queue_timeout_s: float = 1.0
    connect_timeout_s: float = 10.0
    request_timeout_s: float = 6 * 60 * 60
    keepalive_timeout_s: float = 30.0
    dns_cache_ttl_s: int = 300

    def __post_init__(self) -> None:
        for name in ("max_connections", "max_pending_requests", "dns_cache_ttl_s"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_connections == 0:
            raise ValueError("max_connections must be greater than zero")
        for name in (
            "queue_timeout_s",
            "connect_timeout_s",
            "request_timeout_s",
            "keepalive_timeout_s",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class BackendConfig:
    """One remote inference backend."""

    node_id: str
    url: str
    data_parallel_rank: int | None = None
    event_endpoint: str | None = None
    replay_endpoint: str | None = None
    event_topic: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    pool: BackendPoolConfig = field(default_factory=BackendPoolConfig)

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id must be non-empty")
        parsed = urlsplit(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("backend URL must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("backend URL must not contain a query or fragment")
        if self.data_parallel_rank is not None and (
            not isinstance(self.data_parallel_rank, int)
            or isinstance(self.data_parallel_rank, bool)
            or self.data_parallel_rank < 0
        ):
            raise ValueError("data_parallel_rank must be a non-negative integer")
        if (self.event_endpoint is None) != (self.replay_endpoint is None):
            raise ValueError(
                "event_endpoint and replay_endpoint must be configured together"
            )
        if self.event_endpoint is not None and not self.event_endpoint:
            raise ValueError("event_endpoint must be non-empty")
        if self.replay_endpoint is not None and not self.replay_endpoint:
            raise ValueError("replay_endpoint must be non-empty")
        if any(not key or "\n" in key or "\r" in key for key in self.headers):
            raise ValueError("backend header names must be non-empty single lines")
        if any("\n" in value or "\r" in value for value in self.headers.values()):
            raise ValueError("backend header values must be single lines")
