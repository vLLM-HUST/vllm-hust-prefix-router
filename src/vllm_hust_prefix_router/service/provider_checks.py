"""Read-only deployment checks used by the Extension Manager provider."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from vllm_hust_prefix_router._version import __version__

EXTENSION_ID = "org.vllm-hust.prefix-router"


@dataclass(frozen=True)
class DeploymentBinding:
    router_url: str
    config_path: Path
    config_sha256: str
    backend_endpoints: dict[str, str]
    kv_event_endpoints: dict[str, dict[str, str]]
    backend_headers: dict[str, dict[str, str]]
    best_effort_replay: bool


@dataclass(frozen=True)
class ProbeResult:
    reachable: bool
    healthy: bool
    degraded: bool
    evidence: tuple[str, ...]


def _http_url(value: Any, name: str, *, root_only: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an HTTP(S) URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (root_only and parsed.path not in {"", "/"})
    ):
        raise ValueError(f"{name} must be an HTTP(S) URL without credentials or query")
    return value.rstrip("/")


def validate_binding(configuration: dict[str, Any]) -> DeploymentBinding:
    """Bind Manager service claims to the exact Router config on disk."""
    router_url = _http_url(
        configuration.get("router_url"), "router_url", root_only=True
    )
    config_value = configuration.get("router_config_path")
    if not isinstance(config_value, str) or not Path(config_value).is_absolute():
        raise ValueError("router_config_path must be an absolute file path")
    config_path = Path(config_value)
    raw_bytes = config_path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or not isinstance(raw.get("backends"), list):
        raise ValueError("Router config must contain a backends array")

    backends: dict[str, str] = {}
    events: dict[str, dict[str, str]] = {}
    headers: dict[str, dict[str, str]] = {}
    for item in raw["backends"]:
        if not isinstance(item, dict):
            raise ValueError("Router backends must be objects")
        node_id = item.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in backends:
            raise ValueError("Router backend IDs must be non-empty and unique")
        _http_url(item.get("url"), f"backend {node_id} URL")
        backends[node_id] = item["url"]
        event = item.get("event_endpoint")
        replay = item.get("replay_endpoint")
        if not all(isinstance(value, str) and value for value in (event, replay)):
            raise ValueError(f"backend {node_id} needs event and replay endpoints")
        events[node_id] = {"event_endpoint": event, "replay_endpoint": replay}
        env_headers = item.get("headers_from_env", {})
        if not isinstance(env_headers, dict):
            raise ValueError(f"backend {node_id} headers_from_env must be an object")
        node_headers: dict[str, str] = {}
        for header, env_name in env_headers.items():
            if not isinstance(header, str) or not isinstance(env_name, str):
                raise ValueError("headers_from_env must map strings to strings")
            if env_name not in os.environ:
                raise ValueError(
                    f"backend {node_id} requires environment variable {env_name}"
                )
            node_headers[header] = os.environ[env_name]
        headers[node_id] = node_headers
    if not backends:
        raise ValueError("Router config must contain at least one backend")
    if configuration.get("backend_endpoints") != backends:
        raise ValueError("backend_endpoints must exactly match the Router config")
    if configuration.get("kv_event_endpoints") != events:
        raise ValueError("kv_event_endpoints must exactly match the Router config")
    if not isinstance(raw.get("allow_unversioned_replay", False), bool):
        raise ValueError("allow_unversioned_replay must be a boolean")
    return DeploymentBinding(
        router_url=router_url,
        config_path=config_path,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        backend_endpoints=backends,
        kv_event_endpoints=events,
        backend_headers=headers,
        best_effort_replay=raw.get("allow_unversioned_replay", False),
    )


def _get(url: str, *, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    # Health checks must target the configured service directly, not an ambient proxy.
    opener = build_opener(ProxyHandler({}))
    request = Request(url, headers=headers or {}, method="GET")
    try:
        with opener.open(request, timeout=2) as response:  # noqa: S310
            payload = response.read(1024 * 1024)
            return response.status, payload
    except HTTPError as error:
        return error.code, b""


def probe_deployment(binding: DeploymentBinding) -> ProbeResult:
    """Check process identity, KV recovery, and every required HTTP backend."""
    evidence: list[str] = []
    try:
        status, body = _get(f"{binding.router_url}/healthz")
    except (OSError, URLError) as error:
        return ProbeResult(False, False, True, (f"Router is unreachable: {error}",))
    if status != 200:
        return ProbeResult(True, False, True, (f"Router healthz returned {status}",))
    try:
        health = json.loads(body)
    except (TypeError, ValueError):
        return ProbeResult(True, False, True, ("Router healthz is not JSON",))
    if not isinstance(health, dict) or any(
        health.get(key) != expected
        for key, expected in (
            ("status", "ok"),
            ("extension_id", EXTENSION_ID),
            ("version", __version__),
            ("config_sha256", binding.config_sha256),
        )
    ):
        return ProbeResult(
            True, False, True, ("Router identity or running config does not match",)
        )
    evidence.append("Router process identity and config digest match")

    try:
        ready_status, ready_body = _get(f"{binding.router_url}/readyz")
        metrics_status, metrics_body = _get(f"{binding.router_url}/metrics")
    except (OSError, URLError) as error:
        return ProbeResult(
            True, False, True, (*evidence, f"Router probe failed: {error}")
        )
    try:
        ready = json.loads(ready_body)
        metrics = json.loads(metrics_body)
    except (TypeError, ValueError):
        return ProbeResult(
            True, False, True, (*evidence, "Router readiness is not JSON")
        )
    if ready_status != 200 or ready != {"status": "ready"}:
        return ProbeResult(
            True, False, True, (*evidence, "Router KV recovery is not ready")
        )
    if metrics_status != 200 or not isinstance(metrics, dict):
        return ProbeResult(
            True, False, True, (*evidence, "Router metrics are unavailable")
        )
    observed_backends = metrics.get("backends")
    if not isinstance(observed_backends, dict) or set(observed_backends) != set(
        binding.backend_endpoints
    ):
        return ProbeResult(True, False, True, (*evidence, "Router backend set differs"))
    sources = metrics.get("event_sources")
    if not isinstance(sources, list) or len(sources) != len(binding.kv_event_endpoints):
        return ProbeResult(
            True, False, True, (*evidence, "Router KV source set differs")
        )
    source_map = {
        source.get("node_id"): source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("node_id"), str)
    }
    if set(source_map) != set(binding.kv_event_endpoints) or any(
        source.get("ready") is not True
        or source.get("last_error") is not None
        or type(source.get("next_seq")) is not int
        for source in source_map.values()
    ):
        return ProbeResult(
            True, False, True, (*evidence, "KV event recovery is degraded")
        )
    evidence.append(f"{len(source_map)} KV event source(s) reconciled")

    for node_id, url in binding.backend_endpoints.items():
        try:
            backend_status, _ = _get(
                f"{url.rstrip('/')}/health",
                headers=binding.backend_headers[node_id],
            )
        except (OSError, URLError) as error:
            return ProbeResult(
                True,
                False,
                True,
                (*evidence, f"backend {node_id} is unreachable: {error}"),
            )
        if backend_status != 200:
            return ProbeResult(
                True,
                False,
                True,
                (*evidence, f"backend {node_id} health returned {backend_status}"),
            )
    evidence.append(f"{len(binding.backend_endpoints)} OpenAI backend(s) healthy")
    if binding.best_effort_replay:
        evidence.append("unversioned KV replay is best effort")
    return ProbeResult(True, True, binding.best_effort_replay, tuple(evidence))
