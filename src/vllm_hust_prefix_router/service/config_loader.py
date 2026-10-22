"""Strict JSON configuration loading and runtime assembly."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_hust_prefix_router.adapters.vllm_hust.kv_events import (
    VllmHustKvEventCodec,
)
from vllm_hust_prefix_router.adapters.vllm_hust.request_fingerprint import (
    VllmHustFingerprintConfig,
    VllmHustTextFingerprinter,
)
from vllm_hust_prefix_router.config import BackendConfig, BackendPoolConfig
from vllm_hust_prefix_router.core.lifecycle import (
    LifecycleLoadTracker,
    LifecycleRoutingConfig,
)
from vllm_hust_prefix_router.core.planner import RoutePlanner
from vllm_hust_prefix_router.core.prefix_index import GlobalPrefixIndex
from vllm_hust_prefix_router.events.native_replay import (
    NativeReplayConfig,
    NativeReplaySubscriber,
)
from vllm_hust_prefix_router.events.zmq_recovery import (
    ZmqEventSourceConfig,
    ZmqEventSubscriber,
)
from vllm_hust_prefix_router.service.app import RouterService, RouterServiceConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Fully assembled service plus listen address."""

    service: RouterService
    host: str
    port: int


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load a JSON object and construct a fail-closed Router runtime."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Router configuration must be a JSON object")
    allowed = {
        "listen",
        "policy",
        "hash_block_size",
        "default_backend",
        "backends",
        "fingerprint",
        "lifecycle",
        "allow_unversioned_replay",
        "max_request_body_size",
        "runtime_profile",
        "native_replay",
        "require_trusted_cache_view",
        "observe_request_timings",
    }
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported Router configuration keys: {sorted(unknown)}")
    profile = raw.get("runtime_profile", "public")
    if profile not in {
        "public",
        "pr2-enhanced",
        "managed-native-0fc695f-v1",
        "managed-hust-cedb477-v1",
    }:
        raise ValueError("unsupported runtime_profile")
    native = profile.startswith("managed-")
    if not native and "native_replay" in raw:
        raise ValueError("native_replay requires the managed native profile")
    allow_unversioned = raw.get("allow_unversioned_replay", False)
    if not isinstance(allow_unversioned, bool):
        raise ValueError("allow_unversioned_replay must be a boolean")
    if allow_unversioned and (
        profile != "public" or raw.get("require_trusted_cache_view", False)
    ):
        raise ValueError(
            "unversioned best-effort replay requires public profile "
            "and non-strict admission"
        )

    listen = _mapping(raw.get("listen", {}), "listen")
    host = listen.get("host", "127.0.0.1")
    port = listen.get("port", 8000)
    if not isinstance(host, str) or not host:
        raise ValueError("listen.host must be a non-empty string")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("listen.port must be an integer between 1 and 65535")

    block_size = raw.get("hash_block_size")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("hash_block_size must be a positive integer")
    policy = raw.get("policy", "prefix")
    if policy not in ("prefix", "lifecycle"):
        raise ValueError("policy must be 'prefix' or 'lifecycle'")

    raw_backends = raw.get("backends")
    if not isinstance(raw_backends, list) or not raw_backends:
        raise ValueError("backends must be a non-empty list")
    backends = tuple(_backend(value) for value in raw_backends)

    prefix_index = GlobalPrefixIndex()

    lifecycle_tracker = None
    if policy == "lifecycle":
        lifecycle_values = _mapping(raw.get("lifecycle", {}), "lifecycle")
        unknown_lifecycle = (
            lifecycle_values.keys() - LifecycleRoutingConfig.__dataclass_fields__.keys()
        )
        if unknown_lifecycle:
            raise ValueError(f"unsupported lifecycle keys: {sorted(unknown_lifecycle)}")
        lifecycle_tracker = LifecycleLoadTracker(
            block_size=block_size,
            config=LifecycleRoutingConfig(**lifecycle_values),
        )

    fingerprint_values = _mapping(raw.get("fingerprint"), "fingerprint")
    if native:
        from vllm_hust_prefix_router.adapters.vllm_hust.native_profile import (
            NativeKvEventCodec,
            model_identity,
        )

        native_host_range = "==0.23.0+empty"
        if profile == "managed-hust-cedb477-v1":
            from vllm_hust_prefix_router.adapters.vllm_hust.hust_native_profile import (
                HOST_VERSION,
                HustNativeKvEventCodec,
                model_identity,
            )

            NativeKvEventCodec = HustNativeKvEventCodec
            native_host_range = f"=={HOST_VERSION}"

        if (
            block_size != 128
            or fingerprint_values.get("hash_algorithm", "sha256") != "sha256"
        ):
            raise ValueError("native profile requires 128-token blocks and sha256")
        if "host_version_range" in fingerprint_values:
            raise ValueError("native profile pins the host; omit host_version_range")
        event_codec = NativeKvEventCodec()
    fingerprint_config = VllmHustFingerprintConfig(
        tokenizer=_required_string(fingerprint_values, "tokenizer"),
        hash_block_size=block_size,
        hash_algorithm=fingerprint_values.get(
            "hash_algorithm", "xxhash" if profile == "pr2-enhanced" else "sha256"
        ),
        host_version_range=fingerprint_values.get(
            "host_version_range",
            native_host_range
            if native
            else (">=0.23.1,<0.24" if profile == "pr2-enhanced" else ">=0.25.1,<0.30"),
        ),
        tokenizer_revision=fingerprint_values.get("tokenizer_revision"),
        trust_remote_code=fingerprint_values.get("trust_remote_code", False),
    )
    fingerprinter = VllmHustTextFingerprinter(fingerprint_config)
    planner = RoutePlanner(
        prefix_index,
        policy=policy,
        block_size=block_size,
        lifecycle_tracker=lifecycle_tracker,
    )
    default_backend = raw.get("default_backend", backends[0].node_id)
    if not isinstance(default_backend, str):
        raise ValueError("default_backend must be a string")
    if not native:
        event_codec = VllmHustKvEventCodec(fingerprint_config.host_version_range)
        if not event_codec.snapshot_replay and not allow_unversioned:
            raise ValueError(
                "the host has no snapshot/epoch replay; set "
                "allow_unversioned_replay=true only for best-effort deployments"
            )
    for backend in backends:
        if native and backend.group_block_sizes not in ({}, {0: 128}):
            raise ValueError(
                "managed native profile supports only KV group 0 of size 128"
            )
        if (
            not native
            and not event_codec.snapshot_replay
            and (not backend.group_block_sizes or backend.data_parallel_rank is None)
        ):
            raise ValueError(
                "unversioned replay requires group_block_sizes and data_parallel_rank"
            )
        prefix_index.register_node(
            backend.node_id,
            hash_block_size=block_size,
            data_parallel_rank=backend.data_parallel_rank,
            group_block_sizes=backend.group_block_sizes or {0: block_size},
        )
    native_values = _mapping(raw.get("native_replay", {}), "native_replay")
    timing_keys = {
        "timeout_s",
        "interval_s",
        "view_ttl_s",
        "max_batches",
        "max_bytes",
        "observe_replay",
    }
    if native_values.keys() - timing_keys:
        raise ValueError("unsupported native_replay keys")
    event_sources = []
    for backend in backends:
        if native:
            if backend.managed_lease_path is None or backend.data_parallel_rank != 0:
                raise ValueError(
                    "native backend requires a managed lease and explicit rank 0"
                )
            event_sources.append(
                NativeReplaySubscriber(
                    NativeReplayConfig(
                        node_id=backend.node_id,
                        url=backend.url,
                        lease_path=Path(backend.managed_lease_path),
                        model=fingerprint_config.tokenizer,
                        hash_block_size=block_size,
                        identity_digest=model_identity(fingerprint_config.tokenizer),
                        reply_delimiter=profile == "managed-hust-cedb477-v1",
                        **native_values,
                    ),
                    index=prefix_index,
                    codec=event_codec,
                )
            )
            continue
        if backend.managed_lease_path is not None:
            raise ValueError("managed lease requires the native runtime profile")
        if backend.event_endpoint is None or backend.replay_endpoint is None:
            raise ValueError(
                f"backend {backend.node_id!r} requires event and replay endpoints"
            )
        event_sources.append(
            ZmqEventSubscriber(
                ZmqEventSourceConfig(
                    node_id=backend.node_id,
                    event_endpoint=backend.event_endpoint,
                    replay_endpoint=backend.replay_endpoint,
                    data_parallel_rank=backend.data_parallel_rank,
                    topic=backend.event_topic,
                ),
                index=prefix_index,
                codec=event_codec,
            )
        )
    service = RouterService(
        RouterServiceConfig(
            backends=backends,
            default_backend=default_backend,
            max_request_body_size=raw.get("max_request_body_size", 16 * 1024 * 1024),
            require_trusted_cache_view=raw.get("require_trusted_cache_view", False),
            observe_request_timings=raw.get("observe_request_timings", False),
        ),
        planner=planner,
        fingerprinter=fingerprinter,
        event_sources=tuple(event_sources),
    )
    return RuntimeConfig(service=service, host=host, port=port)


def _backend(raw: Any) -> BackendConfig:
    value = _mapping(raw, "backend")
    allowed = {
        "id",
        "url",
        "data_parallel_rank",
        "event_endpoint",
        "replay_endpoint",
        "event_topic",
        "group_block_sizes",
        "headers_from_env",
        "pool",
        "managed_lease_path",
    }
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported backend keys: {sorted(unknown)}")
    pool_values = _mapping(value.get("pool", {}), "backend.pool")
    unknown_pool = pool_values.keys() - BackendPoolConfig.__dataclass_fields__.keys()
    if unknown_pool:
        raise ValueError(f"unsupported backend pool keys: {sorted(unknown_pool)}")
    env_headers = _mapping(value.get("headers_from_env", {}), "headers_from_env")
    raw_group_sizes = _mapping(value.get("group_block_sizes", {}), "group_block_sizes")
    try:
        group_block_sizes = {
            int(group): size for group, size in raw_group_sizes.items()
        }
    except (ValueError, TypeError) as exc:
        raise ValueError("group_block_sizes keys must be integer strings") from exc
    if len(group_block_sizes) != len(raw_group_sizes) or any(
        str(group) not in raw_group_sizes for group in group_block_sizes
    ):
        raise ValueError("group_block_sizes keys must be canonical integer strings")
    headers: dict[str, str] = {}
    for header, environment_name in env_headers.items():
        if not isinstance(header, str) or not isinstance(environment_name, str):
            raise ValueError("headers_from_env must map strings to environment names")
        try:
            headers[header] = os.environ[environment_name]
        except KeyError as exc:
            raise ValueError(
                f"required backend environment variable {environment_name!r} is unset"
            ) from exc
    return BackendConfig(
        node_id=_required_string(value, "id"),
        url=_required_string(value, "url"),
        data_parallel_rank=value.get("data_parallel_rank"),
        event_endpoint=value.get("event_endpoint"),
        replay_endpoint=value.get("replay_endpoint"),
        event_topic=value.get("event_topic", ""),
        group_block_sizes=group_block_sizes,
        headers=headers,
        pool=BackendPoolConfig(**pool_values),
        managed_lease_path=value.get("managed_lease_path"),
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _required_string(mapping: dict[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
