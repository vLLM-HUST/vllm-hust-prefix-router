"""Operator-owned Linux process identities for the managed native profile."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID


@dataclass(frozen=True)
class ProcessIdentity:
    """Distinguish PID reuse and namespace/boot changes without a host hook."""

    pid: int
    boot_id: str
    pid_namespace: str
    start_ticks: int

    @classmethod
    def capture(cls, pid: int) -> ProcessIdentity:
        if type(pid) is not int or pid <= 0:
            raise ValueError("process PID must be a positive integer")
        directory = Path(f"/proc/{pid}")
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            raise ProcessLookupError("process is no longer running")
        return cls(
            pid=pid,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            pid_namespace=os.readlink(directory / "ns/pid"),
            start_ticks=int(fields[19]),
        )

    def is_current(self) -> bool:
        try:
            return self == self.capture(self.pid)
        except (OSError, ValueError, IndexError):
            return False


@dataclass(frozen=True)
class ManagedBackendLease:
    """A trusted launcher assertion, not a publisher-issued cache snapshot.

    The launcher must write this only for a fresh engine, before admitting user
    requests, and use distinct event/replay endpoints for every incarnation.
    The file belongs to the operator and must not be writable by clients.
    """

    node_id: str
    generation: str
    api_process: ProcessIdentity
    engine_process: ProcessIdentity
    url: str
    event_endpoint: str
    replay_endpoint: str
    model: str
    hash_block_size: int
    hash_algorithm: str
    known_empty_initial_state: bool
    identity_digest: str

    def __post_init__(self) -> None:
        UUID(self.generation)
        if not self.node_id or not self.model:
            raise ValueError("lease node and model must be non-empty")
        if self.known_empty_initial_state is not True:
            raise ValueError("native bootstrap requires a known-empty engine")
        if type(self.hash_block_size) is not int or self.hash_block_size <= 0:
            raise ValueError("lease hash block size must be positive")
        if self.hash_algorithm != "sha256":
            raise ValueError("managed native profile currently requires sha256")
        if len(self.identity_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.identity_digest
        ):
            raise ValueError("lease identity_digest must be a SHA-256 digest")
        for endpoint, scheme in (
            (self.url, "http"),
            (self.event_endpoint, "tcp"),
            (self.replay_endpoint, "tcp"),
        ):
            parsed = urlsplit(endpoint)
            if (
                parsed.scheme != scheme
                or parsed.hostname != "127.0.0.1"
                or parsed.port is None
                or not 1 <= parsed.port <= 65535
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or parsed.path not in ("", "/")
                or (scheme == "tcp" and parsed.path)
            ):
                raise ValueError(
                    "managed native endpoints must be explicit loopback ports"
                )

    def is_current(self) -> bool:
        return self.api_process.is_current() and self.engine_process.is_current()

    @classmethod
    def read(cls, path: Path) -> ManagedBackendLease:
        with path.open("rb") as handle:
            payload = handle.read(65537)
        if len(payload) > 65536:
            raise ValueError("managed backend lease exceeds 64 KiB")
        raw = json.loads(payload)
        raw["api_process"] = ProcessIdentity(**raw["api_process"])
        raw["engine_process"] = ProcessIdentity(**raw["engine_process"])
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
