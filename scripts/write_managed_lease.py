"""Publish an operator assertion for a newly launched, unused HUST engine.

Run with the serving host Python and installed Router. This does not launch an
engine or establish that it is empty; the operator must enforce that boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from vllm_hust_prefix_router.adapters.vllm_hust.hust_native_profile import (
    PROFILE,
    model_identity,
    verify_hust_native_profile,
)
from vllm_hust_prefix_router.events.managed_identity import (
    ManagedBackendLease,
    ProcessIdentity,
)


def make_lease(config: dict, node: str, api_pid: int, engine_pid: int):
    if config.get("runtime_profile") != PROFILE or config.get("hash_block_size") != 128:
        raise ValueError("lease helper requires the pinned HUST 128-token profile")
    fingerprint = config["fingerprint"]
    if fingerprint.get("hash_algorithm", "sha256") != "sha256":
        raise ValueError("lease helper requires sha256")
    backends = [b for b in config["backends"] if b["id"] == node]
    if len(backends) != 1 or backends[0].get("data_parallel_rank") != 0:
        raise ValueError("one configured backend with explicit rank 0 is required")
    backend = backends[0]
    path = Path(backend["managed_lease_path"])
    if not path.is_absolute():
        raise ValueError("lease path must be absolute")
    if api_pid == engine_pid:
        raise ValueError("API and engine must have distinct process identities")
    api, engine = ProcessIdentity.capture(api_pid), ProcessIdentity.capture(engine_pid)
    lease = ManagedBackendLease(
        node_id=node,
        generation=str(uuid4()),
        api_process=api,
        engine_process=engine,
        url=backend["url"],
        event_endpoint=backend["event_endpoint"],
        replay_endpoint=backend["replay_endpoint"],
        model=fingerprint["tokenizer"],
        hash_block_size=128,
        hash_algorithm="sha256",
        known_empty_initial_state=True,
        identity_digest=model_identity(fingerprint["tokenizer"]),
    )
    if not lease.is_current():
        raise ProcessLookupError("API or engine identity changed during lease capture")
    return lease, path


def publish(lease: ManagedBackendLease, path: Path) -> None:
    directory = path.parent
    if directory.is_symlink() or directory.stat().st_mode & 0o022:
        raise ValueError("lease directory must not be writable by group or other users")
    if directory.stat().st_uid != os.getuid():
        raise ValueError("lease directory must belong to the operator")
    # Each incarnation gets a new protected path. Existing leases are never
    # overwritten; hard-link publication is atomic and refuses a raced target.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".lease-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            json.dump(lease.to_dict(), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not lease.is_current():
            raise ProcessLookupError(
                "API or engine identity changed before publication"
            )
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--api-pid", type=int, required=True)
    parser.add_argument("--engine-pid", type=int, required=True)
    parser.add_argument("--confirm-new-unused-engine", action="store_true")
    args = parser.parse_args()
    if not args.confirm_new_unused_engine or os.name != "posix":
        parser.error("a Linux operator must confirm a new engine with no user traffic")
    verify_hust_native_profile()
    lease, path = make_lease(
        json.loads(args.config.read_bytes()),
        args.node,
        args.api_pid,
        args.engine_pid,
    )
    publish(lease, path)
    print(json.dumps({"lease": str(path), "generation": lease.generation}))


if __name__ == "__main__":
    main()
