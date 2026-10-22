# Reliability scope and compatibility

This is an external-Router increment on public PR #4 (`main@04142b2`), not a new
Router implementation. Existing LPM, HTTP forwarding, connection pools and
lifecycle load tracking keep their owners. The public manifest is unchanged:
ECPA 0.3, `import_only`, `lifecycle_owner=user`. Discovery is not activation.

## Mechanisms retained

| Concern | Implementation / regression coverage |
| --- | --- |
| Incomplete/gapped/stale event history | Stage replay in an isolated view, publish atomically only after validation; invalidate on failed supervision |
| Incarnation/ABA and delayed completion | Operator-owned generation, API/engine process identities, replay attempt/watermark and retired identity checks; closing a source rejects late publication |
| Request lifecycle | Distinct internal request IDs, cancellation-safe cleanup, bounded pools and receipts; no second response or automatic POST retry after response commitment |
| Cache trust | Expired/dead/mismatched sources cannot supply cache scores; optional strict admission refuses untrusted views |
| Observation | Backend-reported cached-token usage stays separate from predicted cache hints; absent usage remains unknown; optional timings do not govern routing |

`require_trusted_cache_view=false` retains safe fallback forwarding after removing
untrusted cache hints. `true` refuses untrusted admission. These are two admission
modes using the same Router, not new routing algorithms. Test-only retained-hint
shadows are negative oracles, never a production policy.

Receipts include generation/attempt context and whether it was still current at
finish. They cannot retract already generated tokens or prove engine-internal
exactly-once execution. `/metrics` retains only 256 recent diagnostic receipts;
use a protected endpoint and external durable recording when needed.

After waiting for a connection slot, the Router checks the selected source again
before constructing the upstream request. Strict mode rejects a lost view or a
changed generation/model identity with 503. Ordinary same-generation replay
watermark advancement is allowed. Safe fallback may still forward to the chosen
HTTP backend, but clears the stale predicted-hit value. Receipts distinguish
`selection_generation`, the refreshed `generation`, `dispatch_attempted` and
`cache_view_current_at_dispatch`. Attempted dispatch means HTTP transport was
entered, not engine acceptance. This check is not atomic with engine execution;
an incarnation may still change during network connection or inference.
`predicted_cached_tokens` is the selection-time forecast, retained only when the
same source remains trusted. The refreshed view watermark describes the dispatch
check; subsequent same-generation cache deltas can still change actual reuse.

## Profiles are separate contracts

| Profile | Guard / qualification boundary |
| --- | --- |
| `public` (default) | PR #4 host adapter range `>=0.25.1,<0.30`, SHA-256; requires snapshot/epoch replay unless best-effort is explicitly enabled |
| `pr2-enhanced` (explicit legacy compatibility) | Original PR #2 enhanced event contract, default host range `>=0.23.1,<0.24`, xxhash; no unversioned opt-in |
| `managed-native-0fc695f-v1` | Historical source-pinned native adapter; retained because current helpers reuse its identity/normalization code and the existing config supports it; not recommended as a new deployment target |
| `managed-hust-cedb477-v1` | Core `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, Ascend `b47795c63a8e63e7aed4d311f951428d94e045ab`; metadata plus installed-interface SHA guards |

The historical and PR #4 integration native runs use Qwen2.5-14B-Instruct, FP16/eager, two TP1 instances,
Python 3.12.13 on the frozen HUST environment. Its distribution versions are
`vllm==0.30.0+empty` and
`vllm-ascend==0.25.1rc2.dev766+hust.20260903.4.gb47795c63`.
The managed HUST profile uses 128-token blocks, sha256/full-byte identities,
`PYTHONHASHSEED=0`, `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`, rank 0 and audited
local full-attention cache semantics. Unknown builds, tiers and identities are
rejected. This does not certify every byte of an engine build or arbitrary newer
versions; upgrade qualification is a separate change, not a relaxed version guard.

The public adapter and managed profile have different trust prerequisites.
`allow_unversioned_replay=true` is accepted only by the public profile with
`require_trusted_cache_view=false`. Its readiness means the partial sequence
stream is usable for best-effort hints, not verified publisher identity or full
recovery. Source status exposes `recovery_basis=unversioned_best_effort` versus
`publisher_snapshot_epoch`. The managed profile retains protected fresh-engine
leases and PID checks; neither path supplies an engine-atomic fence.

New configurations without `runtime_profile` follow PR #4 defaults. To retain
the previous enhanced-protocol configuration, explicitly set
`runtime_profile=pr2-enhanced`. Do not copy old implicit defaults into a new
deployment. Public `group_block_sizes` must describe every actual KV group;
the frozen managed profile rejects layouts other than group 0, size 128.
Complete publisher snapshots may replace the layout atomically; delta batches
cannot silently add groups or change sizes. Snapshot layout inference still
depends on the publisher supplying a complete, authoritative image.
If a malformed live batch fails after partially applying its deltas, the recovery
state machine clears the partial base and requires reconstruction. An enhanced
publisher must supply a full snapshot; best-effort replay restarts from the
beginning of its retained window. An empty heartbeat cannot restore trust in
that partial base. Normal live updates do not copy the entire index.

The selected DP rank is used consistently in Lifecycle dispatch, upstream
headers and receipts. Client-supplied rank headers cannot override the Router's
choice. TP rank is not DP rank. PR #4's Lifecycle weights remain unchanged;
comparison with older wheels must record routing decisions as well as latency.
Configured backend headers also cannot replace a selected rank or add another
case-variant routing header. Other backend headers retain their existing
precedence, including credentials overriding client-supplied values.

The managed profile requires same-machine, same PID-namespace Linux processes
and explicit loopback HTTP/ZMQ endpoints. An operator supplies a protected lease
only for a newly launched, known-empty engine, before admitting user requests.
Use a fresh generation UUID and fresh event/replay endpoints for each incarnation;
capture both API and engine PIDs with `ProcessIdentity.capture`, never copied PID
numbers. Atomically replace the lease file when publishing a new incarnation.
Its node/model/block/hash and identity digest must agree with the Router config.
The lease file is trusted operator input, not an authenticated engine-issued
cache snapshot. Do not manufacture a new lease for an arbitrary already-used
engine or expose lease writes to clients. Retirement tracking is bounded to 1024
incarnations per subscriber lifetime; exhaustion fails closed.

## Explicit exclusions

Private optimized-session-gate registration, another HTTP publisher, adaptive
arrival control, reservation mechanisms and old performance claims are not
imported. Legacy Issues #14/#15 contributed fault categories: arrivals, stale
views, restart, ABA, retry, delayed completion and partial replay. The equivalents
are tested against this Router, not cherry-picked from unverified student code.
Legacy #17/#18 bootstrap work is not a dependency of this delivery.

There is no atomic engine pre-reuse hook or cross-worker KV-byte transfer path.
Do not claim actual reuse is unconditionally correct, formal P99/throughput gains,
load-balancing benefit, full MOD automatic activation or support for untested
models/TP/topologies. [Evidence boundaries](reliability-evidence.md) are part of
the acceptance contract, not optional caveats.

## Acceptance status and remaining work

The implementation retains PR #4's public protocol/configuration ownership.
Core/Ascend sources and the frozen serving environment are unchanged. Manager
remains `import_only`, with user-owned processes.

- Completed locally: public recovery regression, managed replay/request lifecycle,
  deterministic CPU faults, wheel/sdist and pinned Manager isolation checks.
- Recorded native evidence: frozen two-TP1 fault runs and a later final-wheel
  text/chat normal-path follow-up. These have different artifact identities;
  see [the evidence index](reliability-evidence.md) for counts, hashes and limits.
- Unresolved: the original survivor replay timeout during peer reload. Passing
  repeats do not establish its cause or justify timeout relaxation, a CPU-count
  floor or incident-free restart claims.
- Before publication: recheck target main and final diff, arrange access for any
  native evidence used in the PR, and obtain owner confirmation to push.
  Remote CI and maintainer review remain separate, uncompleted gates.

Documentation/package cleanup alone does not require repeating unchanged native
experiments. Runtime changes require affected-path requalification; never reuse
an old wheel's result under a different hash without an explicit byte/scope audit.

New engine pairs and Qwen3.5 TP2 remain separate qualifications. Public
multi-group support does not certify the managed single-group profile for hybrid
models. Real multi-DP routing, engine-atomic fencing, KV-byte transfer guarantees
and Manager automatic activation are not certified here. Two TP1 backends need
two devices; two TP2 backends require four.
