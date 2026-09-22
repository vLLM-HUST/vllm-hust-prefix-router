# Provenance

Source archive:
[intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831)

Primary history:

- [PR #80: initial prefix routing and global scheduler](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/80)
- [PR #170: merge-gate completion](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/170)
- [PR #173: final integrated implementation](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/173)
- [PR #225: runtime fault hardening](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/225)
- [PR #258: per-node load metrics](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/258)
- [PR #272: generation-scoped route fence draft](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/272)

No source is considered migrated until its exact commit, author, license, and
tests are recorded here.

## Authoritative migration source

The 2026-09 migration uses the local cleaned `codex/lifecycle-load-routing`
worktree as its only source. It was reconstructed from:

- parent commit: `536116cc8b131bec30629b2a9f6d1beb6d018ff8`;
- tracked binary patch SHA-256:
  `ec0bcd2d9bde16f07181e1f4d281f81d5ecab1882c7810c03ba669fa3555400a`;
- immutable local source-freeze commit:
  `4a1a1c6e03f90b2b90bbc5022526fc4d9739e845`.

The server experiment worktree is validation evidence, not a source tree.

## Migrated files

| Destination | Source | Treatment | Tests |
|---|---|---|---|
| `core/lifecycle.py` | `vllm/distributed/lifecycle_routing.py` at source-freeze commit | Exact framework-neutral extraction | `tests/test_lifecycle.py` |
| `core/prefix_index.py` | `vllm/distributed/prefix_scheduler.py` at source-freeze commit | vLLM event/hash types replaced by package protocols; untrusted recovery state added | `tests/test_prefix_index.py` |
| `protocols/cache_events.py` | Event semantics used by the same Prefix scheduler | New stable package DTO boundary | `tests/test_prefix_index.py` |
| `service/backend_pool.py` | New implementation prompted by remote connector saturation evidence | Independent per-backend connector and bounded admission | `tests/test_backend_pool.py` |

All extracted source remains under Apache-2.0 and retains the original SPDX
headers. New files are also licensed Apache-2.0.

## Explicitly excluded

The migration does not include Scheduler snapshot load supplementation, V1/V2
cost models, dynamic Prefill-rate estimation, calibration, fixed-duration
reservations, TTFT/transfer routing models, or their dedicated tests.

