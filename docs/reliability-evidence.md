# Evidence summary and review boundary

This unreleased reliability increment is based on public PR #4,
`main@04142b28e4962321c23b0228331745e3297a5378`. Local checks are not remote CI or
maintainer acceptance. Native records below are owner-held background evidence
until reviewer access is arranged; archive names are not download links.

## Reproducible acceptance gates

Run the commands in [delivery checks](reliability-delivery.md) from the reviewed
source and an isolated installed wheel. Keep skips separate from passes.

| Gate | Recorded result and scope |
| --- | --- |
| Standalone public recovery increment `00e5e76` | 94 CPU tests passed on Windows/Python 3.14.5 |
| Complete source `ab8dcc5`, Windows/Python 3.14.5 | 270 passed, six Linux-only skips; unpacked sdist has the same result |
| Installed wheel outside checkout, Windows/Python 3.14.5 | 236 passed, four skips: two Linux console cases and two source-only modules; source-only ShareGPT test excluded |
| Complete source / installed wheel, Linux/Python 3.12 | 276 passed / 238 passed with two source-only module skips |
| Earlier final CPU package, Linux/Python 3.10/3.12/3.14 | 276 source and 238 installed-wheel passes on each; two source-only modules skipped; retained under its original artifact identity |
| Deterministic HTTP fault matrix | 48 cells / 576 requests; four arrival traces, six faults, two admission modes, synthetic backends and virtual time; not a concurrent arrival benchmark |
| Manager lifecycle | Pinned discovery/inspect/disable/forget/uninstall checks; enable correctly refuses `import_only`, not automatic activation |

The public snapshot-delta and stale-snapshot oracles fail on unmodified
`04142b2`. Removing only the partial-live cleanup protection from an isolated
candidate makes four regressions fail on residual cache state. The corrected
candidate passes. Selected-rank/header conflicts have three pre-fix HTTP
negatives and corrected regression coverage; that is not real multi-DP inference.

Later documentation/packaging changes must preserve all runtime bytes and
recheck the affected delivery paths. The table identifies recorded artifacts,
not test totals for every future commit. Never insert a rebuilt wheel's hash
into an earlier run record.

## Latest native normal-path follow-up

The 2026-10-15 follow-up used source
`ab8dcc51c127abb74afd54cb533081be17331237` and wheel
`2b016751957bb8ffab748a92b72b03f72466e92d6705d96bcdff34924b1dcf3a`.
The [pinned HUST pair and model](reliability.md) were unchanged: Qwen2.5-14B-Instruct,
FP16/eager, two TP1 backends, Python 3.12.13, 8-CPU bandwidth and 64 GiB.
The 188-package freeze SHA-256 was
`ec45b6f95b04db4c677d7b4449bf389572666fba0a3a12303746bd0227c38015`.

Four fresh backend blocks completed 384 measured, 384 warmup and 32 seed
requests. Text/chat output, backend-reported cache usage, receipts and cleanup
checks passed; independent reconstruction rejected eight altered-evidence cases.
Router/direct block/cell median E2E differences ranged from -4.27% to +6.45%,
and TTFT from -0.14% to +9.19%, without the predefined repeated >=10% E2E warning.
Blocks, not individual requests, are the independent comparison units.

These are normal-path regression observations, **not** isolated reliability
overhead, a speedup, P99/throughput or capacity results. This follow-up did not
rerun every fault, certify the public host-version range or exercise real
multi-DP routing. OOM counters remained zero and owned-process cleanup passed.

Raw cell SHA-256:
`2988426762298761417c0f08a5bb67849a5dc5f7cd1b54594b51a75154c56680`.
The separately retained archive includes frozen helpers/spec, raw records,
independent review and package identity; see custody below.

## Earlier fault evidence and unresolved risk

The earlier PR #4 integration runs used wheel
`d4f50f6bdb081ac34176bcdd31ea807ac37d690c6db407c1ee6f58e956df5e58`.
They checked queue-time invalidation, timeout/exit before and after first SSE,
observation recovery, real client cancellation and interrupted responses.
The prospective four-pattern/two-mode restart repeat offered 192 requests:
190 completed and two bounded in-flight failures, with eight new-generation
cold-0 to cached-384 recovery probes. Failures remain in the offered denominator.
Restart-to-view-ready was 126.77–147.23 seconds, including model reload.

**The first restart attempt failed its oracle.** During peer model reload, one
survivor replay timed out in about three seconds; adjacent replies took about
0.1 seconds. Its 48 requests and two probes completed, but that does not make
the cell pass. The original raw-cell SHA-256 is
`3c851b4966e6a0befc3d06d12719d8921d9f11c82cb573c780828c8d3a8600a1`.

A bounded, unchanged-timeout steady diagnostic subsequently completed 48 requests,
two recovery probes and 814 survivor replay transactions with no timeout.
About 42% of CPU scheduling periods were throttled; sampled loop-delay maxima
were 194.83/147.31 ms in the two modes. This observes resource pressure, not the
cause of the earlier unsampled failure. Do not infer a CPU-count floor, relax
timeouts or claim the incident is fixed. A separate controlled CPU-headroom
comparison is needed before broadening restart-availability claims.

The first instrumented diagnostic accumulated file descriptors and was excluded.
The corrected monitor completed cleanly, but its environment-dependent trigger
was not fully isolated. Failed preparations, interrupted runs and cleanup records
remain archived; they do not count as plugin acceptance.

The earlier fault wheel predates the public partial-live fix and configured
backend-header fix. Those paths have targeted CPU regression coverage; the latest
wheel has the normal-path follow-up above, not an all-fault rerun. Native runs
without conflicting configured headers do not qualify real multi-DP behavior.
Pre-PR #4 experiments remain historical, not current acceptance.

## Submission scope and custody

| Included in the candidate | Retained outside the submission |
| --- | --- |
| Recovery, managed identity/attempt checks, request cleanup and receipts | Private duplicate Router/HTTP/session services and engine-transfer mechanisms |
| Runtime/protocol tests, console lifecycle, inert Manager rollback | Host builders, rejected environments and package-download scripts |
| One deterministic CPU HTTP matrix | Hardware-specific NPU launchers and successive performance diagnostics |
| Scope, delivery guide and this evidence index | Raw logs, intermediate reports, wheels, datasets and chronological plans |

PR #4's existing ShareGPT tools and host-wire probe remain baseline assets, not
new reliability contributions. No experiment result files are added. The unused
research-only evidence validator is not an installed API.

| Owner-held archive | SHA-256 |
| --- | --- |
| `20261013-pr4-integration-v1-reviewed.tar.gz` | `6564698e13e1e0142ec5dedb924aa6144dc4a6b0c5be834e2fbca8c46abcf6a2` |
| `20261013-pr4-replay-review-delivery.tar.gz` | `deded5b3f7e72620510c6de210e2a9bf42d9197865192bb976e60a2d279e0939` |
| `20261015-pr4-final-wheel-reviewed.tar.gz` | `689d907113d3acee283aca6f3daa11992b0be6e49315e4870a2c39ce76d6b270` |

Arrange reviewer access before relying on native claims in a PR. Without access,
limit acceptance claims to publicly reproducible code/CPU checks. Keep original
bytes, frozen helpers and failed runs together; never rewrite a verifier or relax
its hashes to turn an old result into a pass.

Engine-atomic pre-reuse fencing, KV-byte transfer guarantees, automatic Manager
activation, new models/TP2, real multi-DP and arbitrary newer hosts remain outside
this qualification. Remote CI and maintainer review are still pending.
