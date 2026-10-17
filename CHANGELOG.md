# Changelog

## 0.1.0.dev0

- Create an inert, discoverable external control-plane extension package.
- Extract the cleaned request-lifecycle block accounting core.
- Extract a framework-neutral Prefix cache index.
- Add isolated per-backend HTTP connection pools with bounded queueing.
- Keep the extension manifest `import_only` with operator-owned external lifecycle.

## Unreleased reliability slice

- Reject partial live-batch recovery after a group/layout validation failure;
  require reconstruction without copying the index on the normal live path.
- Integrate with PR #4 public defaults while isolating source-pinned profiles.
- Stage multi-group snapshot and partial-history replay before publication.
- Reject strict admission with unversioned best-effort recovery; report its basis.
- Bind selected DP rank to forwarding, Lifecycle accounting and request receipts.
- Preserve selected rank through backend-header merging, including case variants.
- Atomically publish recovered cache views and isolate retired replay attempts.
- Bound request cleanup and preserve response commitment across cancellations.
- Add managed, source-pinned host profiles without changing the inference engine.
- Add deterministic reliability and installed-console/rollback regression tests.
- Recheck selected cache identity after pool admission and record dispatch lineage.
- Provide a pinned two-TP1 HUST example and protected operator lease helper.
