# Hash identity, lifecycle ownership and count-cost units

Text fingerprinting delegates full-block hashing to the installed host's
`get_request_block_hasher`. A nonempty `cache_salt` of up to 1024 characters
is supported for text completions and chat requests. `allow_cache_salt=false`
restores explicit fallback for these requests. Token IDs, template, hash-block
size, hash algorithm, initial hash seed and external event representation must
match every backend.

This adapter describes the ordinary text-request identity contract. Host-side
KV materialization that injects a salt or resets the chain at a partial-reuse
boundary requires the effective per-request identity to be communicated to the
Router. Such hidden server policies remain outside this contract. Explicit
`vllm_xargs` and `kv_transfer_params` requests fall back until their identities
are adapted. The host's LoRA, multimodal and prompt-embedding paths still need
their own identity adapters. Supported releases must pass the installed-host
hash qualification before performance measurement.

## Request ownership and progress

The HTTP service holds a weak owner reference to its lifecycle observer.
Reservations with a live owner survive progress-lease expiry, including pending
pool waits, long prefill and non-streaming generation. The service's existing
`finally` path releases on completion, error and cancellation. If an observer
is abandoned, the weak reference does not keep it alive and normal expiry can
reclaim its reservation. Standalone tracker callers retain the original
progress-lease behavior unless they explicitly bind an owner.

`metrics()` uses a read-only load snapshot. Explicit cleanup and new selections
perform expiry checks. Streaming state is updated when upstream data arrives,
before a potentially blocked downstream write. Transport occupancy remains in
the connection pool until the HTTP operation closes. Tool-function name and
argument deltas count as observed generation; this does not expand fingerprint
support for arbitrary tool-bearing input requests.

## Portable count costs

The default `count_weight_unit="blocks"` preserves the existing configuration.
With `count_weight_unit="tokens"`, only `active_request_weight`,
`active_decode_sequence_weight` and `router_queue_weight` are divided by the
actual hash-block size before scoring. Prefill and unique-block terms retain
their existing definitions. For example, a legacy request weight 4 at B=1152
migrates to request weight 4608 in token mode. No automatic migration is applied.
This fixes the relative unit scale; model-specific latency costs still require
measurement. Tail-block granularity and real prefix-hit differences remain.

## Optional decision evidence

`decision_history_size` defaults to zero and accepts 0 through 4096. When enabled,
`routing.decisions` contains bounded recent request IDs, candidate identities,
component costs, observed counts, progress ages, selection temperature and the
random sample actually consumed. Input text, token IDs and block hashes are
excluded. A null random sample means no draw was needed in that decision; an
unobserved hypothetical tie cannot be assigned a fictitious draw. The endpoint
is a snapshot and a collector must preserve records before retention overwrites
them. `routing.effective_count_weights` exposes converted coefficients when
token units or decision history is enabled.

The research campaign lives in the workspace's `optimization-20261119` folder.
Its baseline is PR8 plus the previously qualified explicit-salt adapter; this
keeps all comparison requests on the same routing path. Forced background and
target routing are explicit test-wrapper controls used for matched A/B cases.
