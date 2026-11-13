# Optional Decode and Router queue costs

These terms extend the Lifecycle policy without changing the Prefill term,
adding context-length costs, or requiring scheduler snapshots. They are
workload heuristics, not calibrated TTFT predictions. Prefix policy does not
use them.

## Score

For each candidate worker, the score is:

```text
prefill_load_weight * (active_prefill_tokens + incoming_uncached_tokens) / B
+ decode_block_weight * potential_unique_active_blocks
+ active_request_weight * (active_prompts + incoming_prompts)
+ active_decode_sequence_weight * active_decode_sequences
+ router_queue_weight * router_queue_requests
```

`B` is the tracker's hash block size. The first three terms retain their
previous definitions. New Decode sequences are not charged before their
output is observed. The queue term counts existing waiters, not the incoming
request, active HTTP connections, or backend scheduler waiters.

## Observed active Decode sequences

The response observer starts counting a choice on its first nonempty text,
content, reasoning content, or exact output token IDs in a successful SSE
response. Role-only events do not count. Repeated chunks do not increment
the count. Each choice stops counting on its own `finish_reason`, even while
sibling choices keep running. Batch prompts and `n > 1` are tracked separately.
Invalid choice indices are ignored. Final release, cancellation, and expiry
discard the remaining count together with the request reservation.

This is an observed sequence count, not a scheduler running count. Non-streaming
responses do not expose intermediate Decode activity and do not contribute to
this count. Missing output/finish events, compressed or unrecognized response
formats, network buffering, and slow clients can delay or prevent observation.
The existing expiry lease may release a still-running request with no observed
progress. Traffic that bypasses this Router or fingerprint fallback reservations
is not covered. Output-block tracking need not be enabled, and text chunks
are not treated as token counts.

`GET /metrics` exposes `routing.lifecycle.<node>.active_decode_sequences`.

## Router admission queue

Immediately before Lifecycle selection, the HTTP service reads each backend
pool's `queued_requests` and passes that snapshot to the planner. All candidate
DP ranks of a backend ID use the same pool count, since the HTTP pool is scoped
to that ID. The count is refreshed for every request; it is not accumulated in
the lifecycle tracker. Standalone planner callers can supply
`router_queue_requests={node_id: count}`; missing nodes default to zero.

Existing pool metrics remain at `backends.<node>.queued_requests`. The queue
term partly overlaps with lifecycle reservations: it is an additional transport
congestion penalty, not independent compute work. If the pool never queues,
this term is zero and cannot change routing. It does not proactively penalize
a full pool before any waiter exists, predict wait duration, or retry a rejected
request on another backend.

## Opt-in comparisons

Add the following keys inside the existing `lifecycle` JSON object. All other
settings, including the original weights, should stay identical for a comparison.

```json
{
  "active_decode_sequence_weight": 1.0,
  "router_queue_weight": 1.0
}
```

These positive values are illustrative, not tuned or performance-qualified.
Use `(0, 0)` for the original score, `(1, 0)` for Decode only, `(0, 1)` for
queue only, and `(1, 1)` for both, in the key order above. Both omitted keys
default to zero. Weights must be finite non-negative numbers; booleans are
rejected. No regression or automatic weight adaptation is implemented.

Local tests cover count transitions, shared-block routing decisions, live
streaming observation, real admission waiters, and zero-weight compatibility.
They establish behavior only. Model/NPU performance comparisons are separate;
these changes make no performance claim.
