# vLLM-HUST Prefix Router

Owner-maintained extraction of the prefix-aware routing work developed in
vLLM-HUST.

**Status: active pre-release migration. The current development wheel is
installable and discoverable, but its manifest remains `import_only` until the
external service passes end-to-end acceptance.**

The Manifest 0.3 descriptor names the required OpenAI backend and KV-event
service sets and claims the deployment routing front door exclusively. These
are composition metadata only: ECPA does not start, stop, configure, or health-
check this user-owned process while the carrier is `import_only`.

The target is a control-plane router extension plus a narrow vLLM cache-event
adapter. The global router will not be presented as an in-process scheduler
plugin.

See [PROVENANCE.md](PROVENANCE.md) for the original PR chain and
[MAINTAINERS.md](MAINTAINERS.md) for ownership.

## Architecture

The package runs one external Router process. Every inference node is a remote
backend; there is no privileged in-process or local-node path. The Router owns:

- longest-prefix cache indexing and Prefix routing;
- request-lifecycle deduplicated block accounting;
- KV event gap detection and replay recovery;
- OpenAI-compatible request forwarding and response observation.

Lifecycle accounting never consumes Scheduler snapshots. Removed V1/V2 cost
models, dynamic Prefill-rate estimation, calibration, and fixed-duration
reservations are intentionally absent.

## HTTP connection isolation

Streaming generation occupies an HTTP/1.1 connection until the response ends.
Using aiohttp's default connector would cap all remote nodes at 100 shared
connections while an in-process node bypassed that limit. The external Router
instead creates one connector per backend, defaults each connector to 512
connections, and applies an observable bounded admission queue. All nodes
therefore use the same transport path and one busy node cannot exhaust another
node's connection capacity.

The pool limit must cover the number of concurrent streams, not merely request
rate. A useful starting point for each backend is its peak request rate times
the upper-tail stream duration, plus headroom. For example, 10 requests/s with
a 30-second P99 stream needs more than 300 concurrent connections. The default
of 512 is deliberately above aiohttp's global default of 100, but production
deployments must also raise the process file-descriptor limit when necessary.
`/metrics` exposes active, queued, admitted, rejected, and queue-wait values per
backend. A full or slow admission queue returns HTTP 503 instead of hiding
tens of seconds of delay inside TTFT.

## Run the service

Copy [`examples/router.lifecycle.json`](examples/router.lifecycle.json), then
replace every worker HTTP and KV-event endpoint with the deployment values.
The tokenizer, block size, hash algorithm, host version, and
`PYTHONHASHSEED` must match every worker.

```bash
export PYTHONHASHSEED=0
vllm-hust-prefix-router serve --config examples/router.lifecycle.json
```

All configured backends are HTTP(S) targets. Do not put the Router itself in
the backend list and do not configure an in-process shortcut for a colocated
worker. Backend credentials, if needed, are referenced by environment-variable
name through `headers_from_env`; the JSON file does not contain the secret.

## Development install

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
uv run pytest -q
uv run ruff check .
uv build --no-sources --out-dir dist
```

No package version has been published. Do not treat installation or Extension
Manager discovery as service activation.

Current ECPA validation is intentionally limited to `list`, `inspect`,
`validate`, `check`, `plan`, and `render`. `enable` must fail closed; the
operator starts and stops the service explicitly with the command above. A
future active provider must bind the manifest's `backend_endpoints` and
`kv_event_endpoints` configuration keys to real health evidence before launch
intent can be accepted.

