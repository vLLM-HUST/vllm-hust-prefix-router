# Delivery checks and operator lifecycle

This guide does not install/upgrade Core, Ascend, Torch, drivers or model weights.
CPU checks validate the package, not native engine support on every Python version.
Use a fresh CPU virtualenv, not the serving environment. Clear inherited
`PYTHONPATH`/`PYTHONHOME` so server CANN paths cannot contaminate `pip check`.

## Source, wheel and deterministic regression

```bash
unset PYTHONPATH PYTHONHOME
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
git diff --check
python -m build
```

The CI matrix runs Python 3.10/3.12/3.14. It unpacks the sdist outside the checkout,
checks the required operator/matrix assets, installs it into a fresh venv and
runs its tests. A separate fresh venv tests the wheel. CI copies tests (including
`fixtures/`) outside the checkout and excludes the baseline source-only ShareGPT
test. The deterministic matrix test skips outside source/sdist, as does the
source-only lease helper test; a skip is not coverage. No NPU is needed by CI.

For the retained four-pattern/six-fault/two-mode CPU matrix, use a **new** output
directory and preserve both failures and successes:

```bash
python experiments/reliability/control_fault_matrix.py --output-dir /absolute/new-matrix
python experiments/reliability/control_fault_matrix.py --validate /absolute/new-matrix
```

This runs the actual Router/HTTP/index/replay logic with synthetic backends and
a virtual clock. Requests are sequential; it is not a concurrent burst benchmark,
actual engine hit/recompute measurement or P99 experiment. Unit tests also cover
real local ZMQ framing, timeout, partial reply and sequence rejection.

`test_delivery_lifecycle.py` launches a real console child process, HTTP listener,
pool and ZMQ supervisor, using two explicit CPU host factories. It checks degraded
readiness, safe fallback/strict refusal, SIGTERM, same-port restart and unaffected
direct-backend access. These factories are test assets, not installed wheel code;
the test is not native host activation/fingerprint qualification.

## Manager remains inert

Install Manager at `7bcfe5566146e148de8f6593dc1b8b2e30dbced9` into a **disposable
wheel-test venv** and run its Python against:

```bash
/absolute/disposable-venv/bin/python scripts/check_delivery_manager.py \
  --output /absolute/new-manager-review \
  --uninstall-from-disposable-venv
```

This records discover/inspect/validate/check/plan/render, requires enable to fail
specifically for `import_only`, checks repeated disable, forget, uninstallation
and a plain Python subprocess through Manager after each transition. This checks
environment/import isolation, not real-engine inference after rollback. It removes
only the Router package from that disposable venv and refuses environments
containing vLLM/Torch/Ascend. Forget removes stored state, not the package itself.
Do not use a serving/shared venv. A missing `result.json` is not success.

## Real deployment and rollback

Start from [the two-TP1 managed HUST example](../examples/router.managed-hust.json).
Replace the lease directory with a new operator-owned directory (mode 0700).
Adjust all HTTP/event/replay ports to match the actual owned engines. This sample
uses loopback, explicit rank 0, 128-token sha256 blocks and the pinned HUST profile;
the existing `router.lifecycle.json` follows PR #4's public defaults. That example
refuses a host without snapshot/epoch replay unless the operator explicitly
selects best-effort operation. It is not the frozen managed deployment example.

Before running the lease helper or Router, verify the wheel hash against its
review record and verify the host already has the required dependencies.
Install it into a **new empty** target with the frozen host Python:

```bash
/absolute/host-env/bin/python -m pip install --no-deps \
  --target /absolute/new-router-target \
  /absolute/vllm_hust_prefix_router-0.1.0.dev0-py3-none-any.whl
```

Do not overwrite a live target, upgrade serving
dependencies or infer host compatibility from import success. The following
commands assume this target exists and the [pinned profile](reliability.md)
checks pass; replace placeholder paths with the audited deployment paths.

Launch each engine with the frozen host Python, `PYTHONHASHSEED=0`,
`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`, `VLLM_PLUGINS=ascend`,
`VLLM_USE_RUST_FRONTEND=0` and `VLLM_USE_FASTOKENS=0`. For example, node0 uses:

```bash
export PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 VLLM_PLUGINS=ascend
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_FASTOKENS=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HUST_ACL_DIR=/usr/local/Ascend/cann-9.1.0/python/site-packages/acl
export PYTHONPATH="$HUST_ACL_DIR"
ASCEND_RT_VISIBLE_DEVICES=0 /absolute/host-env/bin/python -m vllm.entrypoints.cli.main serve \
  /data/shared_models/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8100 \
  --tensor-parallel-size 1 --dtype float16 --max-model-len 1024 --max-num-seqs 4 \
  --max-num-batched-tokens 1024 --gpu-memory-utilization 0.65 --enforce-eager \
  --enable-prefix-caching --seed 0 --block-size 128 --prefix-caching-hash-algo sha256 \
  --enable-prompt-tokens-details \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://127.0.0.1:5557","replay_endpoint":"tcp://127.0.0.1:5558","buffer_steps":10000}'
```

Use a second allocated logical NPU and node1's ports for the second TP1 instance.
The frozen host uses the isolated vendor `acl.so` directory above, with SHA-256
`34303ba302bde3cd44dd5bd7bc01fb216cbc5123e7db797a9834c9fa58e3f185`.
Verify it with `sha256sum "$HUST_ACL_DIR/acl.so"` before launch. The existing
host environment must already supply its CANN dynamic-library setup. Do not add
the whole SDK Python directory (which can shadow host NumPy or other packages).
Different SDK bytes or setup require an environment check before inference.
Confirm container logical IDs using `npu-smi info -m`; do not substitute physical
IDs blindly. Keep serving isolated until both `/v1/models` endpoints are live.
Record the owned API PID and its actual EngineCore child PID from the process tree.
Each must be a distinct live process. No user inference may precede lease capture.

The source/sdist operator helper verifies the pinned interfaces, captures process
identities, chooses a new generation UUID and publishes a private lease atomically:

```bash
PYTHONPATH=/absolute/new-router-target:$HUST_ACL_DIR /absolute/host-env/bin/python \
  /absolute/router-checkout/scripts/write_managed_lease.py \
  --config /absolute/audited-router.json --node node0 \
  --api-pid ACTUAL_API_PID --engine-pid ACTUAL_ENGINE_PID --confirm-new-unused-engine
```

Repeat for node1. The confirmation is an operator assertion; the helper cannot
prove engine emptiness or that supplied PIDs belong to the intended service.
It refuses an existing lease path or a group/world-writable directory. On restart,
allocate new event/replay ports and a fresh lease path; update the config and restart
the owned Router before reopening ingress. A supervisor can instead atomically
replace its protected lease after performing the same fresh-engine checks.
Do not turn a used engine into an apparently fresh one by regenerating a lease.

Export the hash/plugin environment above for the Router and lease helper too.
Only after `/readyz` returns 200 should ingress admit requests. Preserve the lease,
config, package hashes and logs; protect `/metrics`. The helper does not launch,
stop, restart or upgrade engines.

Use verified identities and a protected configuration/lease. The already
installed target's console uses the frozen host Python; launch it explicitly:

```bash
PYTHONPATH=/absolute/new-router-target:$HUST_ACL_DIR \
  /absolute/new-router-target/bin/vllm-hust-prefix-router serve \
  --config /absolute/audited-router.json
```

The operator/supervisor owns this service. `/healthz` means the process is alive;
`/readyz` reflects trusted event sources. Protect metrics and bind privately.
Set upstream keepalive below the actual shortest backend/proxy idle close timeout;
the frozen HUST qualifications used 2 seconds. Do not add POST retries to hide errors.

Record the container's actual CPU quota, memory limit and pressure during restart,
in addition to the NPU allocation. Host model reload and Router/event processing
may contend for the same CPU budget. `asyncio.wait_for` uses cooperative task
cancellation; configured timeouts are not a hard wall-clock guarantee during
event-loop stalls. See the [Python timeout documentation](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.wait_for)
and [Linux pressure documentation](https://docs.kernel.org/accounting/psi.html).
The unexplained survivor replay incident remains a qualification risk. No
minimum CPU count or timeout relaxation is inferred from a passing diagnostic.

Rollback order:

1. Stop admitting new Router traffic at ingress/client; drain or cancel in-flight
   requests under their deadlines. Do not retry an already committed SSE response.
2. SIGTERM the **owned current Router process** (or use its supervisor/Ctrl-C).
   Verify exit and listener closure. Never kill a reused PID from an old run.
3. Restore the saved direct-backend address/config at ingress/client and verify
   direct health/request access. Backend engines have independent ownership.
4. Disable then forget Manager state if present; uninstall only the dedicated
   Router environment/retire the inactive target, retaining the previous artifact
   and config. Never delete host or model directories as part of plugin rollback.

Manager disable alone neither redirects traffic nor stops a user-owned service.
Automatic activation remains intentionally blocked under the public contract;
changing that boundary requires a separately reviewed provider/lifecycle design.
