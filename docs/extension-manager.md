# Extension Manager activation

The `0.1.0a1` wheel registers a static Manifest 0.3 Bundle and a
`vllm_hust_ext.providers` host provider named `prefix-router`. The Manager
records enabled intent and checks external services. The Router operator owns
its process lifecycle and the deployment's front-door traffic switch.

The Manager configuration must bind the two required service sets to the same
on-disk JSON used to start the Router. For example, with a Router started from
`/etc/vllm-hust/router.json`:

```json
{
  "router_url": "http://127.0.0.1:8000",
  "router_config_path": "/etc/vllm-hust/router.json",
  "backend_endpoints": {
    "node0": "http://10.0.0.10:8000"
  },
  "kv_event_endpoints": {
    "node0": {
      "event_endpoint": "tcp://10.0.0.10:5557",
      "replay_endpoint": "tcp://10.0.0.10:5558"
    }
  }
}
```

Include every backend from the Router JSON. Endpoint maps must match exactly;
an empty or partially configured map is rejected. Backend credentials remain
referenced by environment name in the Router JSON and must be available to the
Manager process when it checks `/health` on each backend.

```bash
vllm-hust-prefix-router serve --config /etc/vllm-hust/router.json
vllm-hust-ext extension configure org.vllm-hust.prefix-router \
  --file /etc/vllm-hust/router.ecpa.json
vllm-hust-ext extension check org.vllm-hust.prefix-router
vllm-hust-ext extension enable org.vllm-hust.prefix-router
vllm-hust-ext extension plan org.vllm-hust.prefix-router
```

`check` requires the running Router's package version and config SHA-256 to
match the installed wheel and JSON file. It then requires `/readyz`, one
reconciled KV event source per configured node, and HTTP 200 from `/health`
on every backend. The current unversioned replay opt-in reports degraded
best-effort recovery even when the services are reachable. A failed check
blocks a new Manager-supervised `run` because the manifest declares both
service sets as required. Enabling does not imply `runtime_effective`.

To roll back, switch deployment traffic back to the previous front door,
stop the Router using the operator's process manager, then run:

```bash
vllm-hust-ext extension disable org.vllm-hust.prefix-router
vllm-hust-ext extension forget org.vllm-hust.prefix-router
```

`disable` and `forget` only change Manager intent. They do not stop the Router
or switch network traffic. Uninstall the wheel after these steps if desired.
