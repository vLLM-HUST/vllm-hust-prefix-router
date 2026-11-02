import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

import pytest

from vllm_hust_prefix_router._version import __version__
from vllm_hust_prefix_router.provider import PrefixRouterProvider
from vllm_hust_prefix_router.service.provider_checks import (
    EXTENSION_ID,
    probe_deployment,
    validate_binding,
)


@pytest.fixture
def servers():
    state = {"router_ready": True, "backend_ready": True, "digest": ""}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            router = self.server.server_address[1] == state["router_port"]
            if router and self.path == "/healthz":
                body = {
                    "status": "ok",
                    "extension_id": EXTENSION_ID,
                    "version": __version__,
                    "config_sha256": state["digest"],
                }
                code = 200
            elif router and self.path == "/readyz":
                code = 200 if state["router_ready"] else 503
                body = {"status": "ready" if code == 200 else "degraded"}
            elif router and self.path == "/metrics":
                code = 200
                body = {
                    "backends": {"node0": {}},
                    "event_sources": [
                        {
                            "node_id": "node0",
                            "ready": state["router_ready"],
                            "last_error": None,
                            "next_seq": 7,
                        }
                    ],
                }
            elif not router and self.path == "/health":
                code = 200 if state["backend_ready"] else 503
                body = {"status": "ok" if code == 200 else "degraded"}
            else:
                code, body = 404, {}
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    router = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    backend = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state["router_port"] = router.server_address[1]
    threads = [
        threading.Thread(target=s.serve_forever, daemon=True) for s in (router, backend)
    ]
    for thread in threads:
        thread.start()
    try:
        yield (
            state,
            f"http://127.0.0.1:{router.server_address[1]}",
            f"http://127.0.0.1:{backend.server_address[1]}",
        )
    finally:
        for server in (router, backend):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


def _binding(tmp_path, servers):
    state, router_url, backend_url = servers
    event = "tcp://127.0.0.1:5557"
    replay = "tcp://127.0.0.1:5558"
    config_path = tmp_path / "router.json"
    data = {
        "backends": [
            {
                "id": "node0",
                "url": backend_url,
                "event_endpoint": event,
                "replay_endpoint": replay,
            }
        ]
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")
    state["digest"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config = {
        "router_url": router_url,
        "router_config_path": str(config_path),
        "backend_endpoints": {"node0": backend_url},
        "kv_event_endpoints": {
            "node0": {"event_endpoint": event, "replay_endpoint": replay}
        },
    }
    return validate_binding(config), config


def test_live_binding_checks_router_kv_recovery_and_backend(tmp_path, servers):
    binding, _ = _binding(tmp_path, servers)
    result = probe_deployment(binding)
    assert result.reachable and result.healthy and not result.degraded
    assert any("KV event" in item for item in result.evidence)


def test_changed_config_and_unhealthy_backend_fail_closed(tmp_path, servers):
    binding, config = _binding(tmp_path, servers)
    config["backend_endpoints"] = {"node0": "http://wrong.example"}
    with pytest.raises(ValueError, match="exactly match"):
        validate_binding(config)

    servers[0]["backend_ready"] = False
    result = probe_deployment(binding)
    assert result.reachable and not result.healthy and result.degraded
    assert any("503" in item for item in result.evidence)


def test_running_config_digest_must_match_disk(tmp_path, servers):
    binding, _ = _binding(tmp_path, servers)
    servers[0]["digest"] = "wrong"
    result = probe_deployment(binding)
    assert result.reachable and not result.healthy
    assert any("does not match" in item for item in result.evidence)


def test_extension_manager_provider_accepts_only_live_bound_services(tmp_path, servers):
    manager_manifest = pytest.importorskip("vllm_hust_ext.manifest")
    manifest_path = resources.files("vllm_hust_prefix_router.manifests").joinpath(
        "vllm-hust-extension-v0.3.json"
    )
    manifest = manager_manifest.parse_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    _, config = _binding(tmp_path, servers)
    provider = PrefixRouterProvider()
    assert provider.supports(manifest)
    check = provider.check(manifest, config)
    assert check.compatible and check.configured and check.reachable and check.healthy
    plan = provider.plan(manifest, config, enabled=True)
    assert all(not action.mutating for action in plan.actions)
    assert "check_external_router" in {action.operation for action in plan.actions}
    assert json.loads(provider.render(plan)[0].content)["provider"] == "prefix-router"

    servers[0]["router_ready"] = False
    degraded = provider.check(manifest, config)
    assert degraded.configured and not degraded.healthy and degraded.degraded
