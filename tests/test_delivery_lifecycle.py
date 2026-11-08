"""Real console/HTTP/OS-signal tests, with explicitly synthetic host factories."""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sysconfig
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import vllm_hust_prefix_router


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def console():
    name = "vllm-hust-prefix-router" + (".exe" if os.name == "nt" else "")
    path = Path(sysconfig.get_path("scripts")) / name
    assert path.is_file(), "install the distribution before testing its console"
    return str(path)


async def wait_http(client, process, url):
    for _ in range(200):
        assert process.poll() is None, "console exited before becoming live"
        try:
            async with client.get(url) as response:
                assert response.status == 200
                return
        except aiohttp.ClientConnectorError:
            await asyncio.sleep(0.025)
    raise AssertionError("console did not become live within five seconds")


def stop(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
        raise AssertionError("console ignored its bounded shutdown") from None


@pytest.mark.skipif(os.name != "posix", reason="Linux operator signal contract")
@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_console_start_stop_restart_and_native_bypass(tmp_path, strict):
    hits = []

    async def backend(request):
        hits.append(await request.json())
        return web.json_response({"id": "cpu-fixture", "choices": []})

    app = web.Application()
    app.router.add_post("/v1/completions", backend)
    upstream = TestServer(app)
    await upstream.start_server()
    fixture = Path(__file__).with_name("fixtures") / "delivery_sitecustomize.py"
    shutil.copyfile(fixture, tmp_path / "sitecustomize.py")
    package_root = Path(vllm_hust_prefix_router.__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(package_root)))
    port = free_port()
    config = {
        "listen": {"host": "127.0.0.1", "port": port},
        "hash_block_size": 16,
        "fingerprint": {"tokenizer": "CPU-fixture-not-a-real-model"},
        "require_trusted_cache_view": strict,
        "backends": [
            {
                "id": "cpu-backend",
                "url": str(upstream.make_url("/")),
                "event_endpoint": f"tcp://127.0.0.1:{free_port()}",
                "replay_endpoint": f"tcp://127.0.0.1:{free_port()}",
            }
        ],
    }
    config_path = tmp_path / "router.json"
    config_path.write_text(json.dumps(config))
    url = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as client:
            # Rebind the same port after SIGTERM; the backend is not owned/stopped
            # by the Router. No privileged NPU or engine process is involved.
            for cycle in range(2):
                with (tmp_path / f"console-{cycle}.log").open("wb") as log:
                    process = subprocess.Popen(
                        [console(), "serve", "--config", str(config_path)],
                        cwd=tmp_path,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        await wait_http(client, process, url + "/healthz")
                        async with client.get(url + "/readyz") as response:
                            assert response.status == 503
                        async with client.post(
                            url + "/v1/completions", json={"prompt": "fixture"}
                        ) as response:
                            assert response.status == (503 if strict else 200)
                            await response.read()
                        async with client.get(url + "/metrics") as response:
                            metrics = await response.json()
                        pool = metrics["backends"]["cpu-backend"]
                        assert pool["active_requests"] == pool["queued_requests"] == 0
                    finally:
                        stop(process)
                    assert process.returncode == 0
                with pytest.raises(aiohttp.ClientConnectorError):
                    await client.get(url + "/healthz")
                async with client.post(
                    upstream.make_url("/v1/completions"), json={"prompt": "bypass"}
                ) as response:
                    assert response.status == 200
                    assert (await response.json())["id"] == "cpu-fixture"
        assert len(hits) == (2 if strict else 4)
    finally:
        await upstream.close()


def test_console_invalid_configuration_fails_before_host_import(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"runtime_profile": "unknown-build"}))
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    env["PYTHONPATH"] = str(Path(vllm_hust_prefix_router.__file__).resolve().parents[1])
    result = subprocess.run(
        [console(), "serve", "--config", str(path)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "unsupported runtime_profile" in result.stderr
