#!/usr/bin/env python3
"""Run one cold-started Prefix or Lifecycle ShareGPT curve point."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return value


def get_text(url: str, timeout: float = 3.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            value = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            value.bind(("127.0.0.1", 0))
            sockets.append(value)
        return [int(value.getsockname()[1]) for value in sockets]
    finally:
        for value in sockets:
            value.close()


def tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def busy_devices(npu_smi: str) -> set[str]:
    """Return physical device IDs that have a reported NPU process."""
    return {
        device
        for device, _chip, _pid in re.findall(
            r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|",
            npu_smi,
            re.MULTILINE,
        )
    }


def handle_termination(_signum: int, _frame: Any) -> None:
    """Turn SIGTERM into normal stack unwinding so child groups are reaped."""
    raise KeyboardInterrupt


@dataclass
class ChildProcess:
    command: list[str]
    env: dict[str, str]
    log_path: Path
    process: subprocess.Popen[str] | None = None
    _log: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            env=self.env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def assert_running(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(
                f"process exited with {self.process.returncode}: {self.log_path}\n"
                f"{tail(self.log_path)}"
            )

    def stop(self, timeout: float = 30.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self._log is not None:
            self._log.close()
            self._log = None


@dataclass
class MetricsSampler:
    url: str
    interval_s: float
    samples: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        start = time.perf_counter()

        def collect() -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(
                        {
                            "offset_s": time.perf_counter() - start,
                            "metrics": get_json(self.url),
                        }
                    )
                except Exception as exc:  # Evidence must retain sampling failures.
                    self.errors.append(
                        {
                            "offset_s": time.perf_counter() - start,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=collect, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 2))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def measurement_window(
    result: dict[str, Any], *, start_s: float, duration_s: float
) -> dict[str, Any]:
    end_s = start_s + duration_s
    timings = result.get("routing_client_timings", [])
    vectors = [
        timings,
        result.get("latencies", []),
        result.get("ttfts", []),
        result.get("output_lens", []),
        result.get("errors", []),
    ]
    if len({len(value) for value in vectors}) != 1 or not timings:
        raise RuntimeError("detailed client timing vectors are missing or incomplete")
    records = []
    for timing, latency, ttft, output_len, error in zip(*vectors, strict=True):
        arrival = float(timing["arrival_offset_s"])
        dispatch = float(timing["dispatch_offset_s"])
        records.append(
            {
                "arrival": arrival,
                "dispatch": dispatch,
                "completion": dispatch + float(latency),
                "latency": float(latency),
                "ttft": float(ttft),
                "output_len": int(output_len),
                "success": not bool(error),
                "wait_ms": float(timing["concurrency_wait_ms"]),
            }
        )
    cohort = [item for item in records if start_s <= item["arrival"] < end_s]
    terminals = [item for item in records if start_s <= item["completion"] < end_s]
    successful = [item for item in cohort if item["success"]]
    if not cohort:
        raise RuntimeError("measurement window contains no request arrivals")
    ttft = [item["ttft"] * 1000 for item in successful]
    e2el = [item["latency"] * 1000 for item in successful]
    tpot = [
        (item["latency"] - item["ttft"]) / max(1, item["output_len"] - 1) * 1000
        for item in successful
    ]
    waits = [item["wait_ms"] for item in cohort]
    backlog_start = sum(item["arrival"] < start_s for item in records) - sum(
        item["completion"] < start_s for item in records
    )
    backlog_end = sum(item["arrival"] < end_s for item in records) - sum(
        item["completion"] < end_s for item in records
    )
    return {
        "start_s": start_s,
        "duration_s": duration_s,
        "arrival_count": len(cohort),
        "arrival_rate": len(cohort) / duration_s,
        "terminal_count": len(terminals),
        "terminal_rate": len(terminals) / duration_s,
        "successful_completion_rate": sum(item["success"] for item in terminals)
        / duration_s,
        "cohort_failures": len(cohort) - len(successful),
        "backlog_at_start": backlog_start,
        "backlog_at_end": backlog_end,
        "backlog_growth": backlog_end - backlog_start,
        "mean_ttft_ms": statistics.fmean(ttft) if ttft else None,
        "p50_ttft_ms": percentile(ttft, 50),
        "p95_ttft_ms": percentile(ttft, 95),
        "p99_ttft_ms": percentile(ttft, 99),
        "mean_tpot_ms": statistics.fmean(tpot) if tpot else None,
        "p95_tpot_ms": percentile(tpot, 95),
        "mean_e2el_ms": statistics.fmean(e2el) if e2el else None,
        "p95_e2el_ms": percentile(e2el, 95),
        "p99_e2el_ms": percentile(e2el, 99),
        "p95_concurrency_wait_ms": percentile(waits, 95),
        "p99_concurrency_wait_ms": percentile(waits, 99),
    }


def select_phase(manifest: dict[str, Any], rep: int, rate_index: int) -> dict[str, Any]:
    matches = [
        phase
        for repetition in manifest["repetitions"]
        for phase in repetition["phases"]
        if phase["curve_repetition"] == rep and phase["rate_index"] == rate_index
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one phase for rep={rep}, rate_index={rate_index}")
    return matches[0]


def wait_for_health(process: ChildProcess, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        process.assert_running()
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


def worker_command(
    args: argparse.Namespace, http_port: int, event_port: int, replay_port: int
) -> list[str]:
    kv = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://*:{event_port}",
        "replay_endpoint": f"tcp://*:{replay_port}",
    }
    return [
        str(args.vllm_cli),
        "serve",
        str(args.model),
        "--served-model-name",
        args.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.9",
        "--block-size",
        str(args.scheduler_block_size),
        "--enable-prefix-caching",
        "--kv-events-config",
        json.dumps(kv, separators=(",", ":")),
    ]


def router_config(
    args: argparse.Namespace,
    router_port: int,
    http_ports: list[int],
    event_ports: list[int],
    replay_ports: list[int],
) -> dict[str, Any]:
    return {
        "listen": {"host": "127.0.0.1", "port": router_port},
        "policy": args.policy,
        "hash_block_size": args.hash_block_size,
        "default_backend": "node0",
        "fingerprint": {
            "tokenizer": str(args.model),
            "hash_algorithm": "xxhash",
            "host_version_range": args.host_version_range,
            "trust_remote_code": False,
        },
        "lifecycle": {
            "prefill_load_weight": 1.0,
            "active_request_weight": 0.0,
            "selection_temperature": 0.0,
            "track_output_blocks": False,
            "request_expiry_s": 300.0,
        },
        "backends": [
            {
                "id": f"node{index}",
                "url": f"http://127.0.0.1:{http_ports[index]}",
                "data_parallel_rank": 0,
                "event_endpoint": f"tcp://127.0.0.1:{event_ports[index]}",
                "replay_endpoint": f"tcp://127.0.0.1:{replay_ports[index]}",
                "pool": {
                    "max_connections": args.max_connections,
                    "max_pending_requests": args.max_pending_requests,
                    "queue_timeout_s": args.queue_timeout_s,
                },
            }
            for index in range(4)
        ],
    }


def benchmark_command(
    args: argparse.Namespace,
    phase: dict[str, Any],
    router_port: int,
    workload: Path,
    result_dir: Path,
) -> list[str]:
    return [
        str(args.vllm_cli),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{router_port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        str(args.model),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(workload),
        "--num-prompts",
        str(phase["total_requests"]),
        "--custom-output-len",
        "64",
        "--request-rate",
        str(phase["request_rate"]),
        "--burstiness",
        "1.0",
        "--max-concurrency",
        "256",
        "--seed",
        str(phase["arrival_seed"]),
        "--disable-shuffle",
        "--no-oversample",
        "--skip-chat-template",
        "--ignore-eos",
        "--temperature",
        "0",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--save-detailed",
        "--request-id-prefix",
        f"plugin-{args.policy}-r{args.rep}-q{args.rate_index}-",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "benchmark.json",
    ]


def aggregate_router_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    max_lifecycle: dict[str, dict[str, float]] = {}
    max_pool: dict[str, dict[str, float]] = {}
    for sample in samples:
        metrics = sample["metrics"]
        lifecycle = metrics.get("routing", {}).get("lifecycle", {})
        for node, values in lifecycle.items():
            target = max_lifecycle.setdefault(node, {})
            for key, value in values.items():
                if isinstance(value, int | float):
                    target[key] = max(target.get(key, 0.0), float(value))
        for node, values in metrics.get("backends", {}).items():
            target = max_pool.setdefault(node, {})
            for key in ("active_requests", "queued_requests", "queue_wait_seconds"):
                value = values.get(key)
                if isinstance(value, int | float):
                    target[key] = max(target.get(key, 0.0), float(value))
    return {"max_lifecycle": max_lifecycle, "max_backend_pool": max_pool}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("prefix", "lifecycle"), required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--rate-index", type=int, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-cli", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--plugin-repo", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--ascend-source", type=Path, required=True)
    parser.add_argument("--served-model-name", default="prefix-router-evidence")
    parser.add_argument("--host-version-range", default=">=0.23.1,<0.24")
    parser.add_argument("--scheduler-block-size", type=int, default=128)
    parser.add_argument("--hash-block-size", type=int, default=128)
    parser.add_argument("--max-connections", type=int, default=512)
    parser.add_argument("--max-pending-requests", type=int, default=512)
    parser.add_argument("--queue-timeout-s", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--metrics-interval", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, handle_termination)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) != 4 or len(set(devices)) != 4:
        raise SystemExit("--devices must contain four distinct device IDs")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    phase = select_phase(manifest, args.rep, args.rate_index)
    workload = args.manifest.parent / phase["path"]
    if sha256(workload) != phase["sha256"]:
        raise RuntimeError("workload SHA-256 does not match the frozen manifest")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.result_root / (
        f"{args.policy}-rep-{args.rep}-rate-{phase['request_rate']:g}-{timestamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = result_dir / "evidence.json"
    ports = free_ports(13)
    http_ports = ports[:4]
    event_ports = ports[4:8]
    replay_ports = ports[8:12]
    router_port = ports[12]
    config = router_config(args, router_port, http_ports, event_ports, replay_ports)
    config_path = result_dir / "router.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    common_env = os.environ.copy()
    common_env["PYTHONHASHSEED"] = "0"
    common_env["VLLM_PLUGINS"] = "ascend"
    common_env["PYTHONPATH"] = os.pathsep.join(
        (
            str(args.site_packages),
            str(args.vllm_source),
            str(args.ascend_source),
            common_env.get("PYTHONPATH", ""),
        )
    )
    worker_commands = [
        worker_command(args, http_ports[index], event_ports[index], replay_ports[index])
        for index in range(4)
    ]
    workers = []
    for index, command in enumerate(worker_commands):
        env = common_env.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = devices[index]
        workers.append(ChildProcess(command, env, result_dir / f"worker-{index}.log"))
    router_command = [
        str(args.python),
        "-c",
        "from vllm_hust_prefix_router.cli import main; raise SystemExit(main())",
        "serve",
        "--config",
        str(config_path),
    ]
    router = ChildProcess(router_command, common_env, result_dir / "router.log")
    npu_before = run(["npu-smi", "info"], check=False).stdout
    (result_dir / "npu-smi-before.txt").write_text(npu_before, encoding="utf-8")
    occupied = set(devices) & busy_devices(npu_before)
    if occupied:
        raise RuntimeError(
            f"requested devices are already occupied: {sorted(occupied)}"
        )
    plugin_commit = run(
        ["git", "rev-parse", "HEAD"], cwd=args.plugin_repo
    ).stdout.strip()
    plugin_status = run(
        ["git", "status", "--porcelain"], cwd=args.plugin_repo
    ).stdout.strip()
    evidence: dict[str, Any] = {
        "schema_version": "external-router-sharegpt-point/v1",
        "success": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "policy": args.policy,
        "curve_repetition": args.rep,
        "rate_index": args.rate_index,
        "offered_request_rate": phase["request_rate"],
        "topology": {
            "physical_host_count": 1,
            "logical_worker_count": 4,
            "devices": devices,
            "all_backends_use_http": True,
        },
        "software": {
            "plugin_commit": plugin_commit,
            "plugin_status": plugin_status,
            "wheel": str(args.wheel),
            "wheel_sha256": sha256(args.wheel),
            "runner_sha256": sha256(Path(__file__)),
        },
        "workload": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "path": str(workload),
            "sha256": phase["sha256"],
            "total_requests": phase["total_requests"],
            "arrival_seed": phase["arrival_seed"],
            "warmup_seconds": phase["warmup_seconds"],
            "measurement_seconds": phase["measurement_seconds"],
            "prompt_token_min": phase["prompt_token_min"],
            "prompt_token_max": phase["prompt_token_max"],
            "output_tokens": 64,
        },
        "transport": {
            "max_connections_per_backend": args.max_connections,
            "max_pending_requests_per_backend": args.max_pending_requests,
            "queue_timeout_s": args.queue_timeout_s,
        },
        "block_sizes": {
            "scheduler_tokens": args.scheduler_block_size,
            "hash_tokens": args.hash_block_size,
        },
        "commands": {"workers": worker_commands, "router": router_command},
    }
    sampler = MetricsSampler(
        f"http://127.0.0.1:{router_port}/metrics", args.metrics_interval
    )
    started = time.monotonic()
    try:
        for worker in workers:
            worker.start()
        for index, worker in enumerate(workers):
            wait_for_health(
                worker,
                f"http://127.0.0.1:{http_ports[index]}/health",
                args.startup_timeout,
            )
        (result_dir / "npu-smi-running.txt").write_text(
            run(["npu-smi", "info"], check=False).stdout, encoding="utf-8"
        )
        router.start()
        wait_for_health(
            router,
            f"http://127.0.0.1:{router_port}/readyz",
            args.startup_timeout,
        )
        metrics_before = get_json(f"http://127.0.0.1:{router_port}/metrics")
        worker_metrics_before = [
            get_text(f"http://127.0.0.1:{port}/metrics") for port in http_ports
        ]
        sampler.start()
        command = benchmark_command(args, phase, router_port, workload, result_dir)
        evidence["commands"]["benchmark"] = command
        completed = run(command, cwd=args.vllm_source, env=common_env, check=False)
        (result_dir / "benchmark.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark failed with {completed.returncode}")
        result_path = result_dir / "benchmark.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        attempted = int(result.get("completed", 0)) + int(result.get("failed", 0))
        if attempted != int(phase["total_requests"]):
            raise RuntimeError(f"benchmark attempted {attempted} requests")
        if result.get("prompt_sha256") != phase["ordered_prompt_sha256"]:
            raise RuntimeError("benchmark prompt order differs from manifest")
        if result.get("input_lens") != phase["ordered_prompt_tokens"]:
            raise RuntimeError("benchmark token lengths differ from manifest")
        if result.get("failed") == 0 and any(
            value != 64 for value in result.get("output_lens", [])
        ):
            raise RuntimeError("successful responses did not all contain 64 tokens")
        time.sleep(2)
        sampler.stop()
        metrics_after = get_json(f"http://127.0.0.1:{router_port}/metrics")
        worker_metrics_after = [
            get_text(f"http://127.0.0.1:{port}/metrics") for port in http_ports
        ]
        timeline = {"samples": sampler.samples, "errors": sampler.errors}
        timeline_path = result_dir / "router-metrics-timeline.json"
        timeline_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
        evidence.update(
            {
                "benchmark": result,
                "measurement_window": measurement_window(
                    result,
                    start_s=float(phase["warmup_seconds"]),
                    duration_s=float(phase["measurement_seconds"]),
                ),
                "router_metrics_before": metrics_before,
                "router_metrics_after": metrics_after,
                "router_metrics_aggregate": aggregate_router_metrics(sampler.samples),
                "router_metrics_timeline": str(timeline_path),
                "worker_metrics_before": worker_metrics_before,
                "worker_metrics_after": worker_metrics_after,
                "success": True,
            }
        )
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sampler.stop()
        router.stop()
        for worker in reversed(workers):
            worker.stop()
        (result_dir / "npu-smi-after.txt").write_text(
            run(["npu-smi", "info"], check=False).stdout, encoding="utf-8"
        )
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        evidence["duration_seconds"] = time.monotonic() - started
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(evidence_path, flush=True)


if __name__ == "__main__":
    main()
