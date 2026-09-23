#!/usr/bin/env python3
"""Run the paired 3 x 7 Prefix/Lifecycle matrix with resumable progress."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RATES = (2.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0)


def now() -> str:
    return datetime.now(UTC).isoformat()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def npu_state() -> tuple[set[str], set[str], str]:
    completed = run(["npu-smi", "info"])
    if completed.returncode != 0:
        raise RuntimeError(f"npu-smi failed:\n{completed.stdout}")
    visible = set(
        re.findall(r"^\|\s*(\d+)\s+910\S*\s*\|", completed.stdout, re.MULTILINE)
    )
    busy = {
        device
        for device, _chip, _pid in re.findall(
            r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|",
            completed.stdout,
            re.MULTILINE,
        )
    }
    return visible, busy, completed.stdout


def wait_for_fixed_devices(devices: list[str], wait_seconds: float) -> None:
    while True:
        visible, busy, _ = npu_state()
        missing = set(devices) - visible
        occupied = set(devices) & busy
        if missing:
            raise RuntimeError(f"fixed devices are not visible: {sorted(missing)}")
        if not occupied:
            time.sleep(5)
            _, busy_again, _ = npu_state()
            if not set(devices) & busy_again:
                return
        print(
            f"[{now()}] waiting_for_fixed_devices occupied={sorted(occupied)}",
            flush=True,
        )
        time.sleep(wait_seconds)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def successful_evidence(
    result_root: Path, policy: str, rep: int, rate: float
) -> Path | None:
    pattern = f"{policy}-rep-{rep}-rate-{rate:g}-*/evidence.json"
    paths = sorted(result_root.glob(pattern), key=lambda value: value.stat().st_mtime)
    for path in reversed(paths):
        evidence = read_json(path)
        if evidence is not None and evidence.get("success") is True:
            return path
    return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def point_order(
    repetitions: list[int], rate_indices: list[int]
) -> list[tuple[int, int, str]]:
    points = []
    for rep in repetitions:
        for rate_index in rate_indices:
            policies = (
                ("prefix", "lifecycle")
                if (rep + rate_index) % 2 == 0
                else ("lifecycle", "prefix")
            )
            points.extend((rep, rate_index, policy) for policy in policies)
    return points


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-cli", type=Path, required=True)
    parser.add_argument("--plugin-repo", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--ascend-source", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--repetitions", default="0,1,2")
    parser.add_argument("--rate-indices", default="0,1,2,3,4,5,6")
    parser.add_argument("--device-wait-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    repetitions = parse_int_list(args.repetitions)
    rate_indices = parse_int_list(args.rate_indices)
    if any(value not in range(3) for value in repetitions):
        raise SystemExit("repetitions must be selected from 0,1,2")
    if any(value not in range(len(RATES)) for value in rate_indices):
        raise SystemExit("rate indices must be selected from 0..6")
    progress_path = args.result_root / "matrix-progress.json"
    completed_points: list[dict[str, Any]] = []
    for rep, rate_index, policy in point_order(repetitions, rate_indices):
        rate = RATES[rate_index]
        existing = successful_evidence(args.result_root, policy, rep, rate)
        if existing is not None:
            completed_points.append(
                {
                    "rep": rep,
                    "rate_index": rate_index,
                    "rate": rate,
                    "policy": policy,
                    "status": "already_successful",
                    "evidence": str(existing),
                }
            )
            continue
        wait_for_fixed_devices(devices, args.device_wait_seconds)
        point = {
            "rep": rep,
            "rate_index": rate_index,
            "rate": rate,
            "policy": policy,
            "status": "running",
            "started_at": now(),
        }
        write_json(
            progress_path,
            {"updated_at": now(), "current": point, "completed": completed_points},
        )
        command = [
            str(args.python),
            str(args.runner),
            "--policy",
            policy,
            "--rep",
            str(rep),
            "--rate-index",
            str(rate_index),
            "--devices",
            ",".join(devices),
            "--manifest",
            str(args.manifest),
            "--result-root",
            str(args.result_root),
            "--model",
            str(args.model),
            "--vllm-cli",
            str(args.vllm_cli),
            "--python",
            str(args.python),
            "--plugin-repo",
            str(args.plugin_repo),
            "--wheel",
            str(args.wheel),
            "--site-packages",
            str(args.site_packages),
            "--vllm-source",
            str(args.vllm_source),
            "--ascend-source",
            str(args.ascend_source),
        ]
        log_path = args.result_root / (f"matrix-rep-{rep}-rate-{rate:g}-{policy}.log")
        result = run(command)
        log_path.write_text(result.stdout, encoding="utf-8")
        point.update(
            {
                "finished_at": now(),
                "returncode": result.returncode,
                "log": str(log_path),
            }
        )
        evidence = successful_evidence(args.result_root, policy, rep, rate)
        if result.returncode != 0 or evidence is None:
            point["status"] = "failed"
            write_json(
                progress_path,
                {
                    "updated_at": now(),
                    "current": point,
                    "completed": completed_points,
                },
            )
            raise RuntimeError(f"matrix point failed; see {log_path}")
        point.update({"status": "success", "evidence": str(evidence)})
        completed_points.append(point)
        write_json(
            progress_path,
            {"updated_at": now(), "current": None, "completed": completed_points},
        )
    summary = run(
        [
            str(args.python),
            str(args.summarizer),
            "--result-root",
            str(args.result_root),
        ]
    )
    (args.result_root / "summary.log").write_text(summary.stdout, encoding="utf-8")
    if summary.returncode != 0:
        raise RuntimeError("matrix completed but summarization failed")
    write_json(
        progress_path,
        {
            "updated_at": now(),
            "current": None,
            "completed": completed_points,
            "status": "completed",
        },
    )


if __name__ == "__main__":
    main()
