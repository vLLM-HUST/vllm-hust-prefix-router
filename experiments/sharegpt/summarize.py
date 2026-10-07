#!/usr/bin/env python3
"""Produce raw and paired summaries for the ShareGPT routing matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "arrival_rate",
    "terminal_rate",
    "successful_completion_rate",
    "backlog_growth",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "p95_concurrency_wait_ms",
    "p99_concurrency_wait_ms",
)


def load_latest(result_root: Path) -> dict[tuple[int, float, str], dict[str, Any]]:
    selected: dict[tuple[int, float, str], tuple[float, dict[str, Any]]] = {}
    for path in result_root.glob("*-rep-*-rate-*/evidence.json"):
        try:
            evidence = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if evidence.get("success") is not True:
            continue
        key = (
            int(evidence["curve_repetition"]),
            float(evidence["offered_request_rate"]),
            str(evidence["policy"]),
        )
        modified = path.stat().st_mtime
        if key not in selected or modified > selected[key][0]:
            evidence["evidence_path"] = str(path)
            selected[key] = (modified, evidence)
    return {key: value[1] for key, value in selected.items()}


def route_fields(evidence: dict[str, Any]) -> dict[str, Any]:
    after = evidence.get("router_metrics_after", {})
    fields: dict[str, Any] = {}
    for node, stats in after.get("backends", {}).items():
        fields[f"{node}_routes"] = stats.get("admitted_requests")
        fields[f"{node}_rejected"] = stats.get("rejected_requests")
        fields[f"{node}_max_connections"] = stats.get("max_connections")
    fields["event_sources_ready"] = all(
        source.get("ready") is True for source in after.get("event_sources", [])
    )
    return fields


def raw_row(evidence: dict[str, Any]) -> dict[str, Any]:
    window = evidence["measurement_window"]
    benchmark = evidence["benchmark"]
    return {
        "rep": evidence["curve_repetition"],
        "rate": evidence["offered_request_rate"],
        "policy": evidence["policy"],
        "completed": benchmark.get("completed"),
        "failed": benchmark.get("failed"),
        **{metric: window.get(metric) for metric in METRICS},
        **route_fields(evidence),
        "evidence_path": evidence["evidence_path"],
    }


def paired_rows(
    selected: dict[tuple[int, float, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    pairs = sorted({(rep, rate) for rep, rate, _policy in selected})
    for rep, rate in pairs:
        prefix = selected.get((rep, rate, "prefix"))
        lifecycle = selected.get((rep, rate, "lifecycle"))
        if prefix is None or lifecycle is None:
            continue
        row: dict[str, Any] = {"rep": rep, "rate": rate}
        for metric in METRICS:
            left = prefix["measurement_window"].get(metric)
            right = lifecycle["measurement_window"].get(metric)
            row[f"prefix_{metric}"] = left
            row[f"lifecycle_{metric}"] = right
            row[f"delta_{metric}"] = (
                float(right) - float(left)
                if left is not None and right is not None
                else None
            )
            row[f"ratio_{metric}"] = (
                float(right) / float(left)
                if left not in (None, 0) and right is not None
                else None
            )
        rows.append(row)
    return rows


def aggregate_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rate: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rate[float(row["rate"])].append(row)
    aggregates = []
    for rate, group in sorted(by_rate.items()):
        result: dict[str, Any] = {"rate": rate, "pair_count": len(group)}
        for metric in METRICS:
            for prefix in ("prefix", "lifecycle", "delta", "ratio"):
                key = f"{prefix}_{metric}"
                values = [float(row[key]) for row in group if row[key] is not None]
                result[f"median_{key}"] = statistics.median(values) if values else None
                if prefix in ("delta", "ratio"):
                    result[f"min_{key}"] = min(values) if values else None
                    result[f"max_{key}"] = max(values) if values else None
        aggregates.append(result)
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    selected = load_latest(args.result_root)
    raw = [raw_row(value) for _key, value in sorted(selected.items())]
    paired = paired_rows(selected)
    aggregate = aggregate_pairs(paired)
    summary = {
        "schema_version": "external-router-sharegpt-summary/v1",
        "successful_point_count": len(raw),
        "paired_point_count": len(paired),
        "complete": len(raw) == 42 and len(paired) == 21,
        "raw": raw,
        "paired": paired,
        "by_rate": aggregate,
    }
    (args.result_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.result_root / "raw-points.csv", raw)
    write_csv(args.result_root / "paired-points.csv", paired)
    write_csv(args.result_root / "paired-by-rate.csv", aggregate)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("successful_point_count", "paired_point_count", "complete")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
