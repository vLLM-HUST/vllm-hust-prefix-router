from experiments.sharegpt.run_matrix import point_order
from experiments.sharegpt.run_point import (
    aggregate_router_metrics,
    busy_devices,
    measurement_window,
    select_phase,
)


def test_point_order_alternates_first_policy_within_each_pair() -> None:
    assert point_order([0], [0, 1]) == [
        (0, 0, "prefix"),
        (0, 0, "lifecycle"),
        (0, 1, "lifecycle"),
        (0, 1, "prefix"),
    ]


def test_select_phase_requires_exact_rep_and_rate() -> None:
    manifest = {
        "repetitions": [
            {
                "phases": [
                    {"curve_repetition": 0, "rate_index": 0},
                    {"curve_repetition": 0, "rate_index": 1},
                ]
            }
        ]
    }
    assert select_phase(manifest, 0, 1)["rate_index"] == 1


def test_router_metric_aggregate_keeps_observed_maxima() -> None:
    samples = [
        {
            "metrics": {
                "routing": {
                    "lifecycle": {"node0": {"active_requests": 2, "tracked_prefill": 8}}
                },
                "backends": {
                    "node0": {
                        "active_requests": 3,
                        "queued_requests": 1,
                        "queue_wait_seconds": 0.25,
                    }
                },
            }
        },
        {
            "metrics": {
                "routing": {
                    "lifecycle": {
                        "node0": {"active_requests": 1, "tracked_prefill": 12}
                    }
                },
                "backends": {
                    "node0": {
                        "active_requests": 4,
                        "queued_requests": 0,
                        "queue_wait_seconds": 0.5,
                    }
                },
            }
        },
    ]
    result = aggregate_router_metrics(samples)
    assert result["max_lifecycle"]["node0"] == {
        "active_requests": 2.0,
        "tracked_prefill": 12.0,
    }
    assert result["max_backend_pool"]["node0"] == {
        "active_requests": 4.0,
        "queued_requests": 1.0,
        "queue_wait_seconds": 0.5,
    }


def test_busy_devices_reads_only_npu_process_rows() -> None:
    output = """
| 4     910B2               | OK            | 99.1 |
| 4                         | 0000:81:00.0  | 0    |
| 4       0                 | 164986        |      |
| No running processes found in NPU 5              |
"""
    assert busy_devices(output) == {"4"}


def test_measurement_window_accepts_current_benchmark_timings() -> None:
    result = {
        "start_times": [101.0, 103.0, 104.0],
        "queue_times": [0.0, 1.0, 0.5],
        "latencies": [2.0, 3.0, 1.0],
        "ttfts": [0.5, 1.0, 0.25],
        "output_lens": [4, 4, 4],
        "errors": ["", "", ""],
    }
    window = measurement_window(result, start_s=1.0, duration_s=2.0)
    assert window["arrival_count"] == 2
    assert window["terminal_count"] == 1
    assert window["backlog_at_start"] == 1
    assert window["backlog_at_end"] == 2
    assert window["p95_concurrency_wait_ms"] == 975.0
