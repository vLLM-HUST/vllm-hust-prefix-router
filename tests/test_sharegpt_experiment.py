from experiments.sharegpt.run_matrix import point_order
from experiments.sharegpt.run_point import aggregate_router_metrics, select_phase


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
                    "lifecycle": {
                        "node0": {"active_requests": 2, "tracked_prefill": 8}
                    }
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
