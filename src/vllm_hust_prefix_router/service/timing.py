"""Opt-in bounded diagnostics; never used for admission or cache validity."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class RequestTiming:
    """Relative timestamps on one process clock, not engine/token timestamps."""

    def __init__(self, *, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self._clock = clock
        self._start = clock()
        self.events: dict[str, int] = {"entry": 0}
        self.request_id: str | None = None
        self.node_id: str | None = None
        self.terminal = "router_error"
        self.http_status: int | None = None

    def mark(self, name: str) -> None:
        if name not in self.events:
            self.events[name] = self._clock() - self._start

    def snapshot(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "terminal": self.terminal,
            "http_status": self.http_status,
            "elapsed_ns": dict(self.events),
            "clock": "process_relative_perf_counter_ns",
            "first_body_is_first_token": False,
        }


class LoopLagProbe:
    """Bounded aggregate wakeup delay, not an event-loop P99 estimator."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self.samples = 0
        self.total_s = 0.0
        self.max_s = 0.0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="router-loop-lag")

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            expected = loop.time() + 0.05
            await asyncio.sleep(0.05)
            lag = max(0.0, loop.time() - expected)
            self.samples += 1
            self.total_s += lag
            self.max_s = max(self.max_s, lag)

    def snapshot(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "mean_s": self.total_s / self.samples if self.samples else None,
            "max_s": self.max_s if self.samples else None,
            "interval_s": 0.05,
            "running": self._task is not None and not self._task.done(),
        }
