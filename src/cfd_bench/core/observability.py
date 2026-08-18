"""Lightweight benchmark observability helpers.

Progress reporting is deliberately opt-in so benchmark throughput is unchanged
unless the user explicitly requests diagnostics.  The reporter never controls
or interrupts workload execution; it only observes the current phase and a
transaction counter from a daemon thread.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional


def stage(label: str, message: str) -> None:
    """Print a low-frequency stage marker outside transaction hot loops."""
    print(f"[stage] {label}: {message}", flush=True)


@contextmanager
def timed_stage(label: str, message: str) -> Iterator[None]:
    """Print start/end timing for setup stages without changing control flow."""
    stage(label, f"{message} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stage(label, f"{message} done ({time.perf_counter() - t0:.3f}s)")


class BenchmarkProgress:
    """Passive heartbeat for duration-based benchmark loops.

    The worker thread only reads simple state fields.  Workload code updates
    ``phase`` before a potentially expensive operation and increments ``txn``
    after a completed transaction.  No deadline or cancellation logic lives
    here, so enabling progress cannot change workload correctness/control flow.
    """

    def __init__(
        self,
        label: str,
        duration: float,
        *,
        enabled: bool = False,
        interval: float = 5.0,
    ) -> None:
        self.label = str(label)
        self.duration = max(float(duration), 0.0)
        self.enabled = bool(enabled)
        self.interval = max(float(interval), 0.5)
        self.phase = "starting"
        self.transactions = 0
        self._start = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def __enter__(self) -> "BenchmarkProgress":
        self._start = time.monotonic()
        stage(self.label, f"benchmark loop start (target window={self.duration:g}s)")
        if self.enabled:
            self._thread = threading.Thread(target=self._report_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(0.25, self.interval))
        if self.enabled and self._tty:
            print(file=sys.stdout, flush=True)
        elapsed = time.monotonic() - self._start
        stage(self.label, f"benchmark loop end (wall={elapsed:.3f}s, txns={self.transactions})")
        return False

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase)

    def transaction(self, count: int = 1) -> None:
        self.transactions += int(count)

    def _report_loop(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = max(0.0, time.monotonic() - self._start)
            if self.duration > 0:
                pct = min(100.0, 100.0 * elapsed / self.duration)
            else:
                pct = 100.0
            msg = (
                f"[progress] {self.label}: {pct:5.1f}% "
                f"elapsed={elapsed:7.1f}s txns={self.transactions} phase={self.phase}"
            )
            if self._tty:
                width = 20
                filled = int(round(width * pct / 100.0))
                bar = "#" * filled + "-" * (width - filled)
                print(f"\r{msg} [{bar}]", end="", file=sys.stdout, flush=True)
            else:
                print(msg, file=sys.stdout, flush=True)


def benchmark_progress(
    label: str,
    duration: float,
    *,
    enabled: bool = False,
    interval: float = 5.0,
) -> BenchmarkProgress:
    return BenchmarkProgress(label, duration, enabled=enabled, interval=interval)
