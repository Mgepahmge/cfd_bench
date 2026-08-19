"""Low-overhead progress display for legacy CFD topology decoding.

This module is intentionally independent of database backends and never
controls parsing.  It only renders progress updates emitted by the decoder and
canonical exporter.  H5 ingest does not use it.
"""

from __future__ import annotations

import sys
import time
from typing import Optional


def _fmt_count(value: int) -> str:
    n = float(max(0, int(value)))
    for suffix in ("", "K", "M", "G"):
        if n < 1000.0 or suffix == "G":
            if suffix:
                return f"{n:.1f}{suffix}"
            return str(int(n))
        n /= 1000.0
    return str(int(value))


class TopologyProgress:
    """Passive console progress renderer for a single topology parse."""

    def __init__(self, *, enabled: bool = True, min_interval: float = 0.20) -> None:
        self.enabled = bool(enabled)
        self.min_interval = max(float(min_interval), 0.05)
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._phase: Optional[str] = None
        self._last_time = 0.0
        self._last_bucket = -1
        self._line_open = False

    def __call__(self, phase: str, current: int, total: int) -> None:
        if not self.enabled:
            return
        phase = str(phase)
        current = max(0, int(current))
        total = max(1, int(total))
        current = min(current, total)
        now = time.monotonic()
        pct = 100.0 * current / total
        bucket = min(10, int(pct // 10))
        phase_changed = phase != self._phase
        finished = current >= total

        if not phase_changed and not finished:
            if self._tty:
                if now - self._last_time < self.min_interval:
                    return
            else:
                if bucket <= self._last_bucket:
                    return

        if phase_changed:
            self._finish_tty_line()
            self._phase = phase
            self._last_bucket = -1

        if self._tty:
            width = 28
            filled = min(width, int(round(width * pct / 100.0)))
            bar = "#" * filled + "-" * (width - filled)
            msg = (
                f"\r[ingest] topology: {phase:<34.34} "
                f"[{bar}] {pct:6.1f}% ({_fmt_count(current)}/{_fmt_count(total)})"
            )
            print(msg, end="", file=sys.stdout, flush=True)
            self._line_open = True
            if finished:
                print(file=sys.stdout, flush=True)
                self._line_open = False
        else:
            # CI/log files get at most ~11 lines per phase rather than one line
            # per face/cell.
            print(
                f"[ingest] topology: {phase} {pct:5.1f}% "
                f"({_fmt_count(current)}/{_fmt_count(total)})",
                file=sys.stdout,
                flush=True,
            )

        self._last_time = now
        self._last_bucket = bucket

    def _finish_tty_line(self) -> None:
        if self._tty and self._line_open:
            print(file=sys.stdout, flush=True)
            self._line_open = False

    def close(self) -> None:
        self._finish_tty_line()
