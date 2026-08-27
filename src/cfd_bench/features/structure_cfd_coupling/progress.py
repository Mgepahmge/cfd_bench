"""Low-overhead progress reporting for structure/CFD coupling."""

from __future__ import annotations

import sys
import time
from typing import Optional


def _fmt_count(value: int) -> str:
    n = float(max(0, int(value)))
    for suffix in ("", "K", "M", "G"):
        if n < 1000.0 or suffix == "G":
            return f"{n:.1f}{suffix}" if suffix else str(int(n))
        n /= 1000.0
    return str(int(value))


class CouplingProgress:
    """Console progress bar that remains compact in non-interactive logs."""

    def __init__(self, *, enabled: bool = True, min_interval: float = 0.25) -> None:
        self.enabled = bool(enabled)
        self.min_interval = max(0.05, float(min_interval))
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._last_time = 0.0
        self._last_bucket = -1
        self._line_open = False
        self._phase: Optional[str] = None

    def stage(self, message: str) -> None:
        if not self.enabled:
            return
        self._finish_line()
        print(f"[coupling] {message}", flush=True)

    def update(self, current: int, total: int, *, phase: str = "map structure nodes") -> None:
        if not self.enabled:
            return
        current = max(0, int(current))
        total = max(1, int(total))
        current = min(current, total)
        now = time.monotonic()
        pct = 100.0 * current / total
        bucket = min(10, int(pct // 10.0))
        phase_changed = phase != self._phase
        finished = current >= total

        if not phase_changed and not finished:
            if self._tty:
                if now - self._last_time < self.min_interval:
                    return
            elif bucket <= self._last_bucket:
                return

        if phase_changed:
            self._finish_line()
            self._phase = str(phase)
            self._last_bucket = -1

        if self._tty:
            width = 28
            filled = min(width, int(round(width * pct / 100.0)))
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\r[coupling] {phase:<24.24} [{bar}] {pct:6.1f}% "
                f"({_fmt_count(current)}/{_fmt_count(total)})",
                end="",
                flush=True,
            )
            self._line_open = True
            if finished:
                print(flush=True)
                self._line_open = False
        else:
            print(
                f"[coupling] {phase}: {pct:5.1f}% "
                f"({_fmt_count(current)}/{_fmt_count(total)})",
                flush=True,
            )

        self._last_time = now
        self._last_bucket = bucket

    def _finish_line(self) -> None:
        if self._tty and self._line_open:
            print(flush=True)
            self._line_open = False

    def close(self) -> None:
        self._finish_line()
