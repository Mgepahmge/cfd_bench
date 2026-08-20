"""Structured benchmark result output.

The console remains the primary human-readable output.  When ``cfd-bench run``
is given ``--output <file.csv>``, completed benchmark sections are mirrored to
CSV *after* their timed loops finish.  No per-transaction file I/O is performed.
"""

from __future__ import annotations

import csv
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional


_BACKEND_NAMES: Dict[str, str] = {
    "PG": "postgresql",
    "PostgreSQL": "postgresql",
    "IoTDB": "iotdb",
    "TileDB": "tiledb",
    "VTK": "vtk",
}


@dataclass(frozen=True)
class ResultContext:
    workload: str = ""
    dataset: str = ""
    step: Optional[int] = None


_CONTEXT: ContextVar[ResultContext] = ContextVar(
    "cfd_bench_result_context", default=ResultContext()
)
_ACTIVE_WRITER: Optional["CsvResultWriter"] = None


class CsvResultWriter:
    """One CSV writer per ``cfd-bench run`` invocation."""

    FIELDNAMES = (
        "run_id",
        "timestamp_utc",
        "dataset",
        "workload",
        "backend",
        "operation",
        "step",
        "transactions",
        "duration_sec",
        "txns_per_sec",
        "details",
    )

    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:12]
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def write(
        self,
        *,
        context: ResultContext,
        backend: str,
        operation: str,
        transactions: int,
        duration_sec: float,
        step: Optional[int],
        details: str,
    ) -> None:
        duration = float(duration_sec)
        txn = int(transactions)
        row = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": context.dataset,
            "workload": context.workload,
            "backend": _BACKEND_NAMES.get(str(backend), str(backend).lower()),
            "operation": str(operation),
            "step": "" if step is None else int(step),
            "transactions": txn,
            "duration_sec": f"{duration:g}",
            "txns_per_sec": f"{(txn / duration) if duration > 0 else 0.0:.9g}",
            "details": str(details or ""),
        }
        self._writer.writerow(row)
        # Rows are emitted only after a benchmark section has finished, so a
        # flush here cannot perturb the timed transaction loop.  It also keeps
        # useful partial results if a later workload fails.
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@contextmanager
def csv_result_output(path: Optional[str]) -> Iterator[Optional[CsvResultWriter]]:
    """Enable structured CSV output for the duration of one run command."""

    global _ACTIVE_WRITER
    previous = _ACTIVE_WRITER
    if not path:
        yield None
        return

    writer = CsvResultWriter(path)
    _ACTIVE_WRITER = writer
    try:
        yield writer
    finally:
        _ACTIVE_WRITER = previous
        writer.close()


@contextmanager
def result_context(workload: str, dataset: str, step: Optional[int] = None) -> Iterator[None]:
    """Attach dataset/workload identity to results emitted by workload code."""

    token = _CONTEXT.set(
        ResultContext(
            workload=str(workload).lower(),
            dataset=str(dataset),
            step=None if step is None else int(step),
        )
    )
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def emit_benchmark_result(
    message: str,
    *,
    backend: str,
    operation: str,
    transactions: int,
    duration_sec: float,
    step: Optional[int] = None,
    details: str = "",
) -> None:
    """Print an existing result line and optionally mirror it to CSV.

    This function must be called after the benchmark timing loop.  With no
    active CSV writer its behavior is exactly one normal ``print`` call.
    """

    print(message)
    writer = _ACTIVE_WRITER
    if writer is None:
        return
    context = _CONTEXT.get()
    resolved_step = context.step if step is None else int(step)
    writer.write(
        context=context,
        backend=backend,
        operation=operation,
        transactions=transactions,
        duration_sec=duration_sec,
        step=resolved_step,
        details=details,
    )
