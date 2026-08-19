"""IoTDB ingest write helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from iotdb.Session import Session
from iotdb.table_session import Tablet
from iotdb.utils.Field import TSDataType


def insert_tablet_chunked(
    device: str,
    session: Session,
    times: Sequence[int],
    measurements: Sequence[str],
    types: Sequence[TSDataType],
    rows: Sequence[Sequence[object]],
    *,
    chunk_size: int = 50_000,
) -> None:
    """Insert a large logical tablet in bounded chunks.

    Older CFD ingest created one enormous Python list/Tablet per full mesh.
    Chunking keeps peak memory predictable on million-cell datasets while
    preserving identical IoTDB paths and timestamps.
    """
    n = len(times)
    if n == 0:
        return
    if len(rows) != n:
        raise ValueError(f"tablet row/time mismatch for {device}: rows={len(rows)} times={n}")
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        tablet = Tablet(
            device,
            list(measurements),
            list(types),
            [list(r) for r in rows[start:end]],
            [int(x) for x in times[start:end]],
        )
        session.insert_tablet(tablet)


def insert_numpy_columns(
    device: str,
    session: Session,
    times: Sequence[int],
    columns: dict,
    types: Sequence[TSDataType],
    *,
    chunk_size: int = 50_000,
) -> None:
    names = list(columns)
    arrays = [np.asarray(columns[name]) for name in names]
    n = len(times)
    if any(len(a) != n for a in arrays):
        raise ValueError(f"IoTDB column length mismatch for {device}")
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        values = np.column_stack([a[start:end] for a in arrays]).tolist() if arrays else []
        tablet = Tablet(
            device,
            names,
            list(types),
            values,
            [int(x) for x in times[start:end]],
        )
        session.insert_tablet(tablet)


def delete_timeseries_prefix(session: Session, prefix: str) -> None:
    """Best-effort overwrite helper for legacy CFD devices."""
    try:
        session.execute_non_query_statement(f"DELETE TIMESERIES {prefix}.**")
    except Exception:
        # A missing path/server-version syntax difference must not block a
        # fresh ingest; writes below still overwrite matching timestamps.
        pass


def load_dataframe_to_iotdb(device_directory: str, session: Session, df: pd.DataFrame):
    """Backward-compatible DataFrame writer, now chunked."""
    times = [int(x) for x in df.index.values]
    measurements = list(df.columns)
    arrays = [np.asarray(df[col].values) for col in df.columns]
    # Historical callers use numeric topology/field frames; retain DOUBLE
    # semantics for this compatibility wrapper.
    types = [TSDataType.DOUBLE] * len(measurements)
    for start in range(0, len(times), 50_000):
        end = min(start + 50_000, len(times))
        values = np.column_stack([a[start:end] for a in arrays]).tolist() if arrays else []
        session.insert_tablet(
            Tablet(
                device_directory,
                measurements,
                types,
                values,
                times[start:end],
            )
        )


def insert_ragged_int_rows(
    device: str,
    session: Session,
    times: Sequence[int],
    rows: Sequence[Sequence[int]],
    prefix: str,
    width: int,
    *,
    chunk_size: int = 50_000,
) -> None:
    width = int(max(1, width))
    names = [f"{prefix}_{i}" for i in range(width)]
    types = [TSDataType.INT64] * width
    n = len(times)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        vals = []
        for row in rows[start:end]:
            rr = [int(x) for x in row]
            if len(rr) > width:
                raise ValueError(f"{device}: topology row width {len(rr)} exceeds {width}")
            vals.append(rr + [-1] * (width - len(rr)))
        session.insert_tablet(Tablet(device, names, types, vals, [int(x) for x in times[start:end]]))
