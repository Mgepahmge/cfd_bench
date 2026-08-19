"""PostgreSQL schema bootstrap for CFD-Bench mesh / spatial tables."""

from __future__ import annotations

from pathlib import Path

from cfd_bench.ingest.postgresql.pg_io import pg_connect

_SQL_DIR = Path(__file__).resolve().parent / "sql"
_SQL_FILES = (
    "00_extensions.sql",
    "01_mesh_tables.sql",
    "02_spatial_tables.sql",
    "03_boundary_tables.sql",
    "04_h5_metadata.sql",
    "05_benchmark_runtime.sql",
    "06_cfd_runtime.sql",
)


def apply_pg_schema(conn=None) -> None:
    """Apply DDL scripts in order (idempotent)."""
    own = conn is None
    if own:
        conn = pg_connect()
    cur = conn.cursor()
    try:
        for name in _SQL_FILES:
            sql_path = _SQL_DIR / name
            if not sql_path.is_file():
                raise FileNotFoundError(sql_path)
            print(f"Applying PG schema: {name}")
            cur.execute(sql_path.read_text(encoding="utf-8"))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        if own and conn:
            conn.close()
