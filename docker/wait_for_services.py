"""Wait for the database services using the exact CFD-Bench credentials.

This executes in a short-lived Python process before the real command.  That
also isolates any import-time warning-filter side effects from third-party
clients (notably the IoTDB Python client) from the benchmark process itself.
"""

from __future__ import annotations

import os
import sys
import time

TIMEOUT = float(os.getenv("CFD_BENCH_SERVICE_WAIT_TIMEOUT", "180"))
INTERVAL = 2.0


def wait_for(name, connect_once):
    deadline = time.monotonic() + TIMEOUT
    last_error = None
    while time.monotonic() < deadline:
        try:
            connect_once()
            print(f"[docker] {name}: ready", flush=True)
            return
        except Exception as exc:  # service startup can fail in many transient ways
            last_error = exc
            time.sleep(INTERVAL)
    print(f"[docker] {name}: not ready after {TIMEOUT:g}s: {last_error}", file=sys.stderr)
    raise SystemExit(1)


def check_postgres():
    import psycopg2

    conn = psycopg2.connect(
        dbname=os.getenv("CFD_BENCH_PG_DB_NAME", "cae_data"),
        user=os.getenv("CFD_BENCH_PG_USER", "postgres"),
        password=os.getenv("CFD_BENCH_PG_PASSWORD", "123456"),
        host=os.getenv("CFD_BENCH_PG_HOST", "postgres"),
        port=os.getenv("CFD_BENCH_PG_PORT", "5432"),
        connect_timeout=3,
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()


def check_iotdb():
    from iotdb.Session import Session

    session = Session(
        os.getenv("CFD_BENCH_IOTDB_HOST", "iotdb"),
        os.getenv("CFD_BENCH_IOTDB_PORT", "6667"),
        os.getenv("CFD_BENCH_IOTDB_USER", "root"),
        os.getenv("CFD_BENCH_IOTDB_PASSWORD", "root"),
        fetch_size=int(os.getenv("CFD_BENCH_IOTDB_FETCH_SIZE", "50000")),
    )
    try:
        session.open()
    finally:
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    wait_for("PostgreSQL/PostGIS", check_postgres)
    wait_for("Apache IoTDB", check_iotdb)
