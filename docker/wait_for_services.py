"""Wait for the database services using the exact CFD-Bench credentials.

This executes in a short-lived Python process before the real command.  That
also isolates any import-time warning-filter side effects from third-party
clients (notably the IoTDB Python client) from the benchmark process itself.

A TCP/RPC connection alone is not enough for IoTDB: during standalone startup
the DataNode RPC port may already accept sessions before the DataNode has
successfully registered with the ConfigNode.  We therefore require a
``SHOW DATANODES`` row whose status is ``Running`` before releasing the app.
"""

from __future__ import annotations

import os
import sys
import time

TIMEOUT = float(os.getenv("CFD_BENCH_SERVICE_WAIT_TIMEOUT", "300"))
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


def _field_text(field):
    if field is None:
        return ""
    try:
        value = field.get_string_value()
    except Exception:
        value = str(field)
    return "" if value is None else str(value).strip()


def check_iotdb():
    from iotdb.Session import Session

    session = Session(
        os.getenv("CFD_BENCH_IOTDB_HOST", "iotdb"),
        os.getenv("CFD_BENCH_IOTDB_PORT", "6667"),
        os.getenv("CFD_BENCH_IOTDB_USER", "root"),
        os.getenv("CFD_BENCH_IOTDB_PASSWORD", "root"),
        fetch_size=int(os.getenv("CFD_BENCH_IOTDB_FETCH_SIZE", "50000")),
    )
    dataset = None
    try:
        session.open()
        dataset = session.execute_query_statement("SHOW DATANODES")
        running = False
        while dataset.has_next():
            row = dataset.next()
            values = [_field_text(field) for field in row.get_fields()]
            if any(value.lower() == "running" for value in values):
                running = True
                break
        if not running:
            raise RuntimeError("IoTDB RPC is reachable but no Running DataNode is registered")
    finally:
        if dataset is not None:
            try:
                dataset.close_operation_handle()
            except Exception:
                pass
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    wait_for("PostgreSQL/PostGIS", check_postgres)
    wait_for("Apache IoTDB (Running DataNode)", check_iotdb)
