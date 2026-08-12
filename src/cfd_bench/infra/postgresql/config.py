from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class PostgreSQLConfig:
    """PostgreSQL connection settings shared by ingest and workloads.

    Environment variables make the normal CLI path parameter-free while still
    allowing explicit overrides in code:

    - CFD_BENCH_PG_DB_NAME
    - CFD_BENCH_PG_USER
    - CFD_BENCH_PG_PASSWORD
    - CFD_BENCH_PG_HOST
    - CFD_BENCH_PG_PORT
    """

    db_name: str = field(default_factory=lambda: _env("CFD_BENCH_PG_DB_NAME", "cae_data"))
    db_user: str = field(default_factory=lambda: _env("CFD_BENCH_PG_USER", "postgres"))
    db_password: str = field(default_factory=lambda: _env("CFD_BENCH_PG_PASSWORD", "123456"))
    db_host: str = field(default_factory=lambda: _env("CFD_BENCH_PG_HOST", "localhost"))
    db_port: str = field(default_factory=lambda: _env("CFD_BENCH_PG_PORT", "5432"))
