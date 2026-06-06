from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostgreSQLConfig:
    db_name: str = "cae_data"
    db_user: str = "postgres"
    db_password: str = "123456"
    db_host: str = "localhost"
    db_port: str = "5432"
