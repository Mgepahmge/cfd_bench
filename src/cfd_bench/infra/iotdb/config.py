from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class IoTDBConfig:
    host: str = field(default_factory=lambda: os.getenv("CFD_BENCH_IOTDB_HOST", "127.0.0.1"))
    port: str = field(default_factory=lambda: os.getenv("CFD_BENCH_IOTDB_PORT", "6667"))
    user: str = field(default_factory=lambda: os.getenv("CFD_BENCH_IOTDB_USER", "root"))
    password: str = field(default_factory=lambda: os.getenv("CFD_BENCH_IOTDB_PASSWORD", "root"))
    root_path: str = field(default_factory=lambda: os.getenv("CFD_BENCH_IOTDB_ROOT_PATH", "root.simulation_data"))
    query_fetch_size: int = field(
        default_factory=lambda: int(os.getenv("CFD_BENCH_IOTDB_FETCH_SIZE", "50000"))
    )
    max_cache_entries: int = 16
    bbox_eps: float = 1e-9
    line_eps: float = 1e-9
    plane_eps: float = 1e-9
