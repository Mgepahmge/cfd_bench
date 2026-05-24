from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IoTDBConfig:
    host: str = "127.0.0.1"
    port: str = "6667"
    user: str = "root"
    password: str = "root"
    root_path: str = "root.simulation_data"
    max_cache_entries: int = 16
    bbox_eps: float = 1e-9
    line_eps: float = 1e-9
    plane_eps: float = 1e-9
