from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileDBConfig:
    root_path: str = "TileDB_Instances"
    bbox_eps: float = 1e-9
    line_eps: float = 1e-9
    plane_eps: float = 1e-9
    read_chunk: int = 50000
