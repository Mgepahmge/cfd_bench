"""Runtime configuration for the standalone CFD-Bench HTTP adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


MIB = 1024 * 1024


@dataclass(frozen=True)
class ApiConfig:
    data_root: Path
    cfd_bench_executable: str = "cfd-bench"
    recommended_chunk_size: int = 64 * MIB
    max_chunk_size: int = 128 * MIB
    scheduler_poll_sec: float = 0.25
    cancel_grace_sec: float = 5.0

    @classmethod
    def from_env(cls) -> "ApiConfig":
        return cls(
            data_root=Path(
                os.getenv("CFD_BENCH_API_DATA_ROOT", "/app/api-data")
            ).expanduser(),
            cfd_bench_executable=os.getenv("CFD_BENCH_API_CLI", "cfd-bench"),
            recommended_chunk_size=int(
                os.getenv("CFD_BENCH_API_CHUNK_SIZE", str(64 * MIB))
            ),
            max_chunk_size=int(
                os.getenv("CFD_BENCH_API_MAX_CHUNK_SIZE", str(128 * MIB))
            ),
            scheduler_poll_sec=float(
                os.getenv("CFD_BENCH_API_SCHEDULER_POLL_SEC", "0.25")
            ),
            cancel_grace_sec=float(
                os.getenv("CFD_BENCH_API_CANCEL_GRACE_SEC", "5.0")
            ),
        )

    @property
    def uploads_root(self) -> Path:
        return self.data_root / "uploads"

    @property
    def jobs_root(self) -> Path:
        return self.data_root / "jobs"

    @property
    def state_db(self) -> Path:
        return self.data_root / "state.db"

    def ensure_directories(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
