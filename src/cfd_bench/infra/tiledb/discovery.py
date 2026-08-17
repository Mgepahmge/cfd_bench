"""Runtime metadata discovery for H5 datasets stored in TileDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from cfd_bench.infra.tiledb.config import TileDBConfig


@dataclass(frozen=True)
class TileDBDatasetInfo:
    dataset_key: str
    zone_type: str
    timesteps: Tuple[int, ...]
    variables: Tuple[str, ...]


def discover_tiledb_datasets(
    selected_datasets: Iterable[str],
    *,
    config: TileDBConfig | None = None,
) -> List[TileDBDatasetInfo]:
    # Lazy import keeps PostgreSQL/IoTDB-only installations independent from
    # the optional TileDB-Py package.
    from cfd_bench.infra.tiledb.repository import TileDBRepository

    repo = TileDBRepository(config or TileDBConfig())
    out: List[TileDBDatasetInfo] = []
    for dataset in selected_datasets:
        key = str(dataset)
        try:
            meta = repo.h5_dataset_metadata(key)
            if not meta.get("is_h5"):
                continue
            steps = tuple(repo.h5_frame_timesteps(key))
            variables = tuple(str(v).upper() for v in meta.get("common_variables", ()))
            out.append(
                TileDBDatasetInfo(
                    dataset_key=key,
                    zone_type=str(meta.get("zone") or "0_Fluid"),
                    timesteps=steps,
                    variables=variables,
                )
            )
        except Exception:
            continue
    return out
