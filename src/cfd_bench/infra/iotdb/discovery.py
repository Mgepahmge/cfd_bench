"""Runtime metadata discovery for H5 datasets stored in IoTDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from cfd_bench.infra.iotdb.config import IoTDBConfig
from cfd_bench.infra.iotdb.repository import IoTDBRepository


@dataclass(frozen=True)
class IoTDBDatasetInfo:
    dataset_key: str
    zone_type: str
    timesteps: Tuple[int, ...]
    variables: Tuple[str, ...]


def discover_iotdb_datasets(
    selected_datasets: Iterable[str],
    *,
    config: IoTDBConfig | None = None,
) -> List[IoTDBDatasetInfo]:
    repo = IoTDBRepository(config or IoTDBConfig())
    out: List[IoTDBDatasetInfo] = []
    repo.open()
    try:
        for dataset in selected_datasets:
            key = str(dataset)
            try:
                meta = repo.h5_dataset_metadata(key)
                if not meta.get("is_h5"):
                    continue
                steps = tuple(repo.h5_frame_timesteps(key))
                variables = tuple(str(v).upper() for v in meta.get("common_variables", ()))
                out.append(
                    IoTDBDatasetInfo(
                        dataset_key=key,
                        zone_type=str(meta.get("zone") or "0_Fluid"),
                        timesteps=steps,
                        variables=variables,
                    )
                )
            except Exception:
                continue
    finally:
        repo.close()
    return out
