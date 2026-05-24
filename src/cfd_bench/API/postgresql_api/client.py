from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.context import DatasetKey, MeshContext, parse_dataset_key
from cfd_bench.core.types import LiteMesh, LitePolyData


@dataclass
class PostgreSQLConfig:
    db_name: str = "cae_data"
    db_user: str = "postgres"
    db_password: str = "123456"
    db_host: str = "localhost"
    db_port: str = "5432"


class PostgreSQLMeshClient:
    """PostgreSQL mesh client — PostGIS backend + cae_simulation_data via workloads helper."""

    def __init__(self, config: Optional[PostgreSQLConfig] = None, **kwargs):
        self.config = config or PostgreSQLConfig()
        self._kwargs = kwargs
        self._inner = None
        self.ctx: Optional[MeshContext] = None
        self._key: Optional[DatasetKey] = None

    def _ensure_inner(self):
        if self._inner is None:
            from cfd_bench.infra.postgresql.client import LegacyPGMeshBackend

            key = self._key or DatasetKey("JBC", "615k")
            self._inner = LegacyPGMeshBackend(
                ship_type=key.ship,
                scale=key.scale,
                zone_type=key.zone,
                timestep=key.step,
                db_name=self.config.db_name,
                db_user=self.config.db_user,
                db_password=self.config.db_password,
                db_host=self.config.db_host,
                db_port=self.config.db_port,
            )

    def close(self):
        if self._inner is not None:
            self._inner.close()
            self._inner = None

    def connect(
        self,
        dataset_key: str,
        step: int,
        zone: str = "0_Fluid",
        **kwargs,
    ) -> MeshContext:
        self._key = parse_dataset_key(dataset_key, zone=zone, step=step)
        self._inner = None
        self._ensure_inner()
        ctx = MeshContext(dataset_key=self._key.dataset_key, step=int(step), zone=zone)
        ctx.available_caps.update({"cell_vars"})
        self.ctx = ctx
        return ctx

    def _require_ctx(self) -> MeshContext:
        if self.ctx is None:
            raise RuntimeError("请先调用 connect(...) 初始化上下文")
        return self.ctx

    def point_query(self, cell_indexes: Sequence[int], attribute_name: str, step: Optional[int] = None) -> np.ndarray:
        self._ensure_inner()
        return self._inner.point_query(None, cell_indexes, attribute_name, timestep=step)

    def range_query_var(
        self, lower_bound: float, upper_bound: float, attribute_name: str, step: Optional[int] = None
    ) -> np.ndarray:
        self._ensure_inner()
        return self._inner.range_query_var(None, lower_bound, upper_bound, attribute_name, timestep=step)

    def range_query_coord(self, lower_bound: Sequence[float], upper_bound: Sequence[float]) -> np.ndarray:
        raise NotImplementedError("PostgreSQL range_query_coord requires modern mesh_static tables")

    def point_intersection(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError("PostgreSQL point_intersection requires modern mesh_static tables")

    def line_intersection(self, line_start: Sequence[float], line_end: Sequence[float]) -> np.ndarray:
        self._ensure_inner()
        return self._inner.vtk_line_intersection(None, line_start, line_end)

    def plane_intersection(self, plane_origin: Sequence[float], plane_norm: Sequence[float]) -> np.ndarray:
        self._ensure_inner()
        return self._inner.vtk_plane_intersection(None, plane_origin, plane_norm)

    def extract_submesh(self, cell_indexes: Sequence[int], mesh_handle=None) -> LiteMesh:
        raise NotImplementedError("PostgreSQL extract_submesh requires modern mesh_static tables")

    def isosurface_extraction(self, variable_name: str, iso_value: float, step: Optional[int] = None) -> LitePolyData:
        raise NotImplementedError("PostgreSQL isosurface_extraction not yet implemented")

    def surface_norm(self, mesh_handle=None) -> np.ndarray:
        raise NotImplementedError("PostgreSQL surface_norm requires modern mesh_static tables")

    def compute_qcriterion_roi(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        tau: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("PostgreSQL compute_qcriterion_roi not yet implemented")
