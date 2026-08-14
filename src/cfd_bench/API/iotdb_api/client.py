from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.context import MeshContext
from cfd_bench.core.types import LiteMesh, LitePolyData
from cfd_bench.infra.iotdb.config import IoTDBConfig
from cfd_bench.infra.iotdb.mesh_runtime import MeshRuntime
from cfd_bench.infra.iotdb.repository import IoTDBRepository
from cfd_bench.mesh_ops import (
    compute_qcriterion_roi,
    iotdb_extract_submesh,
    iotdb_isosurface_extraction,
    iotdb_line_intersection,
    iotdb_plane_intersection,
    iotdb_point_intersection,
    iotdb_surface_norm,
    iotdb_surface_norm_from_mesh,
)


class IoTDBMeshClient(AbstractContextManager):
    """Unified IoTDB mesh client implementing MeshClient protocol."""

    def __init__(self, config: Optional[IoTDBConfig] = None):
        self.config = config or IoTDBConfig()
        self.repo = IoTDBRepository(self.config)
        self.runtime = MeshRuntime(self.repo)
        self.ctx: Optional[MeshContext] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def open(self):
        self.repo.open()

    def close(self):
        self.repo.close()
        self.runtime.clear()

    def connect(
        self,
        dataset_key: str,
        step: int,
        zone: str = "0_Fluid",
        fields: Sequence[str] = ("U", "V", "W", "P", "K", "E"),
        prefer_materialized: bool = True,
        **kwargs,
    ) -> MeshContext:
        self.open()
        ctx = MeshContext(dataset_key=dataset_key, step=int(step), zone=zone)
        try:
            p = self.repo.path_mesh_static(dataset_key, zone, "cells")
            rows = self.repo.query_rows(f"SELECT cx FROM {p} LIMIT 1;")
            if rows:
                ctx.available_caps.add("mesh_static")
            else:
                ctx.missing_caps.add("mesh_static")
        except Exception:
            ctx.missing_caps.add("mesh_static")

        probed = False
        for var in list(fields) + ["P", "U"]:
            v = str(var).strip()
            if not v:
                continue
            try:
                path = self.repo.resolve_cell_var_path(
                    dataset_key, int(step), zone=zone, probe_var=v
                )
                rows = self.repo.query_rows(f"SELECT {v} FROM {path} LIMIT 1;")
                if rows:
                    ctx.available_caps.add("cell_vars")
                    probed = True
                    break
            except Exception:
                continue
        if not probed:
            ctx.missing_caps.add("cell_vars")

        if prefer_materialized:
            try:
                sql = f"SELECT q FROM {self.repo.path_derived(dataset_key, int(step), 'cell_qcriterion')} LIMIT 1;"
                rows = self.repo.query_rows(sql)
                if rows:
                    ctx.available_caps.add("cell_qcriterion")
            except Exception:
                pass

        self.ctx = ctx
        return ctx

    def _require_ctx(self) -> MeshContext:
        if self.ctx is None:
            raise RuntimeError("请先调用 connect(...) 初始化上下文")
        return self.ctx

    def point_query(self, cell_indexes: Sequence[int], attribute_name: str, step: Optional[int] = None) -> np.ndarray:
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        scalar_map = self.repo.fetch_cell_scalar_map(
            ctx.dataset_key, ts, attribute_name, cell_indexes, zone=ctx.zone
        )
        out = [scalar_map.get(int(cid), np.nan) for cid in cell_indexes]
        return np.array(out, dtype=np.float64)

    def range_query_var(
        self, lower_bound: float, upper_bound: float, attribute_name: str, step: Optional[int] = None
    ) -> np.ndarray:
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        ids = self.repo.fetch_cell_ids_by_var_range(
            ctx.dataset_key, ts, attribute_name, lower_bound, upper_bound, zone=ctx.zone
        )
        return np.array(ids, dtype=np.int32)

    def range_query_coord(self, lower_bound: Sequence[float], upper_bound: Sequence[float]) -> np.ndarray:
        ctx = self._require_ctx()
        data = self.runtime.ensure_cells(ctx.dataset_key, ctx.zone)
        x0, y0, z0 = map(float, lower_bound)
        x1, y1, z1 = map(float, upper_bound)
        out = []
        for cid, bb in data.cell_bbox.items():
            if bb[0] >= x0 and bb[1] <= x1 and bb[2] >= y0 and bb[3] <= y1 and bb[4] >= z0 and bb[5] <= z1:
                out.append(int(cid))
        return np.array(out, dtype=np.int32)

    def point_intersection(self, points: np.ndarray) -> np.ndarray:
        ctx = self._require_ctx()
        data = self.runtime.ensure_cells(ctx.dataset_key, ctx.zone)
        return iotdb_point_intersection(data, points, eps=self.config.bbox_eps)

    def line_intersection(self, line_start: Sequence[float], line_end: Sequence[float]) -> np.ndarray:
        ctx = self._require_ctx()
        data = self.runtime.ensure_cells(ctx.dataset_key, ctx.zone)
        return iotdb_line_intersection(data, line_start, line_end, eps=self.config.line_eps)

    def plane_intersection(self, plane_origin: Sequence[float], plane_norm: Sequence[float]) -> np.ndarray:
        ctx = self._require_ctx()
        data = self.runtime.ensure_cells(ctx.dataset_key, ctx.zone)
        return iotdb_plane_intersection(data, plane_origin, plane_norm, eps=self.config.plane_eps)

    def extract_submesh(self, cell_indexes: Sequence[int], mesh_handle=None) -> LiteMesh:
        ctx = self._require_ctx()
        data = self.runtime.ensure_cell_nodes(ctx.dataset_key, ctx.zone)
        return iotdb_extract_submesh(data, cell_indexes)

    def isosurface_extraction(self, variable_name: str, iso_value: float, step: Optional[int] = None) -> LitePolyData:
        ctx = self._require_ctx()
        data = self.runtime.ensure_cell_nodes(ctx.dataset_key, ctx.zone)
        ts = ctx.step if step is None else int(step)
        cell_ids = list(data.cells.keys())
        scalar_map = self.repo.fetch_cell_scalar_map(
            ctx.dataset_key, ts, variable_name, cell_ids, zone=ctx.zone
        )
        return iotdb_isosurface_extraction(data, scalar_map, float(iso_value))

    def surface_norm(self, mesh_handle=None) -> np.ndarray:
        # W6 lightweight path: return normals from pre-stored boundary_faces (algorithm step 1)
        if mesh_handle is None:
            ctx = self._require_ctx()
            bf = self.repo.fetch_boundary_faces(ctx.dataset_key, ctx.zone)
            if not bf:
                raise RuntimeError(f"No boundary_faces data for {ctx.dataset_key} zone={ctx.zone}")
            # Aggregate per-cell: area-weighted average normal (matches PG fetch_boundary_normals logic)
            cell_norms: dict = {}
            for cid, _patch, nx, ny, nz, area, *_cx_cy_cz in bf:
                area_f = float(area)
                prev = cell_norms.get(int(cid))
                if prev is None:
                    cell_norms[int(cid)] = [float(nx) * area_f, float(ny) * area_f, float(nz) * area_f, area_f]
                else:
                    prev[0] += float(nx) * area_f
                    prev[1] += float(ny) * area_f
                    prev[2] += float(nz) * area_f
                    prev[3] += area_f
            sorted_cids = sorted(cell_norms.keys())
            norms = np.array(
                [[cell_norms[c][0] / max(cell_norms[c][3], 1e-15),
                  cell_norms[c][1] / max(cell_norms[c][3], 1e-15),
                  cell_norms[c][2] / max(cell_norms[c][3], 1e-15)]
                 for c in sorted_cids],
                dtype=np.float64,
            )
            # Normalize each row to unit length
            lengths = np.linalg.norm(norms, axis=1, keepdims=True)
            lengths = np.maximum(lengths, 1e-15)
            norms = norms / lengths
            return norms
        # Fallback: compute from submesh (isosurface / other workloads)
        if isinstance(mesh_handle, LitePolyData):
            return iotdb_surface_norm(mesh_handle)
        if isinstance(mesh_handle, LiteMesh):
            return iotdb_surface_norm_from_mesh(mesh_handle)
        raise TypeError("IoTDB surface_norm requires LiteMesh or LitePolyData")

    def compute_qcriterion_roi(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        tau: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        data = self.runtime.ensure_adjacency(ctx.dataset_key, ctx.zone)
        roi_ids = self.range_query_coord(lower_bound, upper_bound)
        vel_map = self.repo.fetch_velocity_map(ctx.dataset_key, ts, roi_ids, zone=ctx.zone)
        cell_ids, qvals = compute_qcriterion_roi(data, roi_ids, vel_map, tau=tau)
        return np.array(cell_ids, dtype=np.int32), np.array(qvals, dtype=np.float64)

    def var_value_range(
        self, attribute_name: str, step: Optional[int] = None
    ) -> Tuple[float, float]:
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        return self.repo.fetch_var_value_range(
            ctx.dataset_key, ts, str(attribute_name).upper(), zone=ctx.zone
        )

    def get_max_diffs(self, step: Optional[int] = None):
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        meta = self.repo.h5_dataset_metadata(ctx.dataset_key) if self.repo.is_h5_dataset(ctx.dataset_key) else {}
        variables = list(meta.get("variables", ()))
        return self.repo.fetch_max_diffs(ctx.dataset_key, ts, variables)

    def is_h5_dataset(self) -> bool:
        ctx = self._require_ctx()
        return self.repo.is_h5_dataset(ctx.dataset_key)

    def h5_element_ids_in_coordinate_range(
        self, lower_bound: Sequence[float], upper_bound: Sequence[float]
    ) -> np.ndarray:
        ctx = self._require_ctx()
        ids = self.repo.fetch_h5_element_ids_in_coordinate_range(
            ctx.dataset_key, ctx.zone, lower_bound, upper_bound
        )
        return np.asarray(ids, dtype=np.int64)

    def frame_statistics(
        self, attribute_name: Optional[str] = None, step: Optional[int] = None
    ):
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        return self.repo.fetch_frame_statistics(
            ctx.dataset_key, ctx.zone, ts, attribute_name=attribute_name
        )

    def h5_nodal_variables(self) -> Tuple[str, ...]:
        ctx = self._require_ctx()
        meta = self.repo.h5_dataset_metadata(ctx.dataset_key)
        return tuple(str(v).upper() for v in meta.get("common_nodal_variables", ()))

    def h5_point_ids(self) -> np.ndarray:
        ctx = self._require_ctx()
        return np.asarray(
            self.repo.fetch_h5_point_ids(ctx.dataset_key, ctx.zone), dtype=np.int64
        )

    def h5_point_frame_extrema(
        self, point_ids: Sequence[int], attribute_name: str
    ):
        ctx = self._require_ctx()
        return self.repo.fetch_h5_point_frame_extrema(
            ctx.dataset_key, ctx.zone, point_ids, str(attribute_name).upper()
        )

