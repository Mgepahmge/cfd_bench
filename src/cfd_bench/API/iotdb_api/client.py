from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.cfd_nodal_projection import (
    NodeCellCSR,
    build_node_cell_csr,
    point_frame_extrema_from_cell_values,
)
from cfd_bench.core.context import MeshContext
from cfd_bench.core.observability import timed_stage
from cfd_bench.core.runtime_mesh import RuntimeMeshData
from cfd_bench.core.types import LiteMesh, LitePolyData
from cfd_bench.infra.iotdb.config import IoTDBConfig
from cfd_bench.infra.iotdb.mesh_runtime import MeshRuntime
from cfd_bench.infra.iotdb.repository import IoTDBRepository
from cfd_bench.mesh_ops import (
    bboxes_for_cell_ids,
    cells_in_coordinate_range,
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
        self._surface_cells_normals_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._cfd_node_cell_csr: Optional[NodeCellCSR] = None

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
        self._cfd_node_cell_csr = None

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
        self._surface_cells_normals_cache = None
        self._cfd_node_cell_csr = None
        return ctx

    def _require_ctx(self) -> MeshContext:
        if self.ctx is None:
            raise RuntimeError("请先调用 connect(...) 初始化上下文")
        return self.ctx

    def get_cell_count(self) -> int:
        ctx = self._require_ctx()
        try:
            return int(self.repo.fetch_mesh_meta(ctx.dataset_key, ctx.zone).get("cell_count", 0))
        except Exception:
            return len(self.runtime.ensure_cells(ctx.dataset_key, ctx.zone).cells)

    def get_mesh_bounds(self):
        ctx = self._require_ctx()
        try:
            meta = self.repo.fetch_mesh_meta(ctx.dataset_key, ctx.zone)
            vals = [
                meta.get("bbox_min_x"), meta.get("bbox_max_x"),
                meta.get("bbox_min_y"), meta.get("bbox_max_y"),
                meta.get("bbox_min_z"), meta.get("bbox_max_z"),
            ]
            if all(v is not None and np.isfinite(float(v)) for v in vals):
                return [float(v) for v in vals]
        except Exception:
            pass
        return None

    def point_query(self, cell_indexes: Sequence[int], attribute_name: str, step: Optional[int] = None) -> np.ndarray:
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        scalar_map = self.repo.fetch_cell_scalar_map(
            ctx.dataset_key, ts, attribute_name, cell_indexes, zone=ctx.zone
        )
        out = [scalar_map.get(int(cid), np.nan) for cid in cell_indexes]
        return np.array(out, dtype=np.float64)

    def velocity_query(self, cell_indexes: Sequence[int], step: Optional[int] = None) -> np.ndarray:
        """Fetch U/V/W in one backend round-trip for W4/W5/W7-style access."""
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        ids = [int(cid) for cid in cell_indexes]
        values = self.repo.fetch_velocity_map(ctx.dataset_key, ts, ids, zone=ctx.zone)
        return np.asarray([values.get(cid, (np.nan, np.nan, np.nan)) for cid in ids], dtype=np.float64)

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
        return cells_in_coordinate_range(data, lower_bound, upper_bound)

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
        ids = sorted(set(int(x) for x in cell_indexes if int(x) >= 0))
        cells_data = self.runtime.ensure_cells(ctx.dataset_key, ctx.zone)
        # W3 usually selects a small fraction of a large mesh.  Avoid loading
        # the complete node/connectivity tables just to extract that subset.
        if ids and len(ids) < max(1024, int(0.25 * max(1, len(cells_data.cells)))):
            cell_nodes = self.repo.fetch_cell_nodes_subset(ctx.dataset_key, ctx.zone, ids)
            node_ids = sorted({nid for row in cell_nodes.values() for nid in row})
            nodes = self.repo.fetch_nodes_subset(ctx.dataset_key, ctx.zone, node_ids)
            return LiteMesh(
                cell_ids=np.asarray([cid for cid in ids if cid in cell_nodes], dtype=np.int32),
                node_xyz=nodes,
                cell_nodes=cell_nodes,
                cell_bbox=bboxes_for_cell_ids(cells_data, ids),
            )
        data = self.runtime.ensure_cell_nodes(ctx.dataset_key, ctx.zone)
        return iotdb_extract_submesh(data, ids)

    def isosurface_extraction(self, variable_name: str, iso_value: float, step: Optional[int] = None) -> LitePolyData:
        ctx = self._require_ctx()
        data = self.runtime.ensure_cell_nodes(ctx.dataset_key, ctx.zone)
        ts = ctx.step if step is None else int(step)
        cell_ids = list(data.cells.keys())
        scalar_map = self.repo.fetch_cell_scalar_map(
            ctx.dataset_key, ts, variable_name, cell_ids, zone=ctx.zone
        )
        return iotdb_isosurface_extraction(data, scalar_map, float(iso_value))

    def isosurface_from_submesh(
        self, mesh: LiteMesh, variable_name: str, iso_value: float, step: Optional[int] = None
    ) -> LitePolyData:
        """Run W3 on the already selected submesh instead of the full mesh."""
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        cell_ids = [int(x) for x in mesh.cell_ids.tolist()]
        scalar_map = self.repo.fetch_cell_scalar_map(
            ctx.dataset_key, ts, variable_name, cell_ids, zone=ctx.zone
        )
        scoped = RuntimeMeshData(nodes=dict(mesh.node_xyz), cell_nodes=dict(mesh.cell_nodes))
        return iotdb_isosurface_extraction(scoped, scalar_map, float(iso_value))

    def surface_norm(self, mesh_handle=None) -> np.ndarray:
        # W6 database-geometry path.  Prefer persisted CFD boundary faces and
        # fall back to a topology-derived per-cell normal for H5/structural
        # meshes that do not have a separate boundary-face device.
        if mesh_handle is None:
            _, normals = self.surface_cells_and_normals()
            return normals
        # Fallback: compute from submesh (isosurface / other workloads)
        if isinstance(mesh_handle, LitePolyData):
            return iotdb_surface_norm(mesh_handle)
        if isinstance(mesh_handle, LiteMesh):
            return iotdb_surface_norm_from_mesh(mesh_handle)
        raise TypeError("IoTDB surface_norm requires LiteMesh or LitePolyData")

    @staticmethod
    def _normal_from_points(points: Sequence[Sequence[float]]) -> np.ndarray:
        """Return a deterministic unit normal for 1D/2D/3D element points.

        W6 only needs a stable vector to complete the integration workload.
        For true surface cells we use a geometric cross product.  Beam/line
        elements have no unique surface normal, so choose a reproducible
        perpendicular vector.  This intentionally favors workload
        availability over strict physical interpretation.
        """
        pts = [np.asarray(p, dtype=np.float64) for p in points]
        if len(pts) >= 3:
            p0 = pts[0]
            for i in range(1, len(pts) - 1):
                for j in range(i + 1, len(pts)):
                    n = np.cross(pts[i] - p0, pts[j] - p0)
                    length = float(np.linalg.norm(n))
                    if length > 1e-15:
                        return n / length
        if len(pts) >= 2:
            tangent = pts[1] - pts[0]
            length = float(np.linalg.norm(tangent))
            if length > 1e-15:
                tangent = tangent / length
                axes = np.eye(3, dtype=np.float64)
                axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
                n = np.cross(tangent, axis)
                nlen = float(np.linalg.norm(n))
                if nlen > 1e-15:
                    return n / nlen
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def surface_cells_and_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return cell ids aligned with normals for IoTDB-native W6.

        Legacy CFD datasets persist explicit ``boundary_faces``.  H5 datasets
        currently persist only mesh topology, so they use a generic per-cell
        geometry fallback.  The returned cell ids are explicit because
        boundary-face owners are not necessarily ``0..N-1``.
        """
        cached = getattr(self, "_surface_cells_normals_cache", None)
        if cached is not None:
            return cached
        ctx = self._require_ctx()
        boundary = self.repo.fetch_boundary_faces(ctx.dataset_key, ctx.zone)
        if boundary:
            arr = np.asarray(
                [(int(cid), float(nx), float(ny), float(nz), max(abs(float(area)), 1e-15))
                 for cid, _patch, nx, ny, nz, area, *_center in boundary],
                dtype=np.float64,
            )
            order = np.argsort(arr[:, 0], kind="stable")
            arr = arr[order]
            raw_ids = arr[:, 0].astype(np.int64)
            cell_ids64, starts = np.unique(raw_ids, return_index=True)
            weighted = arr[:, 1:4] * arr[:, 4:5]
            sums = np.add.reduceat(weighted, starts, axis=0)
            areas = np.add.reduceat(arr[:, 4], starts)
            normals = sums / np.maximum(areas[:, None], 1e-15)
            lengths = np.linalg.norm(normals, axis=1)
            good = lengths > 1e-15
            normals[good] /= lengths[good, None]
            normals[~good] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            result = (cell_ids64.astype(np.int32), np.asarray(normals, dtype=np.float64))
            self._surface_cells_normals_cache = result
            return result

        # ``ensure_cell_nodes`` loads nodes/connectivity only; it does not
        # populate ``RuntimeMeshData.cells``. W6 enumerates element ids here,
        # so explicitly load cells as well. Keep a connectivity-union fallback
        # so partially populated/legacy IoTDB meshes can still execute W6.
        data = self.runtime.ensure_cells(ctx.dataset_key, ctx.zone)
        data = self.runtime.ensure_cell_nodes(ctx.dataset_key, ctx.zone)
        base_ids = set(int(x) for x in np.asarray(data.all_cell_ids, dtype=np.int64).tolist()) if data.all_cell_ids.size else set(data.cells.keys())
        cell_ids = np.asarray(
            sorted(base_ids | set(data.cell_nodes.keys())),
            dtype=np.int32,
        )
        normals = []
        for cid in cell_ids:
            node_ids = data.cell_nodes.get(int(cid), [])
            points = [data.nodes[nid] for nid in node_ids if nid in data.nodes]
            normals.append(self._normal_from_points(points))
        if not normals:
            return cell_ids, np.zeros((0, 3), dtype=np.float64)
        result = (cell_ids, np.asarray(normals, dtype=np.float64))
        self._surface_cells_normals_cache = result
        return result

    def w6_zone_candidates(
        self, dataset_key: str, preferred_zone: Optional[str] = None, hull_hint: Optional[str] = None
    ):
        """Return legacy-CFD zones in physical-surface preference order."""
        meta = self.repo.cfd_dataset_metadata(dataset_key)
        zones = [str(z) for z in meta.get("zones", ())]
        if not zones:
            zones = [z for z in (hull_hint, preferred_zone) if z]
        ordered = []
        def add(zone):
            if zone and zone in zones and zone not in ordered:
                ordered.append(zone)
        add(hull_hint)
        for keyword in ("hull", "wall", "symmetry"):
            for zone in zones:
                if keyword in zone.lower():
                    add(zone)
        for zone in zones:
            if "fluid" not in zone.lower():
                add(zone)
        add(preferred_zone)
        for zone in zones:
            add(zone)
        return ordered

    def resolve_w6_scalar(self, candidates: Sequence[str] = ("P", "U", "V", "W", "K", "E")) -> str:
        """Choose an available cell scalar for W6, preferring pressure.

        Original CFD data normally resolves to ``P``.  Structural H5 data may
        have only displacement components, in which case W6 still runs using
        the first available component as a force-like scalar benchmark.
        """
        ctx = self._require_ctx()
        ordered = []
        for name in candidates:
            var = str(name).strip().upper()
            if var and var not in ordered:
                ordered.append(var)
        if self.repo.is_h5_dataset(ctx.dataset_key):
            try:
                meta = self.repo.h5_dataset_metadata(ctx.dataset_key)
                for name in meta.get("common_variables", ()):
                    var = str(name).strip().upper()
                    if var and var not in ordered:
                        ordered.append(var)
            except Exception:
                pass
        for var in ordered:
            try:
                path = self.repo.resolve_cell_var_path(
                    ctx.dataset_key, ctx.step, zone=ctx.zone, probe_var=var
                )
                rows = self.repo.query_rows(f"SELECT {var} FROM {path} LIMIT 1;")
                if rows:
                    return var
            except Exception:
                continue
        raise RuntimeError(
            f"No usable IoTDB cell scalar for W6: dataset={ctx.dataset_key} "
            f"step={ctx.step} zone={ctx.zone} candidates={ordered}"
        )

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
        # Q at ROI boundary cells needs neighbor velocities outside the ROI.
        # Expand one adjacency halo before fetching U/V/W.  For low-dimensional
        # structural meshes allow rank-deficient least-squares and emit Q=0
        # when a gradient cannot be estimated; this keeps W7 executable while
        # retaining the normal 3-D calculation for CFD meshes.
        needed = {int(cid) for cid in roi_ids}
        for cid in list(needed):
            needed.update(int(nb) for nb in data.adjacency.get(cid, []))
        vel_map = self.repo.fetch_velocity_map(
            ctx.dataset_key, ts, sorted(needed), zone=ctx.zone
        )
        cell_ids, qvals = compute_qcriterion_roi(
            data,
            roi_ids,
            vel_map,
            tau=tau,
            min_neighbors=1,
            fallback_zero=True,
        )
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
        if self.repo.is_h5_dataset(ctx.dataset_key):
            meta = self.repo.h5_dataset_metadata(ctx.dataset_key)
            variables = list(meta.get("variables", ()))
        else:
            meta = self.repo.cfd_dataset_metadata(ctx.dataset_key)
            variables = list(meta.get("variables", ())) or ["U", "V", "W", "P", "K", "E"]
        return self.repo.fetch_max_diffs(ctx.dataset_key, ts, variables)

    # Legacy CFD-only W9-W11 primitives.  H5 methods below remain separate so
    # structural source-label and genuine nodal-field behaviour is unchanged.
    def cfd_element_ids_in_coordinate_range(
        self, lower_bound: Sequence[float], upper_bound: Sequence[float]
    ) -> np.ndarray:
        ctx = self._require_ctx()
        ids = self.repo.fetch_cfd_element_ids_in_coordinate_range(
            ctx.dataset_key, ctx.zone, lower_bound, upper_bound
        )
        return np.asarray(ids, dtype=np.int64)

    def cfd_frame_statistics(
        self, attribute_name: Optional[str] = None, step: Optional[int] = None
    ):
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        return self.repo.fetch_cfd_frame_statistics(
            ctx.dataset_key, ctx.zone, ts, attribute_name=attribute_name
        )

    def cfd_variables(self) -> Tuple[str, ...]:
        ctx = self._require_ctx()
        meta = self.repo.cfd_dataset_metadata(ctx.dataset_key)
        return tuple(str(v).upper() for v in meta.get("variables", ()))

    def cfd_point_ids(self) -> range:
        ctx = self._require_ctx()
        meta = self.repo.cfd_dataset_metadata(ctx.dataset_key)
        count = int(meta.get("node_count", 0) or 0)
        if count <= 0:
            try:
                count = int(self.repo.fetch_mesh_meta(ctx.dataset_key, ctx.zone).get("node_count", 0))
            except Exception:
                count = 0
        return range(1, count + 1)

    def _ensure_cfd_node_cell_csr(self) -> NodeCellCSR:
        if self._cfd_node_cell_csr is not None:
            return self._cfd_node_cell_csr
        ctx = self._require_ctx()
        meta = self.repo.cfd_dataset_metadata(ctx.dataset_key)
        node_count = int(meta.get("node_count", 0) or 0)
        if node_count <= 0:
            try:
                node_count = int(self.repo.fetch_mesh_meta(ctx.dataset_key, ctx.zone).get("node_count", 0))
            except Exception:
                node_count = 0
        with timed_stage(
            "IoTDB W11",
            f"build runtime node-to-cell projection dataset={ctx.dataset_key} zone={ctx.zone}",
        ):
            cell_nodes = self.repo.fetch_cell_nodes(ctx.dataset_key, ctx.zone)
            if node_count <= 0 and cell_nodes:
                node_count = 1 + max(
                    (max(nodes) for nodes in cell_nodes.values() if nodes),
                    default=-1,
                )
            self._cfd_node_cell_csr = build_node_cell_csr(cell_nodes, node_count)
        return self._cfd_node_cell_csr

    def prepare_cfd_point_queries(self) -> None:
        self._ensure_cfd_node_cell_csr()

    def cfd_point_frame_extrema(self, point_ids: Sequence[int], attribute_name: str):
        ctx = self._require_ctx()
        csr = self._ensure_cfd_node_cell_csr()
        meta = self.repo.cfd_dataset_metadata(ctx.dataset_key)
        steps = [int(x) for x in meta.get("timesteps", ())]

        def fetch_values(ts: int, cell_ids: np.ndarray) -> np.ndarray:
            ids = [int(x) for x in np.asarray(cell_ids, dtype=np.int64).tolist()]
            values = self.repo.fetch_cell_scalar_map(
                ctx.dataset_key,
                int(ts),
                str(attribute_name).upper(),
                ids,
                zone=ctx.zone,
            )
            return np.asarray([values.get(cid, np.nan) for cid in ids], dtype=np.float64)

        return point_frame_extrema_from_cell_values(csr, point_ids, steps, fetch_values)

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

