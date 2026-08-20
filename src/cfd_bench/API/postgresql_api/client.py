from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.cfd_nodal_projection import (
    NodeCellCSR,
    build_node_cell_csr_from_incidence,
    point_frame_extrema_from_cell_values,
)
from cfd_bench.core.context import DatasetKey, MeshContext, parse_dataset_key
from cfd_bench.core.observability import timed_stage
from cfd_bench.core.types import LiteMesh, LitePolyData
from cfd_bench.infra.postgresql.client import PGMeshBackend
from cfd_bench.infra.postgresql.config import PostgreSQLConfig
from cfd_bench.infra.postgresql.mesh_runtime import PGMeshRuntime
from cfd_bench.infra.postgresql import spatial as pg_spatial


class PostgreSQLMeshClient:
    """PostgreSQL mesh client with PostGIS spatial acceleration."""

    def __init__(self, config: Optional[PostgreSQLConfig] = None, **kwargs):
        self.config = config or PostgreSQLConfig()
        self._kwargs = kwargs
        self._inner: Optional[PGMeshBackend] = None
        self.runtime: Optional[PGMeshRuntime] = None
        self.ctx: Optional[MeshContext] = None
        self._key: Optional[DatasetKey] = None
        self._centroids_cache: Optional[Dict] = None
        self._var_range_cache: Dict[Tuple[int, str], Tuple[float, float]] = {}
        self._surface_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._qc_centroids_cache: Optional[Dict] = None
        self._qc_neighbors_cache: Optional[Dict] = None
        self._cfd_node_cell_csr: Optional[NodeCellCSR] = None

    def _ensure_inner(self):
        if self._inner is None:
            key = self._key or DatasetKey("JBC", "615k")
            self._inner = PGMeshBackend(
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
            self.runtime = PGMeshRuntime(self._inner.conn, key.ship, key.scale, key.zone)

    def close(self):
        if self._inner is not None:
            self._inner.close()
            self._inner = None
        self.runtime = None
        self._centroids_cache = None
        self._var_range_cache.clear()
        self._surface_cache = None
        self._qc_centroids_cache = None
        self._qc_neighbors_cache = None
        self._cfd_node_cell_csr = None

    def connect(
        self,
        dataset_key: str,
        step: int,
        zone: str = "0_Fluid",
        **kwargs,
    ) -> MeshContext:
        self._key = parse_dataset_key(dataset_key, zone=zone, step=step)
        self._inner = None
        self._centroids_cache = None
        self._var_range_cache.clear()
        self._surface_cache = None
        self._qc_centroids_cache = None
        self._qc_neighbors_cache = None
        self._cfd_node_cell_csr = None
        self._ensure_inner()
        ctx = MeshContext(dataset_key=self._key.dataset_key, step=int(step), zone=zone)
        conn = self._inner.conn
        st, sc, zt = self._key.ship, self._key.scale, self._key.zone
        if pg_spatial.fetch_cell_count(conn, st, sc, zt) > 0:
            ctx.available_caps.add("mesh_static")
        else:
            ctx.missing_caps.add("mesh_static")
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT 1 FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s AND timestep=%s
                LIMIT 1
                """,
                (st, sc, zt, int(step)),
            )
            if cur.fetchone():
                ctx.available_caps.add("cell_vars")
            else:
                ctx.missing_caps.add("cell_vars")
        finally:
            cur.close()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1 FROM cell_geom_full
                WHERE ship_type=%s AND scale=%s AND zone_type=%s LIMIT 1
                """,
                (st, sc, zt),
            )
            if cur.fetchone():
                ctx.available_caps.add("postgis_spatial")
        except Exception:
            pass
        finally:
            cur.close()
        self.ctx = ctx
        return ctx

    def _require_ctx(self) -> MeshContext:
        if self.ctx is None:
            raise RuntimeError("请先调用 connect(...) 初始化上下文")
        return self.ctx

    def _sync_timestep(self, step: Optional[int]):
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        if self._inner is not None:
            self._inner.timestep = int(ts)

    def get_mesh_bounds(self):
        self._ensure_inner()
        key = self._key
        return pg_spatial.fetch_mesh_bounds(self._inner.conn, key.ship, key.scale, key.zone)

    def get_cell_count(self) -> int:
        self._ensure_inner()
        key = self._key
        return pg_spatial.fetch_cell_count(self._inner.conn, key.ship, key.scale, key.zone)

    def get_max_diffs(self, step: Optional[int] = None) -> Dict[str, float]:
        """Return W3 search widths from PostgreSQL, never from a sidecar path.

        New H5 ingests materialize ``benchmark_max_diff``.  For databases
        created by older versions, the method computes the same values from
        ``cell_scalar`` + ``cell_adjacency`` and finally falls back to the
        variable range for meshes without usable adjacency.
        """
        self._ensure_inner()
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        key = self._key
        conn = self._inner.conn
        cur = conn.cursor()
        try:
            cur.execute("SELECT to_regclass('public.benchmark_max_diff')")
            if cur.fetchone()[0] is not None:
                cur.execute(
                    """
                    SELECT var, max_diff
                    FROM benchmark_max_diff
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s AND timestep=%s
                    """,
                    (key.ship, key.scale, key.zone, ts),
                )
                rows = cur.fetchall()
                if rows:
                    return {str(var).upper(): float(value) for var, value in rows}

            cur.execute(
                """
                SELECT a.var, MAX(ABS(a.value - b.value))
                FROM cell_scalar a
                JOIN cell_adjacency adj
                  ON adj.ship_type=a.ship_type AND adj.scale=a.scale
                 AND adj.zone_type=a.zone_type AND adj.cell_id=a.cell_id
                CROSS JOIN LATERAL unnest(adj.neighbor_ids) AS n(cell_id)
                JOIN cell_scalar b
                  ON b.ship_type=a.ship_type AND b.scale=a.scale
                 AND b.zone_type=a.zone_type AND b.timestep=a.timestep
                 AND b.var=a.var AND b.cell_id=n.cell_id
                WHERE a.ship_type=%s AND a.scale=%s AND a.zone_type=%s
                  AND a.timestep=%s
                GROUP BY a.var
                """,
                (key.ship, key.scale, key.zone, ts),
            )
            result = {
                str(var).upper(): float(value)
                for var, value in cur.fetchall()
                if value is not None
            }
            if result:
                return result

            cur.execute(
                """
                SELECT var, MAX(value) - MIN(value)
                FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s AND timestep=%s
                GROUP BY var
                """,
                (key.ship, key.scale, key.zone, ts),
            )
            return {
                str(var).upper(): float(value or 0.0)
                for var, value in cur.fetchall()
            }
        finally:
            cur.close()

    def var_value_range(self, attribute_name: str, step: Optional[int] = None) -> Tuple[float, float]:
        self._ensure_inner()
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        cache_key = (ts, str(attribute_name).upper())
        if cache_key not in self._var_range_cache:
            key = self._key
            self._var_range_cache[cache_key] = pg_spatial.fetch_var_value_range(
                self._inner.conn, key.ship, key.scale, key.zone, ts, attribute_name
            )
        return self._var_range_cache[cache_key]

    def point_query(self, cell_indexes: Sequence[int], attribute_name: str, step: Optional[int] = None) -> np.ndarray:
        self._ensure_inner()
        self._sync_timestep(step)
        return self._inner.point_query(None, cell_indexes, attribute_name, timestep=step)

    def bulk_point_query(
        self, cell_indexes: Sequence[int], attribute_name: str, step: Optional[int] = None
    ) -> np.ndarray:
        """Fast ordered scalar read for large contiguous CFD selections.

        W6 usually asks for every cell in a surface zone. Passing tens of
        thousands of IDs through ``ANY(array)`` adds avoidable parameter and
        planning overhead; a contiguous primary-key range scan is equivalent
        and substantially cheaper. Sparse selections keep the frozen point
        query path.
        """
        self._ensure_inner()
        self._sync_timestep(step)
        ids = np.asarray(list(cell_indexes), dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.zeros((0,), dtype=np.float64)
        unique_ids = np.unique(ids)
        contiguous = (
            unique_ids.size == ids.size
            and int(unique_ids[-1]) - int(unique_ids[0]) + 1 == unique_ids.size
        )
        if not contiguous:
            return self.point_query(ids, attribute_name, step=step)

        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT cell_id, value
                FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND timestep=%s AND var=%s
                  AND cell_id BETWEEN %s AND %s
                ORDER BY cell_id
                """,
                (
                    key.ship, key.scale, key.zone, ts,
                    str(attribute_name).upper(),
                    int(unique_ids[0]), int(unique_ids[-1]),
                ),
            )
            rows = cur.fetchall()
        finally:
            cur.close()
        if len(rows) != ids.size:
            return self.point_query(ids, attribute_name, step=step)
        row_ids = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
        if not np.array_equal(row_ids, ids):
            mapping = {int(cid): float(value) for cid, value in rows}
            return np.asarray([mapping.get(int(cid), np.nan) for cid in ids], dtype=np.float64)
        return np.asarray([float(r[1]) for r in rows], dtype=np.float64)

    def velocity_query(self, cell_indexes: Sequence[int], step: Optional[int] = None) -> np.ndarray:
        """Fetch U/V/W for cells with one SQL query instead of three point queries."""
        self._ensure_inner()
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        ids = [int(cid) for cid in cell_indexes]
        if not ids:
            return np.zeros((0, 3), dtype=np.float64)
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT cell_id,
                       MAX(value) FILTER (WHERE var='U') AS u,
                       MAX(value) FILTER (WHERE var='V') AS v,
                       MAX(value) FILTER (WHERE var='W') AS w
                FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND timestep=%s AND var IN ('U','V','W')
                  AND cell_id = ANY(%s)
                GROUP BY cell_id
                """,
                (key.ship, key.scale, key.zone, ts, ids),
            )
            rows = {int(cid): (float(u), float(v), float(w)) for cid, u, v, w in cur.fetchall() if u is not None and v is not None and w is not None}
        finally:
            cur.close()
        return np.asarray([rows.get(cid, (np.nan, np.nan, np.nan)) for cid in ids], dtype=np.float64)

    def range_query_var(
        self, lower_bound: float, upper_bound: float, attribute_name: str, step: Optional[int] = None
    ) -> np.ndarray:
        self._ensure_inner()
        self._sync_timestep(step)
        return self._inner.range_query_var(None, lower_bound, upper_bound, attribute_name, timestep=step)

    def range_query_coord(self, lower_bound: Sequence[float], upper_bound: Sequence[float]) -> np.ndarray:
        self._ensure_inner()
        key = self._key
        return pg_spatial.range_query_coord(
            self._inner.conn, key.ship, key.scale, key.zone, lower_bound, upper_bound
        )

    def point_intersection(self, points: np.ndarray) -> np.ndarray:
        self._ensure_inner()
        key = self._key
        # Keep candidate ranking inside PostgreSQL.  Loading the full centroid
        # table into Python made a single point workload scale with total mesh
        # size even though point_locator_grid already narrows the candidates.
        return pg_spatial.point_intersection(
            self._inner.conn, key.ship, key.scale, key.zone, points, None
        )

    def line_intersection(self, line_start: Sequence[float], line_end: Sequence[float]) -> np.ndarray:
        self._ensure_inner()
        return self._inner.vtk_line_intersection(None, line_start, line_end)

    def plane_intersection(self, plane_origin: Sequence[float], plane_norm: Sequence[float]) -> np.ndarray:
        self._ensure_inner()
        return self._inner.vtk_plane_intersection(None, plane_origin, plane_norm)

    def extract_submesh(self, cell_indexes: Sequence[int], mesh_handle=None) -> LiteMesh:
        self._ensure_inner()
        ids = sorted(set(int(x) for x in cell_indexes if int(x) >= 0))
        if not ids:
            return LiteMesh(
                cell_ids=np.zeros((0,), dtype=np.int32), node_xyz={}, cell_nodes={}, cell_bbox={}
            )
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """SELECT cell_id, node_ids FROM cell_nodes
                   WHERE ship_type=%s AND scale=%s AND zone_type=%s AND cell_id = ANY(%s)""",
                (key.ship, key.scale, key.zone, ids),
            )
            cell_nodes = {int(cid): [int(n) for n in (nids or [])] for cid, nids in cur.fetchall()}
            node_ids = sorted({nid for row in cell_nodes.values() for nid in row})
            cur.execute(
                """SELECT node_id,x,y,z FROM node_coordinates
                   WHERE ship_type=%s AND scale=%s AND zone_type=%s AND node_id = ANY(%s)""",
                (key.ship, key.scale, key.zone, node_ids),
            )
            nodes = {int(nid): (float(x), float(y), float(z)) for nid, x, y, z in cur.fetchall()}
        finally:
            cur.close()
        return LiteMesh(
            cell_ids=np.asarray(sorted(cell_nodes), dtype=np.int32),
            node_xyz=nodes,
            cell_nodes=cell_nodes,
            cell_bbox={},
        )

    def isosurface_extraction(self, variable_name: str, iso_value: float, step: Optional[int] = None) -> LitePolyData:
        self._ensure_inner()
        self._sync_timestep(step)
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        data = self.runtime.ensure_cell_nodes()
        cell_ids = list(data.cells.keys())
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT cell_id, value FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND timestep=%s AND var=%s AND cell_id = ANY(%s)
                """,
                (
                    self._key.ship,
                    self._key.scale,
                    self._key.zone,
                    ts,
                    str(variable_name).upper(),
                    cell_ids,
                ),
            )
            scalar_map = {int(cid): float(v) for cid, v in cur.fetchall()}
        finally:
            cur.close()
        return self.runtime.isosurface(scalar_map, float(iso_value))

    def isosurface_from_submesh(
        self, mesh: LiteMesh, variable_name: str, iso_value: float, step: Optional[int] = None
    ) -> LitePolyData:
        self._ensure_inner()
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        ids = [int(x) for x in mesh.cell_ids.tolist()]
        if not ids:
            return self.runtime.isosurface({}, float(iso_value), mesh=mesh)
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """SELECT cell_id,value FROM cell_scalar
                   WHERE ship_type=%s AND scale=%s AND zone_type=%s
                     AND timestep=%s AND var=%s AND cell_id = ANY(%s)""",
                (self._key.ship, self._key.scale, self._key.zone, ts, str(variable_name).upper(), ids),
            )
            scalar_map = {int(cid): float(v) for cid, v in cur.fetchall()}
        finally:
            cur.close()
        return self.runtime.isosurface(scalar_map, float(iso_value), mesh=mesh)

    # ------------------------------------------------------------------
    # Legacy CFD-only W9-W11 primitives.  The H5 methods below are kept
    # separate so the already-validated structural path retains its exact
    # source-label / genuine-nodal semantics.
    # ------------------------------------------------------------------
    def cfd_element_ids_in_coordinate_range(
        self, lower_bound: Sequence[float], upper_bound: Sequence[float]
    ) -> np.ndarray:
        """Return implicit one-based Tecplot element IDs by centroid range."""
        self._ensure_inner()
        key = self._key
        lo = np.asarray(lower_bound, dtype=np.float64).reshape(3)
        hi = np.asarray(upper_bound, dtype=np.float64).reshape(3)
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT cell_id + 1
                FROM cell_centroid
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND x BETWEEN %s AND %s
                  AND y BETWEEN %s AND %s
                  AND z BETWEEN %s AND %s
                ORDER BY cell_id
                """,
                (
                    key.ship, key.scale, key.zone,
                    float(lo[0]), float(hi[0]),
                    float(lo[1]), float(hi[1]),
                    float(lo[2]), float(hi[2]),
                ),
            )
            return np.asarray([int(row[0]) for row in cur.fetchall()], dtype=np.int64)
        finally:
            cur.close()

    def cfd_frame_statistics(
        self, attribute_name: Optional[str] = None, step: Optional[int] = None
    ) -> Dict[str, Dict[str, object]]:
        """Cell-centered statistics for one legacy CFD frame."""
        self._ensure_inner()
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            params = [key.ship, key.scale, key.zone, ts]
            clause = ""
            if attribute_name is not None:
                clause = " AND var=%s"
                params.append(str(attribute_name).upper())
            cur.execute(
                f"""
                SELECT var, COUNT(*) AS n,
                       MIN(value) AS vmin, MAX(value) AS vmax,
                       AVG(value) AS mean, STDDEV_POP(value) AS stddev
                FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND timestep=%s{clause}
                GROUP BY var
                ORDER BY var
                """,
                tuple(params),
            )
            rows = cur.fetchall()
            if not rows:
                suffix = (
                    f" variable={str(attribute_name).upper()}"
                    if attribute_name is not None
                    else ""
                )
                raise ValueError(f"no CFD values for frame={ts}{suffix}")
            return {
                str(var).upper(): {
                    "position": "cell",
                    "count": int(count),
                    "min": float(vmin),
                    "max": float(vmax),
                    "mean": float(mean),
                    "stddev": float(stddev or 0.0),
                }
                for var, count, vmin, vmax, mean, stddev in rows
            }
        finally:
            cur.close()

    def cfd_variables(self) -> Tuple[str, ...]:
        """Cell variables present in every stored CFD timestep for this zone."""
        self._ensure_inner()
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                WITH frame_count AS (
                    SELECT COUNT(DISTINCT timestep) AS n
                    FROM cell_scalar
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                )
                SELECT c.var
                FROM cell_scalar c CROSS JOIN frame_count f
                WHERE c.ship_type=%s AND c.scale=%s AND c.zone_type=%s
                  AND f.n > 0
                GROUP BY c.var, f.n
                HAVING COUNT(DISTINCT c.timestep) = f.n
                ORDER BY c.var
                """,
                (
                    key.ship, key.scale, key.zone,
                    key.ship, key.scale, key.zone,
                ),
            )
            return tuple(str(row[0]).upper() for row in cur.fetchall())
        finally:
            cur.close()

    def cfd_point_ids(self) -> range:
        """Return implicit one-based Tecplot node IDs."""
        self._ensure_inner()
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT node_count FROM mesh_metadata
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                """,
                (key.ship, key.scale, key.zone),
            )
            row = cur.fetchone()
            count = int(row[0]) if row and row[0] is not None else 0
            if count <= 0:
                cur.execute(
                    """
                    SELECT COALESCE(MAX(node_id) + 1, 0)
                    FROM node_coordinates
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    """,
                    (key.ship, key.scale, key.zone),
                )
                count = int(cur.fetchone()[0] or 0)
            return range(1, count + 1)
        finally:
            cur.close()

    def _ensure_cfd_node_cell_csr(self) -> NodeCellCSR:
        if self._cfd_node_cell_csr is not None:
            return self._cfd_node_cell_csr
        self._ensure_inner()
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            with timed_stage(
                "PostgreSQL W11",
                f"build runtime node-to-cell projection dataset={key.dataset_key} zone={key.zone}",
            ):
                cur.execute(
                    """
                    SELECT node_count FROM mesh_metadata
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    """,
                    (key.ship, key.scale, key.zone),
                )
                row = cur.fetchone()
                node_count = int(row[0]) if row and row[0] is not None else 0
                cur.execute(
                    """
                    SELECT COALESCE(SUM(cardinality(node_ids)), 0)
                    FROM cell_nodes
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    """,
                    (key.ship, key.scale, key.zone),
                )
                incidence_count = int(cur.fetchone()[0] or 0)
                dense_nodes = np.empty((incidence_count,), dtype=np.int64)
                dense_cells = np.empty((incidence_count,), dtype=np.int32)
                cur.execute(
                    """
                    SELECT cell_id, node_ids
                    FROM cell_nodes
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    ORDER BY cell_id
                    """,
                    (key.ship, key.scale, key.zone),
                )
                pos = 0
                max_node_id = -1
                while True:
                    rows = cur.fetchmany(5000)
                    if not rows:
                        break
                    for cid, node_ids in rows:
                        ids = np.asarray(node_ids or (), dtype=np.int64).reshape(-1)
                        ids = ids[ids >= 0]
                        n = int(ids.size)
                        if n <= 0:
                            continue
                        end = pos + n
                        if end > dense_nodes.size:
                            # Defensive compatibility for malformed/old rows
                            # where cardinality changed during the read.
                            grow = max(end, int(dense_nodes.size * 1.25) + 1)
                            dense_nodes.resize((grow,), refcheck=False)
                            dense_cells.resize((grow,), refcheck=False)
                        dense_nodes[pos:end] = ids
                        dense_cells[pos:end] = int(cid)
                        max_node_id = max(max_node_id, int(np.max(ids)))
                        pos = end
                dense_nodes = dense_nodes[:pos]
                dense_cells = dense_cells[:pos]
                if node_count <= 0:
                    node_count = max_node_id + 1
                self._cfd_node_cell_csr = build_node_cell_csr_from_incidence(
                    dense_nodes, dense_cells, node_count
                )
        finally:
            cur.close()
        return self._cfd_node_cell_csr

    def prepare_cfd_point_queries(self) -> None:
        """Prebuild W11's runtime-only node/cell projection outside the timer."""
        self._ensure_cfd_node_cell_csr()

    def cfd_point_frame_extrema(
        self, point_ids: Sequence[int], attribute_name: str
    ) -> Dict[int, Tuple[float, float]]:
        """Per-node extrema from runtime cell->node averaging across CFD frames."""
        self._ensure_inner()
        key = self._key
        csr = self._ensure_cfd_node_cell_csr()
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT DISTINCT timestep FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s AND var=%s
                ORDER BY timestep
                """,
                (key.ship, key.scale, key.zone, str(attribute_name).upper()),
            )
            steps = [int(row[0]) for row in cur.fetchall()]
        finally:
            cur.close()

        def fetch_values(ts: int, cell_ids: np.ndarray) -> np.ndarray:
            ids = [int(x) for x in np.asarray(cell_ids, dtype=np.int64).tolist()]
            if not ids:
                return np.zeros((0,), dtype=np.float64)
            q = self._inner.conn.cursor()
            try:
                q.execute(
                    """
                    SELECT cell_id, value FROM cell_scalar
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                      AND timestep=%s AND var=%s AND cell_id = ANY(%s)
                    """,
                    (
                        key.ship, key.scale, key.zone, int(ts),
                        str(attribute_name).upper(), ids,
                    ),
                )
                values = {int(cid): float(value) for cid, value in q.fetchall()}
            finally:
                q.close()
            return np.asarray([values.get(cid, np.nan) for cid in ids], dtype=np.float64)

        return point_frame_extrema_from_cell_values(csr, point_ids, steps, fetch_values)

    def is_h5_dataset(self) -> bool:
        """Whether the connected dataset was ingested from the ODB-like HDF5 path."""
        self._ensure_inner()
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute("SELECT to_regclass('public.h5_frame_metadata')")
            row = cur.fetchone()
            if not row or row[0] is None:
                return False
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM h5_frame_metadata
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                )
                """,
                (key.ship, key.scale, key.zone),
            )
            return bool(cur.fetchone()[0])
        finally:
            cur.close()

    def h5_element_ids_in_coordinate_range(
        self, lower_bound: Sequence[float], upper_bound: Sequence[float]
    ) -> np.ndarray:
        """W9 primitive: source H5 element labels whose centroids fall in a 3-D box."""
        self._ensure_inner()
        key = self._key
        lo = np.asarray(lower_bound, dtype=np.float64).reshape(3)
        hi = np.asarray(upper_bound, dtype=np.float64).reshape(3)
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT src.source_element_label
                FROM cell_centroid c
                JOIN h5_cell_source src
                  ON src.ship_type=c.ship_type AND src.scale=c.scale
                 AND src.zone_type=c.zone_type AND src.cell_id=c.cell_id
                WHERE c.ship_type=%s AND c.scale=%s AND c.zone_type=%s
                  AND c.x BETWEEN %s AND %s
                  AND c.y BETWEEN %s AND %s
                  AND c.z BETWEEN %s AND %s
                ORDER BY src.source_element_label
                """,
                (
                    key.ship, key.scale, key.zone,
                    float(lo[0]), float(hi[0]),
                    float(lo[1]), float(hi[1]),
                    float(lo[2]), float(hi[2]),
                ),
            )
            return np.asarray([int(row[0]) for row in cur.fetchall()], dtype=np.int64)
        finally:
            cur.close()

    def frame_statistics(
        self, attribute_name: Optional[str] = None, step: Optional[int] = None
    ) -> Dict[str, Dict[str, object]]:
        """W10 primitive: descriptive statistics for one HDF5 frame.

        HDF5 nodal quantities are summarized from their genuine ``node_scalar``
        values.  A mapped quantity that has no direct nodal representation is
        summarized from ``cell_scalar`` instead.  This keeps W10 faithful to
        the source field position while retaining the cell-centered projection
        used by W1-W8.
        """
        self._ensure_inner()
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            params = [
                key.ship, key.scale, key.zone, ts,
                key.ship, key.scale, key.zone, ts,
            ]
            nodal_var_clause = ""
            cell_var_clause = ""
            if attribute_name is not None:
                wanted = str(attribute_name).upper()
                nodal_var_clause = " AND var=%s"
                cell_var_clause = " AND c.var=%s"
                params.insert(4, wanted)
                params.append(wanted)
            cur.execute(
                f"""
                WITH nodal AS (
                    SELECT var, 'node'::text AS position, COUNT(*) AS n,
                           MIN(value) AS vmin, MAX(value) AS vmax,
                           AVG(value) AS mean, STDDEV_POP(value) AS stddev
                    FROM node_scalar
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                      AND timestep=%s{nodal_var_clause}
                    GROUP BY var
                ),
                cells AS (
                    SELECT c.var, 'cell'::text AS position, COUNT(*) AS n,
                           MIN(c.value) AS vmin, MAX(c.value) AS vmax,
                           AVG(c.value) AS mean, STDDEV_POP(c.value) AS stddev
                    FROM cell_scalar c
                    WHERE c.ship_type=%s AND c.scale=%s AND c.zone_type=%s
                      AND c.timestep=%s{cell_var_clause}
                      AND NOT EXISTS (
                          SELECT 1 FROM node_scalar n
                          WHERE n.ship_type=c.ship_type AND n.scale=c.scale
                            AND n.zone_type=c.zone_type AND n.timestep=c.timestep
                            AND n.var=c.var
                      )
                    GROUP BY c.var
                )
                SELECT var, position, n, vmin, vmax, mean, stddev FROM nodal
                UNION ALL
                SELECT var, position, n, vmin, vmax, mean, stddev FROM cells
                ORDER BY var
                """,
                tuple(params),
            )
            rows = cur.fetchall()
            if not rows:
                suffix = (
                    f" variable={str(attribute_name).upper()}"
                    if attribute_name is not None
                    else ""
                )
                raise ValueError(f"no values for frame={ts}{suffix}")
            return {
                str(var).upper(): {
                    "position": str(position),
                    "count": int(count),
                    "min": float(vmin),
                    "max": float(vmax),
                    "mean": float(mean),
                    "stddev": float(stddev or 0.0),
                }
                for var, position, count, vmin, vmax, mean, stddev in rows
            }
        finally:
            cur.close()

    def h5_nodal_variables(self) -> Tuple[str, ...]:
        """Variables with direct nodal values available in every ingested HDF5 frame."""
        self._ensure_inner()
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                WITH frame_count AS (
                    SELECT COUNT(*) AS n
                    FROM h5_frame_metadata
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                )
                SELECT ns.var
                FROM node_scalar ns CROSS JOIN frame_count f
                WHERE ns.ship_type=%s AND ns.scale=%s AND ns.zone_type=%s
                  AND f.n > 0
                GROUP BY ns.var, f.n
                HAVING COUNT(DISTINCT ns.timestep) = f.n
                ORDER BY ns.var
                """,
                (
                    key.ship, key.scale, key.zone,
                    key.ship, key.scale, key.zone,
                ),
            )
            return tuple(str(row[0]).upper() for row in cur.fetchall())
        finally:
            cur.close()

    def h5_point_ids(self) -> np.ndarray:
        """Return source HDF5 node labels for the connected dataset."""
        self._ensure_inner()
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT source_node_label
                FROM h5_node_source
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                ORDER BY source_node_label
                """,
                (key.ship, key.scale, key.zone),
            )
            return np.asarray([int(row[0]) for row in cur.fetchall()], dtype=np.int64)
        finally:
            cur.close()

    def h5_point_frame_extrema(
        self, point_ids: Sequence[int], attribute_name: str
    ) -> Dict[int, Tuple[float, float]]:
        """W11 primitive: per-point min/max across every frame for one nodal field."""
        self._ensure_inner()
        ids = [int(x) for x in point_ids]
        if not ids:
            return {}
        key = self._key
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """
                SELECT src.source_node_label, MIN(ns.value), MAX(ns.value)
                FROM h5_node_source src
                JOIN node_scalar ns
                  ON ns.ship_type=src.ship_type AND ns.scale=src.scale
                 AND ns.zone_type=src.zone_type AND ns.node_id=src.node_id
                WHERE src.ship_type=%s AND src.scale=%s AND src.zone_type=%s
                  AND src.source_node_label = ANY(%s::bigint[])
                  AND ns.var=%s
                GROUP BY src.source_node_label
                ORDER BY src.source_node_label
                """,
                (key.ship, key.scale, key.zone, ids, str(attribute_name).upper()),
            )
            return {
                int(point_id): (float(vmin), float(vmax))
                for point_id, vmin, vmax in cur.fetchall()
            }
        finally:
            cur.close()

    def w6_zone_candidates(
        self, dataset_key: str, preferred_zone: Optional[str] = None, hull_hint: Optional[str] = None
    ):
        """Return legacy-CFD mesh zones in W6 preference order.

        Used only by the CFD workload branch. H5 keeps its frozen W6 path.
        """
        self._ensure_inner()
        key = parse_dataset_key(dataset_key)
        cur = self._inner.conn.cursor()
        try:
            cur.execute(
                """SELECT DISTINCT zone_type FROM mesh_metadata
                   WHERE ship_type=%s AND scale=%s ORDER BY zone_type""",
                (key.ship, key.scale),
            )
            zones = [str(r[0]) for r in cur.fetchall()]
            if not zones:
                cur.execute(
                    """SELECT DISTINCT zone_type FROM cell_centroid
                       WHERE ship_type=%s AND scale=%s ORDER BY zone_type""",
                    (key.ship, key.scale),
                )
                zones = [str(r[0]) for r in cur.fetchall()]
        finally:
            cur.close()
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

    def resolve_w6_scalar(self, candidates=("P", "U", "V", "W", "K", "E")) -> str:
        self._ensure_inner()
        ctx = self._require_ctx()
        ordered = []
        for value in candidates:
            name = str(value).strip().upper()
            if name and name not in ordered:
                ordered.append(name)
        cur = self._inner.conn.cursor()
        try:
            for var in ordered:
                cur.execute(
                    """SELECT 1 FROM cell_scalar
                       WHERE ship_type=%s AND scale=%s AND zone_type=%s
                         AND timestep=%s AND var=%s LIMIT 1""",
                    (self._key.ship, self._key.scale, self._key.zone, int(ctx.step), var),
                )
                if cur.fetchone():
                    return var
        finally:
            cur.close()
        raise RuntimeError(
            f"No usable PostgreSQL cell scalar for W6: dataset={ctx.dataset_key} "
            f"step={ctx.step} zone={ctx.zone} candidates={ordered}"
        )

    @staticmethod
    def _w6_normal_from_points(points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) >= 3:
            p0 = pts[0]
            for i in range(1, len(pts) - 1):
                n = np.cross(pts[i] - p0, pts[i + 1] - p0)
                length = float(np.linalg.norm(n))
                if length > 1e-15:
                    return n / length
        if len(pts) >= 2:
            tangent = pts[1] - pts[0]
            length = float(np.linalg.norm(tangent))
            if length > 1e-15:
                tangent /= length
                axes = np.eye(3, dtype=np.float64)
                axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
                n = np.cross(tangent, axis)
                nlen = float(np.linalg.norm(n))
                if nlen > 1e-15:
                    return n / nlen
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    def surface_cells_and_normals(self):
        """Return CFD W6 cell ids aligned to boundary/fallback normals."""
        self._ensure_inner()
        if self._surface_cache is None:
            key = self._key
            self._surface_cache = pg_spatial.fetch_boundary_normals(
                self._inner.conn, key.ship, key.scale, key.zone
            )
        cells, normals = self._surface_cache
        if len(cells):
            return np.asarray(cells, dtype=np.int32), np.asarray(normals, dtype=np.float64)

        # Incomplete legacy datasets may have topology but no materialized
        # boundary_face_geom. Build stable per-cell normals so W6 still runs.
        data = self.runtime.ensure_cell_nodes()
        ids = np.asarray(sorted(set(data.cells.keys()) | set(data.cell_nodes.keys())), dtype=np.int32)
        out = []
        for cid in ids:
            pts = [data.nodes[n] for n in data.cell_nodes.get(int(cid), ()) if n in data.nodes]
            out.append(self._w6_normal_from_points(pts))
        if not out:
            return ids, np.zeros((0, 3), dtype=np.float64)
        return ids, np.asarray(out, dtype=np.float64)

    def surface_norm(self, mesh_handle=None) -> np.ndarray:
        self._ensure_inner()
        if self._surface_cache is None:
            key = self._key
            self._surface_cache = pg_spatial.fetch_boundary_normals(
                self._inner.conn, key.ship, key.scale, key.zone
            )
        return self._surface_cache[1]

    def compute_qcriterion_roi(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        tau: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_inner()
        self._sync_timestep(step)
        ctx = self._require_ctx()
        ts = ctx.step if step is None else int(step)
        key = self._key
        if self._qc_centroids_cache is None:
            from cfd_bench.infra.postgresql.qc_ops import _fetch_centroids, _fetch_neighbors
            self._qc_centroids_cache = _fetch_centroids(
                self._inner.conn, key.ship, key.scale, key.zone
            )
            self._qc_neighbors_cache = _fetch_neighbors(
                self._inner.conn, key.ship, key.scale, key.zone
            )
        return pg_spatial.compute_qcriterion_roi(
            self._inner.conn,
            key.ship,
            key.scale,
            key.zone,
            ts,
            lower_bound,
            upper_bound,
            tau if tau is not None else 0.0,
            centroids=self._qc_centroids_cache,
            neighbors=self._qc_neighbors_cache,
        )
