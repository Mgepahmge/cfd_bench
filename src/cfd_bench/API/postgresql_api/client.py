from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.core.context import DatasetKey, MeshContext, parse_dataset_key
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
        key = self._key
        return pg_spatial.fetch_var_value_range(
            self._inner.conn, key.ship, key.scale, key.zone, ts, attribute_name
        )

    def point_query(self, cell_indexes: Sequence[int], attribute_name: str, step: Optional[int] = None) -> np.ndarray:
        self._ensure_inner()
        self._sync_timestep(step)
        return self._inner.point_query(None, cell_indexes, attribute_name, timestep=step)

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
        if self._centroids_cache is None:
            self._centroids_cache = pg_spatial._fetch_centroids_map(
                self._inner.conn, key.ship, key.scale, key.zone
            )
        return pg_spatial.point_intersection(
            self._inner.conn, key.ship, key.scale, key.zone, points, self._centroids_cache
        )

    def line_intersection(self, line_start: Sequence[float], line_end: Sequence[float]) -> np.ndarray:
        self._ensure_inner()
        return self._inner.vtk_line_intersection(None, line_start, line_end)

    def plane_intersection(self, plane_origin: Sequence[float], plane_norm: Sequence[float]) -> np.ndarray:
        self._ensure_inner()
        return self._inner.vtk_plane_intersection(None, plane_origin, plane_norm)

    def extract_submesh(self, cell_indexes: Sequence[int], mesh_handle=None) -> LiteMesh:
        self._ensure_inner()
        return self.runtime.extract_submesh(cell_indexes)

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

    def surface_norm(self, mesh_handle=None) -> np.ndarray:
        self._ensure_inner()
        key = self._key
        _, norms = pg_spatial.fetch_boundary_normals(
            self._inner.conn, key.ship, key.scale, key.zone
        )
        return norms

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
        return pg_spatial.compute_qcriterion_roi(
            self._inner.conn,
            key.ship,
            key.scale,
            key.zone,
            ts,
            lower_bound,
            upper_bound,
            tau if tau is not None else 0.0,
        )
