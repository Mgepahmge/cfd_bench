from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import psycopg2
from numpy.typing import NDArray


# Plane/AABB geometry is static across timesteps.  Keep a small process-local
# cache so W1 does not rebuild the same multi-million-cell bounds for every
# step.  It is deliberately tiny to cap memory use on large CFD cases.
_PLANE_BBOX_CACHE = OrderedDict()
_PLANE_BBOX_CACHE_LIMIT = 2


def _plane_cache_key(conn, ship_type: str, scale: str, zone_type: str):
    dsn = getattr(conn, "dsn", None)
    if not dsn:
        return None
    return (str(dsn), str(ship_type), str(scale), str(zone_type))


class PGMeshBackend:
    """
    PostGIS-backed mesh backend for PostgreSQLMeshClient.

    Implemented replaceable methods:
      - point_query
      - range_query_var
      - vtk_line_intersection
      - vtk_plane_intersection
    """

    def __init__(
        self,
        ship_type: str = "JBC",
        scale: str = "615k",
        zone_type: str = "0_Fluid",
        timestep: int = 0,
        db_name: str = "cae_data",
        db_user: str = "postgres",
        db_password: str = "123456",
        db_host: str = "localhost",
        db_port: str = "5432",
    ):
        self.ship_type = ship_type
        self.scale = scale
        self.zone_type = zone_type
        self.timestep = int(timestep)
        self._plane_bbox_cache = None
        self.conn = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )

    def close(self):
        if self.conn:
            self.conn.close()

    # ------------------------------------------------------------------
    # 1) point_query replacement
    # ------------------------------------------------------------------
    def point_query(
        self,
        vtk_mesh,  # kept for API compatibility, not used
        cell_indexes: np.array,
        attribute_name: str,
        timestep: Optional[int] = None,
    ) -> NDArray[np.float64]:
        ts = self.timestep if timestep is None else int(timestep)
        attr = str(attribute_name).upper()
        cell_ids = [int(c) for c in np.array(cell_indexes, dtype=np.int64).tolist()]
        if not cell_ids:
            return np.array([], dtype=np.float64)
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT cell_id, value
                FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND timestep=%s AND var=%s
                  AND cell_id = ANY(%s)
                """,
                (self.ship_type, self.scale, self.zone_type, ts, attr, cell_ids),
            )
            m = {int(cid): float(v) for cid, v in cur.fetchall()}
            # Keep output order consistent with input IDs.
            out = [m.get(cid, np.nan) for cid in cell_ids]
            return np.array(out, dtype=np.float64)
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # 2) range_query_var replacement
    # ------------------------------------------------------------------
    def range_query_var(
        self,
        vtk_mesh,  # kept for API compatibility, not used
        lower_bound: float,
        upper_bound: float,
        attribute_name: str,
        timestep: Optional[int] = None,
    ):
        ts = self.timestep if timestep is None else int(timestep)
        attr = str(attribute_name).upper()
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT cell_id
                FROM cell_scalar
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                  AND timestep=%s AND var=%s
                  AND value BETWEEN %s AND %s
                ORDER BY cell_id
                """,
                (
                    self.ship_type,
                    self.scale,
                    self.zone_type,
                    ts,
                    attr,
                    float(lower_bound),
                    float(upper_bound),
                ),
            )
            return np.array([int(r[0]) for r in cur.fetchall()], dtype=np.int32)
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # 3) vtk_line_intersection replacement
    # ------------------------------------------------------------------
    def vtk_line_intersection(
        self,
        vtk_mesh,  # kept for API compatibility, not used
        line_start: Sequence[float],
        line_end: Sequence[float],
    ):
        x0, y0, z0 = map(float, line_start)
        x1, y1, z1 = map(float, line_end)
        vx, vy, vz = x1 - x0, y1 - y0, z1 - z0
        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                WITH ln AS (
                    SELECT ST_SetSRID(
                        ST_MakeLine(
                            ST_MakePoint(%s, %s, %s),
                            ST_MakePoint(%s, %s, %s)
                        ),
                        0
                    ) AS g
                )
                SELECT cg.cell_id
                FROM cell_geom_full cg
                CROSS JOIN ln
                WHERE cg.ship_type=%s AND cg.scale=%s AND cg.zone_type=%s
                  AND cg.geom &&& (SELECT g FROM ln)
                  AND ST_3DIntersects(cg.geom, (SELECT g FROM ln))
                ORDER BY
                    (ST_X(cg.centroid)-%s)*%s
                    + (ST_Y(cg.centroid)-%s)*%s
                    + (ST_Z(cg.centroid)-%s)*%s
                """,
                (
                    x0,
                    y0,
                    z0,
                    x1,
                    y1,
                    z1,
                    self.ship_type,
                    self.scale,
                    self.zone_type,
                    x0,
                    vx,
                    y0,
                    vy,
                    z0,
                    vz,
                ),
            )
            return np.array([int(r[0]) for r in cur.fetchall()], dtype=np.int32)
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # 4) vtk_plane_intersection replacement
    # ------------------------------------------------------------------
    def vtk_plane_intersection(
        self,
        vtk_mesh,  # kept for API compatibility, not used
        plane_origin: Sequence[float],
        plane_norm: Sequence[float],
        eps: float = 1e-9,
    ):
        """Vectorized plane/AABB intersection with one-time topology loading.

        Older code re-read every node and every ``cell_nodes`` row for every
        W1 plane transaction.  The static cell AABBs are now assembled once
        per client and all subsequent plane tests run in NumPy.  This matches
        the bounding-box geometry used by the IoTDB/TileDB DB engines.
        """
        p0 = np.asarray(plane_origin, dtype=np.float64).reshape(3)
        n = np.asarray(plane_norm, dtype=np.float64).reshape(3)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-15:
            raise ValueError("plane_norm too small")
        n = n / n_norm

        if self._plane_bbox_cache is None:
            cache_key = _plane_cache_key(self.conn, self.ship_type, self.scale, self.zone_type)
            if cache_key is not None and cache_key in _PLANE_BBOX_CACHE:
                self._plane_bbox_cache = _PLANE_BBOX_CACHE.pop(cache_key)
                _PLANE_BBOX_CACHE[cache_key] = self._plane_bbox_cache

        if self._plane_bbox_cache is None:
            cache_key = _plane_cache_key(self.conn, self.ship_type, self.scale, self.zone_type)
            cur = self.conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT node_id, x, y, z
                    FROM node_coordinates
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    """,
                    (self.ship_type, self.scale, self.zone_type),
                )
                node_xyz = {
                    int(nid): np.array((float(x), float(y), float(z)), dtype=np.float64)
                    for nid, x, y, z in cur.fetchall()
                }
                cur.execute(
                    """
                    SELECT cell_id, node_ids
                    FROM cell_nodes
                    WHERE ship_type=%s AND scale=%s AND zone_type=%s
                    ORDER BY cell_id
                    """,
                    (self.ship_type, self.scale, self.zone_type),
                )
                rows = cur.fetchall()
            finally:
                cur.close()

            ids = []
            mins = []
            maxs = []
            for cell_id, node_ids in rows:
                pts = [node_xyz.get(int(nid)) for nid in (node_ids or [])]
                pts = [p for p in pts if p is not None]
                if not pts:
                    continue
                xyz = np.asarray(pts, dtype=np.float64)
                ids.append(int(cell_id))
                mins.append(np.min(xyz, axis=0))
                maxs.append(np.max(xyz, axis=0))
            mins_arr = np.asarray(mins, dtype=np.float64).reshape(-1, 3)
            maxs_arr = np.asarray(maxs, dtype=np.float64).reshape(-1, 3)
            self._plane_bbox_cache = (
                np.asarray(ids, dtype=np.int32),
                mins_arr,
                maxs_arr,
                np.ascontiguousarray(0.5 * (mins_arr + maxs_arr), dtype=np.float64),
                np.ascontiguousarray(0.5 * (maxs_arr - mins_arr), dtype=np.float64),
            )
            if cache_key is not None:
                _PLANE_BBOX_CACHE[cache_key] = self._plane_bbox_cache
                while len(_PLANE_BBOX_CACHE) > _PLANE_BBOX_CACHE_LIMIT:
                    _PLANE_BBOX_CACHE.popitem(last=False)

        cached = self._plane_bbox_cache
        if len(cached) >= 5:
            ids, mins, maxs, centers, extents = cached[:5]
        else:  # compatibility with a cache populated by an older client object
            ids, mins, maxs = cached
            centers = np.ascontiguousarray(0.5 * (mins + maxs), dtype=np.float64)
            extents = np.ascontiguousarray(0.5 * (maxs - mins), dtype=np.float64)
            self._plane_bbox_cache = (ids, mins, maxs, centers, extents)
        if ids.size == 0:
            return np.zeros((0,), dtype=np.int32)
        origin_dot = float(p0 @ n)
        abs_n = np.abs(n)
        chunk = 500_000
        if ids.size <= chunk:
            signed = centers @ n - origin_dot
            radius = extents @ abs_n
            return ids[np.abs(signed) <= radius + float(eps)].astype(np.int32, copy=False)
        hits = []
        for start in range(0, ids.size, chunk):
            end = min(start + chunk, ids.size)
            signed = centers[start:end] @ n - origin_dot
            radius = extents[start:end] @ abs_n
            mask = np.abs(signed) <= radius + float(eps)
            if np.any(mask):
                hits.append(ids[start:end][mask])
        return np.concatenate(hits).astype(np.int32, copy=False) if hits else np.zeros((0,), dtype=np.int32)

