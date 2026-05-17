from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import psycopg2
from numpy.typing import NDArray


class PG_Interface:
    """
    PostGIS-backed replacement for selected VTK_Interface methods.

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
        p0 = np.array(plane_origin, dtype=np.float64)
        n = np.array(plane_norm, dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            raise ValueError("plane_norm too small")
        n = n / n_norm
        d = -float(np.dot(n, p0))

        cur = self.conn.cursor()
        try:
            # Load nodes once for current zone.
            cur.execute(
                """
                SELECT node_id, x, y, z
                FROM node_coordinates
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                """,
                (self.ship_type, self.scale, self.zone_type),
            )
            node_xyz = {int(nid): (float(x), float(y), float(z)) for nid, x, y, z in cur.fetchall()}

            cur.execute(
                """
                SELECT cell_id, node_ids
                FROM cell_nodes
                WHERE ship_type=%s AND scale=%s AND zone_type=%s
                """,
                (self.ship_type, self.scale, self.zone_type),
            )
            rows = cur.fetchall()
        finally:
            cur.close()

        out = []
        for cell_id, node_ids in rows:
            nids = list(node_ids or [])
            if not nids:
                continue
            s_min, s_max = None, None
            for nid in nids:
                xyz = node_xyz.get(int(nid))
                if xyz is None:
                    continue
                s = float(n[0] * xyz[0] + n[1] * xyz[1] + n[2] * xyz[2] + d)
                s_min = s if s_min is None else min(s_min, s)
                s_max = s if s_max is None else max(s_max, s)
            if s_min is None:
                continue
            if s_min <= eps and s_max >= -eps:
                out.append(int(cell_id))
        return np.array(out, dtype=np.int32)

