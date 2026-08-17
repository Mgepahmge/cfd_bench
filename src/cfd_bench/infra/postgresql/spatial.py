"""PostGIS / SQL spatial queries for PostgreSQL mesh backend."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
    return (float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2


def fetch_mesh_bounds(conn, ship_type: str, scale: str, zone_type: str) -> Optional[List[float]]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT MIN(x), MAX(x), MIN(y), MAX(y), MIN(z), MAX(z)
            FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        return [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
    finally:
        cur.close()


def fetch_cell_count(conn, ship_type: str, scale: str, zone_type: str) -> int:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT COUNT(*) FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        return int(cur.fetchone()[0])
    finally:
        cur.close()


def fetch_var_value_range(
    conn, ship_type: str, scale: str, zone_type: str, timestep: int, var: str
) -> Tuple[float, float]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT MIN(value), MAX(value)
            FROM cell_scalar
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
              AND timestep=%s AND var=%s
            """,
            (ship_type, scale, zone_type, int(timestep), str(var).upper()),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return 0.0, 1.0
        return float(row[0]), float(row[1])
    finally:
        cur.close()


def range_query_coord(
    conn, ship_type: str, scale: str, zone_type: str, lower_bound: Sequence[float], upper_bound: Sequence[float]
) -> NDArray[np.int32]:
    x0, y0, z0 = map(float, lower_bound)
    x1, y1, z1 = map(float, upper_bound)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT cell_id
            FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
              AND x BETWEEN %s AND %s
              AND y BETWEEN %s AND %s
              AND z BETWEEN %s AND %s
            ORDER BY cell_id
            """,
            (ship_type, scale, zone_type, min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), min(z0, z1), max(z0, z1)),
        )
        return np.array([int(r[0]) for r in cur.fetchall()], dtype=np.int32)
    finally:
        cur.close()


def _bucket_candidates(conn, ship_type: str, scale: str, zone_type: str, point_xyz: Sequence[float]) -> List[int]:
    x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT cell_ids FROM point_locator_grid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
              AND %s BETWEEN x_min AND x_max
              AND %s BETWEEN y_min AND y_max
              AND %s BETWEEN z_min AND z_max
            LIMIT 1
            """,
            (ship_type, scale, zone_type, x, y, z),
        )
        row = cur.fetchone()
        if not row:
            return []
        return [int(c) for c in (row[0] or [])]
    finally:
        cur.close()


def _fetch_centroids_map(conn, ship_type: str, scale: str, zone_type: str) -> Dict[int, Tuple[float, float, float]]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT cell_id, x, y, z FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        return {int(cid): (float(x), float(y), float(z)) for cid, x, y, z in cur.fetchall()}
    finally:
        cur.close()


def point_intersection(
    conn,
    ship_type: str,
    scale: str,
    zone_type: str,
    points: NDArray[np.float64],
    centroids: Optional[Dict[int, Tuple[float, float, float]]] = None,
) -> NDArray[np.int32]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.array([], dtype=np.int32)
    if centroids is None:
        centroids = _fetch_centroids_map(conn, ship_type, scale, zone_type)
    out: List[int] = []
    for pt in pts:
        candidates = _bucket_candidates(conn, ship_type, scale, zone_type, pt)
        if candidates:
            cid = min(candidates, key=lambda c: _dist2(pt, centroids.get(c, (1e18, 1e18, 1e18))))
            out.append(int(cid))
        # No bucket hit means no containing cell candidate.  Do not snap the
        # point to a globally-nearest centroid: streamline workloads rely on
        # an empty result to detect that a particle has left the mesh.
    return np.array(out, dtype=np.int32)


def fetch_boundary_normals(
    conn, ship_type: str, scale: str, zone_type: str
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Return (cell_ids, normals Nx3) aggregated per cell from boundary_face_geom."""
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT cell_id,
                   SUM(nx * area) / NULLIF(SUM(area), 0),
                   SUM(ny * area) / NULLIF(SUM(area), 0),
                   SUM(nz * area) / NULLIF(SUM(area), 0)
            FROM boundary_face_geom
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            GROUP BY cell_id
            ORDER BY cell_id
            """,
            (ship_type, scale, zone_type),
        )
        rows = cur.fetchall()
        if not rows:
            return np.array([], dtype=np.int32), np.zeros((0, 3), dtype=np.float64)
        cids = [int(r[0]) for r in rows]
        norms = np.array([[float(r[1] or 0), float(r[2] or 0), float(r[3] or 0)] for r in rows], dtype=np.float64)
        lens = np.linalg.norm(norms, axis=1, keepdims=True)
        lens = np.maximum(lens, 1e-15)
        norms = norms / lens
        return np.array(cids, dtype=np.int32), norms
    finally:
        cur.close()


def compute_qcriterion_roi(
    conn,
    ship_type: str,
    scale: str,
    zone_type: str,
    timestep: int,
    lower_bound: Sequence[float],
    upper_bound: Sequence[float],
    tau: float = 0.0,
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    from cfd_bench.infra.postgresql.qc_ops import qcriterion_roi

    return qcriterion_roi(
        conn, ship_type, scale, zone_type, int(timestep), lower_bound, upper_bound, float(tau)
    )
