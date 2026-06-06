"""Q-criterion ROI computation for PostgreSQL (W7)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def _fetch_centroids(conn, ship_type: str, scale: str, zone_type: str) -> Dict[int, Tuple[float, float, float]]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT cell_id, x, y, z FROM cell_centroid
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        return {int(r[0]): (float(r[1]), float(r[2]), float(r[3])) for r in cur.fetchall()}
    finally:
        cur.close()


def _fetch_neighbors(conn, ship_type: str, scale: str, zone_type: str) -> Dict[int, List[int]]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT cell_id, neighbor_ids FROM cell_adjacency
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        out: Dict[int, List[int]] = {}
        for cid, nbrs in cur.fetchall():
            out[int(cid)] = [int(n) for n in (nbrs or []) if int(n) >= 0]
        return out
    finally:
        cur.close()


def _fetch_velocity_map(conn, ship_type: str, scale: str, zone_type: str, timestep: int, cell_ids: List[int]) -> Dict[int, Tuple[float, float, float]]:
    if not cell_ids:
        return {}
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT cell_id, var, value FROM cell_scalar
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
              AND timestep=%s AND var IN ('U','V','W')
              AND cell_id = ANY(%s)
            """,
            (ship_type, scale, zone_type, int(timestep), cell_ids),
        )
        partial: Dict[int, Dict[str, float]] = {}
        for cid, var, val in cur.fetchall():
            partial.setdefault(int(cid), {})[str(var)] = float(val)
        return {
            cid: (d.get("U", 0.0), d.get("V", 0.0), d.get("W", 0.0))
            for cid, d in partial.items()
            if "U" in d and "V" in d and "W" in d
        }
    finally:
        cur.close()


def _gradient_ls(center_xyz, center_vel, nb_xyz, nb_vel):
    if len(nb_xyz) < 3:
        return None
    A = []
    b = []
    for (x, y, z), (u, v, w) in zip(nb_xyz, nb_vel):
        A.append([x - center_xyz[0], y - center_xyz[1], z - center_xyz[2]])
        b.append([u - center_vel[0], v - center_vel[1], w - center_vel[2]])
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    try:
        G, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return G.T  # 3x3
    except Exception:
        return None


def _qc_from_gradient(G: np.ndarray) -> float:
    S = 0.5 * (G + G.T)
    Omega = 0.5 * (G - G.T)
    return 0.5 * (np.linalg.norm(Omega, "fro") ** 2 - np.linalg.norm(S, "fro") ** 2)


def qcriterion_roi(
    conn,
    ship_type: str,
    scale: str,
    zone_type: str,
    timestep: int,
    lower_bound: Sequence[float],
    upper_bound: Sequence[float],
    tau: float,
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    from cfd_bench.infra.postgresql.spatial import range_query_coord

    roi = range_query_coord(conn, ship_type, scale, zone_type, lower_bound, upper_bound)
    if roi.size == 0:
        return roi, np.array([], dtype=np.float64)
    centroids = _fetch_centroids(conn, ship_type, scale, zone_type)
    neighbors = _fetch_neighbors(conn, ship_type, scale, zone_type)
    vel_map = _fetch_velocity_map(conn, ship_type, scale, zone_type, timestep, roi.tolist())
    qc_vals = []
    keep = []
    for cid in roi.tolist():
        cxyz = centroids.get(int(cid))
        cvel = vel_map.get(int(cid))
        if cxyz is None or cvel is None:
            continue
        nb_xyz, nb_vel = [], []
        for nb in neighbors.get(int(cid), []):
            if nb in vel_map and nb in centroids:
                nb_xyz.append(centroids[nb])
                nb_vel.append(vel_map[nb])
        G = _gradient_ls(cxyz, cvel, nb_xyz, nb_vel)
        if G is None:
            continue
        qc = _qc_from_gradient(G)
        if qc >= tau:
            keep.append(int(cid))
            qc_vals.append(qc)
    return np.array(keep, dtype=np.int32), np.array(qc_vals, dtype=np.float64)
