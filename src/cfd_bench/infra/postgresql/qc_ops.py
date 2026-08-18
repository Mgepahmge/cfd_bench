"""Q-criterion ROI computation for PostgreSQL (W7)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# Centroids and adjacency are static for all timesteps.  W7 creates a fresh
# client per step, so keep a tiny shared LRU to avoid re-downloading millions
# of rows three times for the same CFD mesh.
_STATIC_Q_CACHE = OrderedDict()
_STATIC_Q_CACHE_LIMIT = 2


def _static_cache_key(conn, ship_type: str, scale: str, zone_type: str):
    dsn = getattr(conn, "dsn", None)
    if not dsn:
        return None
    return (str(dsn), str(ship_type), str(scale), str(zone_type))


def _get_static_q_maps(conn, ship_type: str, scale: str, zone_type: str):
    key = _static_cache_key(conn, ship_type, scale, zone_type)
    if key is not None and key in _STATIC_Q_CACHE:
        value = _STATIC_Q_CACHE.pop(key)
        _STATIC_Q_CACHE[key] = value
        return value
    value = (
        _fetch_centroids_uncached(conn, ship_type, scale, zone_type),
        _fetch_neighbors_uncached(conn, ship_type, scale, zone_type),
    )
    if key is not None:
        _STATIC_Q_CACHE[key] = value
        while len(_STATIC_Q_CACHE) > _STATIC_Q_CACHE_LIMIT:
            _STATIC_Q_CACHE.popitem(last=False)
    return value


def _fetch_centroids_uncached(conn, ship_type: str, scale: str, zone_type: str) -> Dict[int, Tuple[float, float, float]]:
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


def _fetch_neighbors_uncached(conn, ship_type: str, scale: str, zone_type: str) -> Dict[int, List[int]]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT cell_id, neighbor_ids FROM cell_adjacency
            WHERE ship_type=%s AND scale=%s AND zone_type=%s
            """,
            (ship_type, scale, zone_type),
        )
        return {
            int(cid): [int(n) for n in (nbrs or []) if int(n) >= 0]
            for cid, nbrs in cur.fetchall()
        }
    finally:
        cur.close()


def _fetch_centroids(conn, ship_type: str, scale: str, zone_type: str) -> Dict[int, Tuple[float, float, float]]:
    return _get_static_q_maps(conn, ship_type, scale, zone_type)[0]


def _fetch_neighbors(conn, ship_type: str, scale: str, zone_type: str) -> Dict[int, List[int]]:
    return _get_static_q_maps(conn, ship_type, scale, zone_type)[1]


def _fetch_velocity_map(conn, ship_type: str, scale: str, zone_type: str, timestep: int, cell_ids: List[int]) -> Dict[int, Tuple[float, float, float]]:
    if not cell_ids:
        return {}
    cur = conn.cursor()
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
            (ship_type, scale, zone_type, int(timestep), cell_ids),
        )
        return {
            int(cid): (float(u), float(v), float(w))
            for cid, u, v, w in cur.fetchall()
            if u is not None and v is not None and w is not None
        }
    finally:
        cur.close()


def _gradient_ls(center_xyz, center_vel, nb_xyz, nb_vel):
    if len(nb_xyz) < 3:
        return None
    A = np.asarray(nb_xyz, dtype=np.float64) - np.asarray(center_xyz, dtype=np.float64)
    b = np.asarray(nb_vel, dtype=np.float64) - np.asarray(center_vel, dtype=np.float64)
    try:
        G, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return G.T
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
    *,
    centroids: Optional[Dict[int, Tuple[float, float, float]]] = None,
    neighbors: Optional[Dict[int, List[int]]] = None,
) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
    from cfd_bench.infra.postgresql.spatial import range_query_coord

    roi = range_query_coord(conn, ship_type, scale, zone_type, lower_bound, upper_bound)
    if roi.size == 0:
        return roi, np.array([], dtype=np.float64)
    if centroids is None:
        centroids = _fetch_centroids(conn, ship_type, scale, zone_type)
    if neighbors is None:
        neighbors = _fetch_neighbors(conn, ship_type, scale, zone_type)

    # Fetch one adjacency halo so boundary cells can use neighboring velocity;
    # the static centroid/adjacency maps themselves are cached by the client.
    needed = {int(cid) for cid in roi.tolist()}
    for cid in list(needed):
        needed.update(int(nb) for nb in neighbors.get(cid, ()))
    vel_map = _fetch_velocity_map(conn, ship_type, scale, zone_type, timestep, sorted(needed))

    qc_vals = []
    keep = []
    for cid in roi.tolist():
        cxyz = centroids.get(int(cid))
        cvel = vel_map.get(int(cid))
        if cxyz is None or cvel is None:
            continue
        nb_ids = [nb for nb in neighbors.get(int(cid), ()) if nb in vel_map and nb in centroids]
        if len(nb_ids) < 3:
            continue
        G = _gradient_ls(cxyz, cvel, [centroids[nb] for nb in nb_ids], [vel_map[nb] for nb in nb_ids])
        if G is None:
            continue
        qc = _qc_from_gradient(G)
        if qc >= tau:
            keep.append(int(cid))
            qc_vals.append(float(qc))
    return np.asarray(keep, dtype=np.int32), np.asarray(qc_vals, dtype=np.float64)
