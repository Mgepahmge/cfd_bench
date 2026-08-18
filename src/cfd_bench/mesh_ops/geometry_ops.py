from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData


def _point_in_bbox(pt: Sequence[float], bb: Tuple[float, float, float, float, float, float], eps: float = 1e-9) -> bool:
    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
    return (
        bb[0] - eps <= x <= bb[1] + eps
        and bb[2] - eps <= y <= bb[3] + eps
        and bb[4] - eps <= z <= bb[5] + eps
    )


def _ensure_bbox_arrays(data: RuntimeMeshData):
    """Return cell ids and AABB arrays, building them once for ad-hoc meshes."""
    if (
        data.all_cell_ids.size
        and len(data.all_cell_ids) == len(data.all_bbox_min) == len(data.all_bbox_max)
    ):
        return data.all_cell_ids, data.all_bbox_min, data.all_bbox_max
    if not data.cells:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )
    items = sorted(data.cells.items(), key=lambda x: x[0])
    ids = np.fromiter((int(cid) for cid, _ in items), dtype=np.int32, count=len(items))
    rows = np.asarray([v for _, v in items], dtype=np.float64)
    mins = np.ascontiguousarray(rows[:, [3, 5, 7]], dtype=np.float64)
    maxs = np.ascontiguousarray(rows[:, [4, 6, 8]], dtype=np.float64)
    data.all_cell_ids = ids
    data.all_bbox_min = mins
    data.all_bbox_max = maxs
    if data.all_centroids.size == 0:
        data.all_centroids = np.ascontiguousarray(rows[:, :3], dtype=np.float64)
    return ids, mins, maxs


def bboxes_for_cell_ids(
    data: RuntimeMeshData, cell_ids: Sequence[int]
) -> dict:
    ids, mins, maxs = _ensure_bbox_arrays(data)
    wanted = np.asarray([int(x) for x in cell_ids], dtype=np.int64)
    if ids.size == 0 or wanted.size == 0:
        return {}
    pos = np.searchsorted(ids, wanted)
    valid = (pos >= 0) & (pos < ids.size)
    valid &= ids[np.clip(pos, 0, max(ids.size - 1, 0))] == wanted
    out = {}
    for cid, idx, ok in zip(wanted.tolist(), pos.tolist(), valid.tolist()):
        if not ok:
            continue
        mn, mx = mins[idx], maxs[idx]
        out[int(cid)] = (float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2]))
    return out


def _bbox_for_id(ids: np.ndarray, mins: np.ndarray, maxs: np.ndarray, cid: int):
    pos = int(np.searchsorted(ids, int(cid)))
    if pos < 0 or pos >= ids.size or int(ids[pos]) != int(cid):
        return None
    mn, mx = mins[pos], maxs[pos]
    return (float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2]))


def cells_in_coordinate_range(
    data: RuntimeMeshData,
    lower_bound: Sequence[float],
    upper_bound: Sequence[float],
) -> np.ndarray:
    """Vectorized W2/W7 coordinate selection using prebuilt AABB arrays."""
    ids, mins, maxs = _ensure_bbox_arrays(data)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int32)
    lo = np.minimum(np.asarray(lower_bound, dtype=np.float64), np.asarray(upper_bound, dtype=np.float64))
    hi = np.maximum(np.asarray(lower_bound, dtype=np.float64), np.asarray(upper_bound, dtype=np.float64))
    mask = np.all(mins >= lo, axis=1) & np.all(maxs <= hi, axis=1)
    return ids[mask].astype(np.int32, copy=False)


def _line_intersects_bbox(
    p0: Sequence[float],
    p1: Sequence[float],
    bb: Tuple[float, float, float, float, float, float],
    eps: float = 1e-9,
) -> Tuple[bool, float]:
    tmin, tmax = 0.0, 1.0
    d = [float(p1[i] - p0[i]) for i in range(3)]
    p = [float(p0[i]) for i in range(3)]
    bounds = [(bb[0], bb[1]), (bb[2], bb[3]), (bb[4], bb[5])]
    for i in range(3):
        if abs(d[i]) < eps:
            if p[i] < bounds[i][0] - eps or p[i] > bounds[i][1] + eps:
                return (False, 0.0)
            continue
        inv = 1.0 / d[i]
        t1 = (bounds[i][0] - p[i]) * inv
        t2 = (bounds[i][1] - p[i]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return (False, 0.0)
    return (True, tmin)


def _plane_hits_bbox(
    origin: Sequence[float],
    normal: Sequence[float],
    bb: Tuple[float, float, float, float, float, float],
    eps: float = 1e-9,
) -> bool:
    n = np.array(normal, dtype=np.float64)
    p0 = np.array(origin, dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < eps:
        return False
    n = n / n_norm
    center = np.array([(bb[0] + bb[1]) * 0.5, (bb[2] + bb[3]) * 0.5, (bb[4] + bb[5]) * 0.5])
    extent = np.array([(bb[1] - bb[0]) * 0.5, (bb[3] - bb[2]) * 0.5, (bb[5] - bb[4]) * 0.5])
    signed = float(np.dot(center - p0, n))
    radius = float(np.dot(extent, np.abs(n)))
    return abs(signed) <= radius + eps


def iotdb_point_intersection(data: RuntimeMeshData, points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    out: List[int] = []
    ox, oy, oz = data.spatial_origin
    sx, sy, sz = data.spatial_step
    dx, dy, dz = data.spatial_dims
    all_ids, all_min, all_max = _ensure_bbox_arrays(data)
    if all_ids.size == 0:
        return np.zeros((0,), dtype=np.int32)
    global_min = np.min(all_min, axis=0)
    global_max = np.max(all_max, axis=0)

    def _bucket_of(pt):
        ix = int(np.floor((float(pt[0]) - ox) / sx)) if sx > 0 else 0
        iy = int(np.floor((float(pt[1]) - oy) / sy)) if sy > 0 else 0
        iz = int(np.floor((float(pt[2]) - oz) / sz)) if sz > 0 else 0
        ix = min(max(ix, 0), max(dx - 1, 0))
        iy = min(max(iy, 0), max(dy - 1, 0))
        iz = min(max(iz, 0), max(dz - 1, 0))
        return (ix, iy, iz)

    for pt in np.asarray(points, dtype=np.float64).reshape(-1, 3):
        # Very common for W4/W5: a particle has left the mesh. Avoid clamping
        # it into an edge bucket and then scanning every AABB just to prove it.
        if np.any(pt < global_min - eps) or np.any(pt > global_max + eps):
            continue

        hit = -1
        if data.spatial_buckets:
            bx, by, bz = _bucket_of(pt)
            for ix in (bx - 1, bx, bx + 1):
                if ix < 0 or ix >= dx:
                    continue
                for iy in (by - 1, by, by + 1):
                    if iy < 0 or iy >= dy:
                        continue
                    for iz in (bz - 1, bz, bz + 1):
                        if iz < 0 or iz >= dz:
                            continue
                        for cid in data.spatial_buckets.get((ix, iy, iz), ()):
                            bb = _bbox_for_id(all_ids, all_min, all_max, int(cid))
                            if bb is not None and _point_in_bbox(pt, bb, eps):
                                hit = int(cid)
                                break
                        if hit >= 0:
                            break
                    if hit >= 0:
                        break
                if hit >= 0:
                    break

        if hit < 0:
            mask = np.all(all_min - eps <= pt, axis=1) & np.all(all_max + eps >= pt, axis=1)
            idx = np.flatnonzero(mask)
            if idx.size:
                hit = int(all_ids[int(idx[0])])
        if hit >= 0:
            out.append(hit)
    return np.asarray(out, dtype=np.int32)


def iotdb_line_intersection(
    data: RuntimeMeshData, line_start: Sequence[float], line_end: Sequence[float], eps: float = 1e-9
) -> np.ndarray:
    """Vectorized slab test for a segment against every cell AABB."""
    ids, mins, maxs = _ensure_bbox_arrays(data)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int32)
    p0 = np.asarray(line_start, dtype=np.float64).reshape(3)
    p1 = np.asarray(line_end, dtype=np.float64).reshape(3)
    d = p1 - p0
    tmin = np.zeros(ids.size, dtype=np.float64)
    tmax = np.ones(ids.size, dtype=np.float64)
    valid = np.ones(ids.size, dtype=bool)
    for axis in range(3):
        if abs(float(d[axis])) < eps:
            valid &= (p0[axis] >= mins[:, axis] - eps) & (p0[axis] <= maxs[:, axis] + eps)
            continue
        inv = 1.0 / float(d[axis])
        ta = (mins[:, axis] - p0[axis]) * inv
        tb = (maxs[:, axis] - p0[axis]) * inv
        lo = np.minimum(ta, tb)
        hi = np.maximum(ta, tb)
        tmin = np.maximum(tmin, lo)
        tmax = np.minimum(tmax, hi)
        valid &= tmin <= tmax + eps
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int32)
    order = np.argsort(tmin[idx], kind="stable")
    return ids[idx[order]].astype(np.int32, copy=False)


def iotdb_plane_intersection(
    data: RuntimeMeshData, plane_origin: Sequence[float], plane_norm: Sequence[float], eps: float = 1e-9
) -> np.ndarray:
    """Vectorized plane/AABB overlap using center-radius projection."""
    ids, mins, maxs = _ensure_bbox_arrays(data)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int32)
    origin = np.asarray(plane_origin, dtype=np.float64).reshape(3)
    normal = np.asarray(plane_norm, dtype=np.float64).reshape(3)
    nlen = float(np.linalg.norm(normal))
    if nlen < eps:
        return np.zeros((0,), dtype=np.int32)
    normal = normal / nlen
    centers = 0.5 * (mins + maxs)
    extents = 0.5 * (maxs - mins)
    signed = (centers - origin) @ normal
    radius = extents @ np.abs(normal)
    mask = np.abs(signed) <= radius + eps
    return ids[mask].astype(np.int32, copy=False)
