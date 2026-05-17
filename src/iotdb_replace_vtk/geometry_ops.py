from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .mesh_runtime import RuntimeMeshData


def _point_in_bbox(pt: Sequence[float], bb: Tuple[float, float, float, float, float, float], eps: float = 1e-9) -> bool:
    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
    return (
        bb[0] - eps <= x <= bb[1] + eps
        and bb[2] - eps <= y <= bb[3] + eps
        and bb[4] - eps <= z <= bb[5] + eps
    )


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
    corners = np.array(
        [
            [bb[0], bb[2], bb[4]],
            [bb[0], bb[2], bb[5]],
            [bb[0], bb[3], bb[4]],
            [bb[0], bb[3], bb[5]],
            [bb[1], bb[2], bb[4]],
            [bb[1], bb[2], bb[5]],
            [bb[1], bb[3], bb[4]],
            [bb[1], bb[3], bb[5]],
        ],
        dtype=np.float64,
    )
    signed = (corners - p0) @ n
    return float(np.min(signed)) <= eps and float(np.max(signed)) >= -eps


def iotdb_point_intersection(data: RuntimeMeshData, points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    bboxes = data.cell_bbox
    out: List[int] = []
    ox, oy, oz = data.spatial_origin
    sx, sy, sz = data.spatial_step
    dx, dy, dz = data.spatial_dims

    # Pre-bind for fast fallback vectorized scan.
    all_ids = data.all_cell_ids
    all_min = data.all_bbox_min
    all_max = data.all_bbox_max

    def _bucket_of(pt):
        ix = int(np.floor((float(pt[0]) - ox) / sx)) if sx > 0 else 0
        iy = int(np.floor((float(pt[1]) - oy) / sy)) if sy > 0 else 0
        iz = int(np.floor((float(pt[2]) - oz) / sz)) if sz > 0 else 0
        ix = min(max(ix, 0), max(dx - 1, 0))
        iy = min(max(iy, 0), max(dy - 1, 0))
        iz = min(max(iz, 0), max(dz - 1, 0))
        return (ix, iy, iz)

    for pt in points:
        hit = -1
        # 1) spatial bucket candidates (bucket + 26-neighborhood).
        if data.spatial_buckets:
            bx, by, bz = _bucket_of(pt)
            cand: List[int] = []
            for ix in (bx - 1, bx, bx + 1):
                if ix < 0 or ix >= dx:
                    continue
                for iy in (by - 1, by, by + 1):
                    if iy < 0 or iy >= dy:
                        continue
                    for iz in (bz - 1, bz, bz + 1):
                        if iz < 0 or iz >= dz:
                            continue
                        cand.extend(data.spatial_buckets.get((ix, iy, iz), []))
            for cid in cand:
                bb = bboxes.get(int(cid))
                if bb is not None and _point_in_bbox(pt, bb, eps):
                    hit = int(cid)
                    break

        # 2) fallback: vectorized full scan (rare path).
        if hit < 0 and all_ids.size > 0:
            px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
            mask = (
                (all_min[:, 0] - eps <= px)
                & (all_max[:, 0] + eps >= px)
                & (all_min[:, 1] - eps <= py)
                & (all_max[:, 1] + eps >= py)
                & (all_min[:, 2] - eps <= pz)
                & (all_max[:, 2] + eps >= pz)
            )
            idx = np.flatnonzero(mask)
            if idx.size > 0:
                hit = int(all_ids[int(idx[0])])
        out.append(hit)
    return np.array(out, dtype=np.int32)


def iotdb_line_intersection(
    data: RuntimeMeshData, line_start: Sequence[float], line_end: Sequence[float], eps: float = 1e-9
) -> np.ndarray:
    hits: List[Tuple[float, int]] = []
    for cid, bb in data.cell_bbox.items():
        ok, t = _line_intersects_bbox(line_start, line_end, bb, eps)
        if ok:
            hits.append((float(t), int(cid)))
    hits.sort(key=lambda x: x[0])
    return np.array([cid for _, cid in hits], dtype=np.int32)


def iotdb_plane_intersection(
    data: RuntimeMeshData, plane_origin: Sequence[float], plane_norm: Sequence[float], eps: float = 1e-9
) -> np.ndarray:
    out = [int(cid) for cid, bb in data.cell_bbox.items() if _plane_hits_bbox(plane_origin, plane_norm, bb, eps)]
    return np.array(out, dtype=np.int32)
