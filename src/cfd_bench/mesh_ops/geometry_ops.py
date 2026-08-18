from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData


# Above this size, process whole-mesh vector operations in chunks to cap peak
# temporary memory. This avoids swapping on multi-million-cell CFD meshes while
# retaining the fast single-vector path for small/medium datasets.
_GEOM_CHUNK = 500_000


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
    data.all_bbox_center = np.ascontiguousarray(0.5 * (mins + maxs), dtype=np.float64)
    data.all_bbox_extent = np.ascontiguousarray(0.5 * (maxs - mins), dtype=np.float64)
    data.global_bbox_min = np.min(mins, axis=0)
    data.global_bbox_max = np.max(maxs, axis=0)
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
    if ids.size <= _GEOM_CHUNK:
        mask = np.all(mins >= lo, axis=1) & np.all(maxs <= hi, axis=1)
        return ids[mask].astype(np.int32, copy=False)
    hits = []
    for start in range(0, ids.size, _GEOM_CHUNK):
        end = min(start + _GEOM_CHUNK, ids.size)
        mask = np.all(mins[start:end] >= lo, axis=1) & np.all(maxs[start:end] <= hi, axis=1)
        if np.any(mask):
            hits.append(ids[start:end][mask])
    return np.concatenate(hits).astype(np.int32, copy=False) if hits else np.zeros((0,), dtype=np.int32)


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
    global_min = data.global_bbox_min if data.global_bbox_min.size == 3 else np.min(all_min, axis=0)
    global_max = data.global_bbox_max if data.global_bbox_max.size == 3 else np.max(all_max, axis=0)

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
        bx, by, bz = _bucket_of(pt)
        candidate_chunks = []
        # Prefer the compact CSR-like bucket index built by current runtimes.
        if data.spatial_bucket_keys.size and data.spatial_bucket_offsets.size:
            keys = data.spatial_bucket_keys
            offsets = data.spatial_bucket_offsets
            bucket_cids = data.spatial_bucket_cell_ids
            for ix in (bx - 1, bx, bx + 1):
                if ix < 0 or ix >= dx:
                    continue
                for iy in (by - 1, by, by + 1):
                    if iy < 0 or iy >= dy:
                        continue
                    for iz in (bz - 1, bz, bz + 1):
                        if iz < 0 or iz >= dz:
                            continue
                        key = (int(ix) * int(dy) + int(iy)) * int(dz) + int(iz)
                        pos = int(np.searchsorted(keys, key))
                        if pos < keys.size and int(keys[pos]) == key:
                            a, b = int(offsets[pos]), int(offsets[pos + 1])
                            if b > a:
                                candidate_chunks.append(bucket_cids[a:b])
        elif data.spatial_buckets:
            # Compatibility path for ad-hoc/test meshes built by older callers.
            for ix in (bx - 1, bx, bx + 1):
                if ix < 0 or ix >= dx:
                    continue
                for iy in (by - 1, by, by + 1):
                    if iy < 0 or iy >= dy:
                        continue
                    for iz in (bz - 1, bz, bz + 1):
                        if iz < 0 or iz >= dz:
                            continue
                        vals = data.spatial_buckets.get((ix, iy, iz), ())
                        if vals:
                            candidate_chunks.append(np.asarray(vals, dtype=np.int32))

        if candidate_chunks:
            candidates = candidate_chunks[0] if len(candidate_chunks) == 1 else np.concatenate(candidate_chunks)
            pos = np.searchsorted(all_ids, candidates)
            valid = (pos >= 0) & (pos < all_ids.size)
            clipped = np.clip(pos, 0, max(all_ids.size - 1, 0))
            valid &= all_ids[clipped] == candidates
            if np.any(valid):
                cids = candidates[valid]
                pidx = pos[valid]
                inside = np.all(all_min[pidx] - eps <= pt, axis=1) & np.all(all_max[pidx] + eps >= pt, axis=1)
                if np.any(inside):
                    hit = int(cids[int(np.flatnonzero(inside)[0])])

        if hit < 0:
            # Rare fallback for cells whose AABB is much larger than the
            # centroid bucket. Scan in bounded chunks and stop at the first hit
            # instead of allocating whole-mesh temporary boolean matrices.
            for start in range(0, all_ids.size, _GEOM_CHUNK):
                end = min(start + _GEOM_CHUNK, all_ids.size)
                mask = np.all(all_min[start:end] - eps <= pt, axis=1) & np.all(all_max[start:end] + eps >= pt, axis=1)
                idx = np.flatnonzero(mask)
                if idx.size:
                    hit = int(all_ids[start + int(idx[0])])
                    break
        if hit >= 0:
            out.append(hit)
    return np.asarray(out, dtype=np.int32)


def iotdb_line_intersection(
    data: RuntimeMeshData, line_start: Sequence[float], line_end: Sequence[float], eps: float = 1e-9
) -> np.ndarray:
    """Vectorized slab test for a segment against every cell AABB.

    Large meshes are processed in bounded chunks to avoid temporary arrays
    large enough to trigger swapping.
    """
    ids, mins, maxs = _ensure_bbox_arrays(data)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int32)
    p0 = np.asarray(line_start, dtype=np.float64).reshape(3)
    p1 = np.asarray(line_end, dtype=np.float64).reshape(3)
    d = p1 - p0
    if ids.size <= _GEOM_CHUNK:
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

    hit_ids = []
    hit_t = []
    for start_idx in range(0, ids.size, _GEOM_CHUNK):
        end_idx = min(start_idx + _GEOM_CHUNK, ids.size)
        cmins = mins[start_idx:end_idx]
        cmaxs = maxs[start_idx:end_idx]
        n = end_idx - start_idx
        tmin = np.zeros(n, dtype=np.float64)
        tmax = np.ones(n, dtype=np.float64)
        valid = np.ones(n, dtype=bool)
        for axis in range(3):
            if abs(float(d[axis])) < eps:
                valid &= (p0[axis] >= cmins[:, axis] - eps) & (p0[axis] <= cmaxs[:, axis] + eps)
                continue
            inv = 1.0 / float(d[axis])
            ta = (cmins[:, axis] - p0[axis]) * inv
            tb = (cmaxs[:, axis] - p0[axis]) * inv
            lo = np.minimum(ta, tb)
            hi = np.maximum(ta, tb)
            tmin = np.maximum(tmin, lo)
            tmax = np.minimum(tmax, hi)
            valid &= tmin <= tmax + eps
        idx = np.flatnonzero(valid)
        if idx.size:
            hit_ids.append(ids[start_idx:end_idx][idx])
            hit_t.append(tmin[idx])
    if not hit_ids:
        return np.zeros((0,), dtype=np.int32)
    out_ids = hit_ids[0] if len(hit_ids) == 1 else np.concatenate(hit_ids)
    out_t = hit_t[0] if len(hit_t) == 1 else np.concatenate(hit_t)
    order = np.argsort(out_t, kind="stable")
    return out_ids[order].astype(np.int32, copy=False)


def iotdb_plane_intersection(
    data: RuntimeMeshData, plane_origin: Sequence[float], plane_norm: Sequence[float], eps: float = 1e-9
) -> np.ndarray:
    """Vectorized plane/AABB overlap using cached center-radius projection."""
    ids, mins, maxs = _ensure_bbox_arrays(data)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int32)
    origin = np.asarray(plane_origin, dtype=np.float64).reshape(3)
    normal = np.asarray(plane_norm, dtype=np.float64).reshape(3)
    nlen = float(np.linalg.norm(normal))
    if nlen < eps:
        return np.zeros((0,), dtype=np.int32)
    normal = normal / nlen
    centers = data.all_bbox_center if data.all_bbox_center.shape == mins.shape else 0.5 * (mins + maxs)
    extents = data.all_bbox_extent if data.all_bbox_extent.shape == mins.shape else 0.5 * (maxs - mins)
    origin_dot = float(origin @ normal)
    abs_normal = np.abs(normal)
    if ids.size <= _GEOM_CHUNK:
        signed = centers @ normal - origin_dot
        radius = extents @ abs_normal
        mask = np.abs(signed) <= radius + eps
        return ids[mask].astype(np.int32, copy=False)

    hits = []
    for start_idx in range(0, ids.size, _GEOM_CHUNK):
        end_idx = min(start_idx + _GEOM_CHUNK, ids.size)
        signed = centers[start_idx:end_idx] @ normal - origin_dot
        radius = extents[start_idx:end_idx] @ abs_normal
        mask = np.abs(signed) <= radius + eps
        if np.any(mask):
            hits.append(ids[start_idx:end_idx][mask])
    return np.concatenate(hits).astype(np.int32, copy=False) if hits else np.zeros((0,), dtype=np.int32)
