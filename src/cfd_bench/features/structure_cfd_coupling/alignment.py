"""Optional uniform 3-D similarity alignment for structure/CFD coupling.

The alignment is deliberately opt-in.  It estimates one global transform

    p_cfd = scale * rotation @ p_structure + translation

with a *single* scale factor for all three axes.  The implementation uses a
robust, trimmed similarity ICP.  Umeyama's closed-form similarity solve is
used at every ICP iteration.  No source coordinates are modified in IoTDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return float(self.scale) * (pts @ np.asarray(self.rotation, dtype=np.float64).T) + np.asarray(
            self.translation, dtype=np.float64
        )


@dataclass(frozen=True)
class AlignmentDiagnostics:
    transform: SimilarityTransform
    method: str
    structure_sample_count: int
    cfd_sample_count: int
    iterations: int
    initial_scale: float
    rms_scale: float
    principal_scale: float
    principal_extent_scale: float
    scale_consistency: float
    rmse_before: float
    rmse_after: float
    median_error: float
    p95_error: float
    inlier_fraction: float
    confidence: str


class _NearestIndex:
    """Nearest-neighbour helper with an optional SciPy fast path.

    SciPy is intentionally not a mandatory coupling dependency.  Minimal
    IoTDB+H5 installations therefore remain valid; their fallback uses
    chunked NumPy distance evaluation on a smaller deterministic sample.
    """

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self._tree = None
        try:  # pragma: no cover - availability depends on installation extras
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.points)
        except Exception:
            self._tree = None

    @property
    def fast(self) -> bool:
        return self._tree is not None

    def query(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        query = np.asarray(points, dtype=np.float64)
        if self._tree is not None:
            distance, index = self._tree.query(query, k=1, workers=1)
            return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.int64)

        # Keep peak memory bounded.  This is exact, but intentionally used on
        # a capped sample when SciPy is absent.
        best_d2 = np.full((query.shape[0],), np.inf, dtype=np.float64)
        best_idx = np.full((query.shape[0],), -1, dtype=np.int64)
        query_chunk = 256
        target_chunk = 2048
        for qs in range(0, query.shape[0], query_chunk):
            qe = min(qs + query_chunk, query.shape[0])
            q = query[qs:qe]
            local_d2 = np.full((q.shape[0],), np.inf, dtype=np.float64)
            local_idx = np.full((q.shape[0],), -1, dtype=np.int64)
            for ts in range(0, self.points.shape[0], target_chunk):
                te = min(ts + target_chunk, self.points.shape[0])
                t = self.points[ts:te]
                diff = q[:, None, :] - t[None, :, :]
                d2 = np.einsum("qti,qti->qt", diff, diff, optimize=True)
                arg = np.argmin(d2, axis=1)
                val = d2[np.arange(q.shape[0]), arg]
                improve = val < local_d2
                local_d2[improve] = val[improve]
                local_idx[improve] = ts + arg[improve]
            best_d2[qs:qe] = local_d2
            best_idx[qs:qe] = local_idx
        return np.sqrt(best_d2), best_idx


def _finite_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} point array must have shape (N, 3); got {arr.shape}")
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] < 4:
        raise ValueError(f"{name} point cloud needs at least 4 finite points")
    return arr


def _deterministic_sample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    n = int(points.shape[0])
    if n <= int(max_points):
        return np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
    return np.asarray(points[idx], dtype=np.float64)



def _surface_proxy(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Extract an outer-envelope proxy from a possibly volumetric structure mesh.

    Structural H5 meshes often contain internal framing while the CFD reference
    is a hull/wetted surface.  A small axis-envelope proxy keeps min/max points
    in 2-D bins for each principal coordinate direction.  It does not modify the
    actual coupling nodes; it is used only to estimate the optional transform.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= 64:
        return pts
    pre_limit = max(int(max_points) * 8, int(max_points))
    pts = _deterministic_sample(pts, min(pre_limit, pts.shape[0]), seed)
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    span = hi - lo
    valid_span = span > max(float(np.max(span)) * 1.0e-12, 1.0e-15)
    if np.count_nonzero(valid_span) < 2:
        return _deterministic_sample(pts, max_points, seed + 1)

    # Roughly 6*bins^2 envelope points before de-duplication.
    bins = int(np.clip(np.sqrt(max(float(max_points), 100.0) / 6.0), 8, 64))
    norm = np.zeros_like(pts)
    norm[:, valid_span] = (pts[:, valid_span] - lo[valid_span]) / span[valid_span]
    ibin = np.minimum((norm * bins).astype(np.int64), bins - 1)
    selected = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        key = ibin[:, other[0]] * bins + ibin[:, other[1]]
        order = np.lexsort((pts[:, axis], key))
        sorted_key = key[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
        ends = np.r_[starts[1:] - 1, order.size - 1]
        selected.append(order[starts])
        selected.append(order[ends])
    idx = np.unique(np.concatenate(selected))
    proxy = pts[idx]
    return _deterministic_sample(proxy, max_points, seed + 2)

def _trim_radial_outliers(points: np.ndarray, keep_fraction: float = 0.98) -> np.ndarray:
    if points.shape[0] < 32:
        return points
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center[None, :], axis=1)
    cutoff = float(np.quantile(radius, float(keep_fraction)))
    kept = points[radius <= cutoff]
    return kept if kept.shape[0] >= 4 else points


def _principal_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    centered = points - center[None, :]
    covariance = centered.T @ centered / float(max(1, points.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order]
    std = np.sqrt(eigenvalues)
    projected = centered @ basis
    extents = np.ptp(projected, axis=0)
    return center, basis, std, extents


def _safe_ratios(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    eps = max(float(np.max(np.abs(source))) * 1.0e-12, 1.0e-15)
    mask = (source > eps) & np.isfinite(source) & np.isfinite(target) & (target > 0.0)
    return target[mask] / source[mask]


def _initial_scale(source: np.ndarray, target: np.ndarray):
    source_center, source_basis, source_std, source_extents = _principal_frame(source)
    target_center, target_basis, target_std, target_extents = _principal_frame(target)

    source_centered = source - source_center[None, :]
    target_centered = target - target_center[None, :]
    source_rms = float(np.sqrt(np.mean(np.sum(source_centered * source_centered, axis=1))))
    target_rms = float(np.sqrt(np.mean(np.sum(target_centered * target_centered, axis=1))))
    if source_rms <= 0.0 or not np.isfinite(source_rms) or not np.isfinite(target_rms):
        raise ValueError("cannot estimate alignment scale from degenerate point cloud")
    rms_scale = target_rms / source_rms

    principal_ratios = _safe_ratios(target_std, source_std)
    principal_scale = float(np.median(principal_ratios)) if principal_ratios.size else rms_scale
    extent_ratios = _safe_ratios(target_extents, source_extents)
    principal_extent_scale = float(np.median(extent_ratios)) if extent_ratios.size else rms_scale

    candidates = np.asarray(
        [rms_scale, principal_scale, principal_extent_scale], dtype=np.float64
    )
    candidates = candidates[np.isfinite(candidates) & (candidates > 0.0)]
    if candidates.size == 0:
        raise ValueError("cannot estimate a finite positive uniform alignment scale")
    scale = float(np.median(candidates))

    if principal_ratios.size >= 2:
        scale_consistency = float(np.max(principal_ratios) / max(np.min(principal_ratios), 1.0e-15))
    else:
        scale_consistency = 1.0
    return (
        scale,
        rms_scale,
        principal_scale,
        principal_extent_scale,
        scale_consistency,
        source_center,
        source_basis,
        target_center,
        target_basis,
    )


def _umeyama(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Closed-form least-squares similarity transform, proper rotation only."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("paired point arrays must both have shape (N, 3)")
    n = int(source.shape[0])
    if n < 4:
        raise ValueError("similarity solve requires at least 4 point pairs")

    src_mean = np.mean(source, axis=0)
    dst_mean = np.mean(target, axis=0)
    src = source - src_mean[None, :]
    dst = target - dst_mean[None, :]
    variance = float(np.sum(src * src) / float(n))
    if variance <= 1.0e-30:
        raise ValueError("degenerate source point cloud in similarity solve")

    covariance = (dst.T @ src) / float(n)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones((3,), dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular * sign) / variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("similarity solve produced an invalid scale")
    translation = dst_mean - scale * (rotation @ src_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def _trimmed_indices(distances: np.ndarray, fraction: float, target_indices: np.ndarray) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(distances) & (target_indices >= 0))
    if valid.size < 4:
        return valid
    order = valid[np.argsort(distances[valid], kind="stable")]
    keep = max(4, int(np.ceil(float(fraction) * float(order.size))))
    order = order[:keep]

    # Reduce many-to-one collapse: for duplicated target correspondences keep
    # only the closest structure point.
    _, first = np.unique(target_indices[order], return_index=True)
    unique_order = order[np.sort(first)]
    return unique_order if unique_order.size >= 4 else order


def _rotation_candidates(
    source_basis: np.ndarray,
    target_basis: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    candidates = [np.eye(3, dtype=np.float64)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                d = np.diag([sx, sy, sz])
                rotation = target_basis @ d @ source_basis.T
                if np.linalg.det(rotation) < 0.0:
                    continue
                if not any(np.allclose(rotation, existing, atol=1.0e-10) for existing in candidates):
                    candidates.append(rotation)
    return tuple(candidates)


def _trimmed_rmse(distances: np.ndarray, fraction: float) -> float:
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return float("inf")
    keep = max(1, int(np.ceil(float(fraction) * float(finite.size))))
    selected = np.partition(finite, keep - 1)[:keep]
    return float(np.sqrt(np.mean(selected * selected)))


def _confidence(
    rmse: float,
    p95: float,
    target: np.ndarray,
    scale_consistency: float,
    inlier_fraction: float,
) -> str:
    span = np.ptp(target, axis=0)
    diagonal = max(float(np.linalg.norm(span)), 1.0e-15)
    nrmse = float(rmse) / diagonal
    np95 = float(p95) / diagonal
    # Final geometric fit is the strongest signal.  Initial PCA-scale
    # consistency is only a secondary diagnostic because a volumetric
    # structural mesh and a surface CFD reference naturally have different
    # point-density covariance even when the true similarity transform is exact.
    if nrmse <= 0.005 and np95 <= 0.02 and float(inlier_fraction) >= 0.50:
        return "high"
    if (
        nrmse <= 0.03
        and np95 <= 0.08
        and float(inlier_fraction) >= 0.35
        and scale_consistency <= 1.75
    ):
        return "medium"
    return "low"


def estimate_similarity_alignment(
    structure_points: np.ndarray,
    cfd_reference_points: np.ndarray,
    *,
    max_points: int = 10000,
    max_iterations: int = 30,
    trim_fraction: float = 0.80,
    seed: int = 0,
) -> AlignmentDiagnostics:
    """Estimate a robust uniform 3-D similarity transform.

    The point clouds do not need one-to-one node correspondence.  A PCA/RMS
    initialisation is followed by nearest-neighbour ICP; each iteration uses
    trimmed correspondences and Umeyama's similarity solve.  The scale is kept
    within a conservative factor of the initial estimate to prevent the
    well-known scale-collapse failure mode of unconstrained similarity ICP.
    """
    if int(max_points) < 100:
        raise ValueError("alignment max_points must be >= 100")
    if int(max_iterations) < 1:
        raise ValueError("alignment max_iterations must be >= 1")
    if not 0.25 <= float(trim_fraction) <= 1.0:
        raise ValueError("alignment trim_fraction must be in [0.25, 1.0]")

    source_all = _finite_points(structure_points, "structure")
    target_all = _finite_points(cfd_reference_points, "CFD reference")

    # Remove only extreme radial outliers.  The structure point cloud may be a
    # full volumetric FE mesh, so estimate the transform from an outer-envelope
    # proxy rather than letting dense internal framing dominate the fit.
    source_all = _trim_radial_outliers(source_all)
    target_all = _trim_radial_outliers(target_all)

    requested_max = int(max_points)
    probe_target = _deterministic_sample(target_all, min(requested_max, 256), seed + 97)
    fast_probe = _NearestIndex(probe_target).fast
    effective_max = requested_max if fast_probe else min(requested_max, 3000)
    source = _surface_proxy(source_all, effective_max, seed)
    target = _deterministic_sample(target_all, effective_max, seed + 1)
    target_index = _NearestIndex(target)

    (
        initial_scale,
        rms_scale,
        principal_scale,
        principal_extent_scale,
        scale_consistency,
        source_center,
        source_basis,
        target_center,
        target_basis,
    ) = _initial_scale(source, target)

    # Choose the best PCA sign configuration (plus identity) before ICP.
    scored_candidates = []
    for rotation in _rotation_candidates(source_basis, target_basis):
        translation = target_center - initial_scale * (rotation @ source_center)
        candidate = SimilarityTransform(initial_scale, rotation, translation)
        distances, _ = target_index.query(candidate.apply(source))
        score = _trimmed_rmse(distances, trim_fraction)
        # Ship-like geometry can be nearly symmetric, so several PCA sign
        # configurations may have virtually identical NN error.  Prefer the
        # candidate closest to the as-ingested orientation when scores are
        # effectively tied instead of introducing an arbitrary 180-degree flip.
        trace_value = float(np.trace(rotation))
        angle = float(np.arccos(np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)))
        scored_candidates.append((float(score), angle, candidate))
    if not scored_candidates:
        raise RuntimeError("could not construct an initial similarity transform")
    raw_best = min(item[0] for item in scored_candidates)
    tie_limit = raw_best * 1.05 + 1.0e-15
    near_best = [item for item in scored_candidates if item[0] <= tie_limit]
    best_score, _best_angle, best_transform = min(near_best, key=lambda item: (item[1], item[0]))

    rmse_before = float(best_score)
    transform = best_transform
    previous_rmse = float("inf")
    iteration_count = 0
    scale_lo = initial_scale * 0.5
    scale_hi = initial_scale * 2.0
    target_diagonal = max(float(np.linalg.norm(np.ptp(target, axis=0))), 1.0e-12)

    for iteration in range(1, int(max_iterations) + 1):
        transformed = transform.apply(source)
        distances, target_ids = target_index.query(transformed)
        chosen = _trimmed_indices(distances, float(trim_fraction), target_ids)
        if chosen.size < 4:
            break
        updated = _umeyama(source[chosen], target[target_ids[chosen]])
        if updated.scale < scale_lo or updated.scale > scale_hi:
            clipped = min(max(float(updated.scale), scale_lo), scale_hi)
            updated = SimilarityTransform(clipped, updated.rotation, updated.translation)

        new_distances, _ = target_index.query(updated.apply(source))
        rmse = _trimmed_rmse(new_distances, trim_fraction)
        iteration_count = iteration

        scale_delta = abs(float(updated.scale) - float(transform.scale)) / max(
            abs(float(transform.scale)), 1.0e-15
        )
        rotation_delta = float(np.linalg.norm(updated.rotation - transform.rotation, ord="fro"))
        translation_delta = float(np.linalg.norm(updated.translation - transform.translation)) / target_diagonal
        rmse_delta = abs(float(previous_rmse) - float(rmse)) / max(float(rmse), target_diagonal * 1.0e-12)
        transform = updated

        if rmse <= target_diagonal * 1.0e-10:
            break
        if (
            scale_delta <= 1.0e-7
            and rotation_delta <= 1.0e-6
            and translation_delta <= 1.0e-7
            and rmse_delta <= 1.0e-5
        ):
            break
        previous_rmse = rmse

    final_distances, final_target_ids = target_index.query(transform.apply(source))
    final_chosen = _trimmed_indices(final_distances, float(trim_fraction), final_target_ids)
    if final_chosen.size < 4:
        raise RuntimeError("alignment produced too few valid point correspondences")
    selected_distances = final_distances[final_chosen]
    rmse_after = float(np.sqrt(np.mean(selected_distances * selected_distances)))
    median_error = float(np.median(selected_distances))
    p95_error = float(np.quantile(selected_distances, 0.95))
    inlier_fraction = float(final_chosen.size) / float(source.shape[0])
    confidence = _confidence(
        rmse_after, p95_error, target, scale_consistency, inlier_fraction
    )

    return AlignmentDiagnostics(
        transform=transform,
        method="trimmed-similarity-icp/umeyama",
        structure_sample_count=int(source.shape[0]),
        cfd_sample_count=int(target.shape[0]),
        iterations=int(iteration_count),
        initial_scale=float(initial_scale),
        rms_scale=float(rms_scale),
        principal_scale=float(principal_scale),
        principal_extent_scale=float(principal_extent_scale),
        scale_consistency=float(scale_consistency),
        rmse_before=float(rmse_before),
        rmse_after=float(rmse_after),
        median_error=float(median_error),
        p95_error=float(p95_error),
        inlier_fraction=float(inlier_fraction),
        confidence=str(confidence),
    )
