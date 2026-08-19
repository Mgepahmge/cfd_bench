"""Random geometry helpers for workloads."""

from __future__ import annotations

import random
import time
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def random_points_in_bbox(bounds: Sequence[float], count: int = None) -> NDArray[np.float64]:
    """Generate random points within axis-aligned bbox [xmin,xmax,ymin,ymax,zmin,zmax]."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    n = count if count is not None else random.randint(1, 100)
    pts = np.random.rand(n, 3)
    pts[:, 0] = pts[:, 0] * (xmax - xmin) + xmin
    pts[:, 1] = pts[:, 1] * (ymax - ymin) + ymin
    pts[:, 2] = pts[:, 2] * (zmax - zmin) + zmin
    return pts


def random_line_in_bbox(bounds: Sequence[float]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    start = np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(zmin, zmax)])
    end = np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(zmin, zmax)])
    return start, end


def random_plane_in_bbox(bounds: Sequence[float]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    origin = np.array([random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(zmin, zmax)])
    normal = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
    return origin, normal


def random_coord_range(bounds: Sequence[float]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Random axis-aligned box as (lower_corner, upper_corner)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x1, x2 = sorted([random.uniform(xmin, xmax), random.uniform(xmin, xmax)])
    y1, y2 = sorted([random.uniform(ymin, ymax), random.uniform(ymin, ymax)])
    z1, z2 = sorted([random.uniform(zmin, zmax), random.uniform(zmin, zmax)])
    return np.array([x1, y1, z1], dtype=np.float64), np.array([x2, y2, z2], dtype=np.float64)


def random_start_point(intersect_fn, bounds: Sequence[float], *, deadline: float | None = None, max_attempts: int | None = None):
    """Random point with exactly one containing cell.

    ``deadline`` is an optional ``time.monotonic()`` deadline used by
    long-running workloads so a sparse/degenerate mesh cannot trap the
    benchmark forever while searching for a valid start point.
    """
    attempts = 0
    while deadline is None or time.monotonic() < deadline:
        if max_attempts is not None and attempts >= int(max_attempts):
            raise LookupError("no mesh hit after bounded start-point retries")
        attempts += 1
        pt = random_points_in_bbox(bounds, count=1)[0]
        cells = intersect_fn(np.array([pt], dtype=np.float64))
        if len(cells) == 1:
            return int(cells[0]), pt
    raise TimeoutError("timed out while searching for a streamline start point")
