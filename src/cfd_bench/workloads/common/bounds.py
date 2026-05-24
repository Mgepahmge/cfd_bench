"""Mesh bounding-box helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def bbox_from_cell_bboxes(cell_bboxes: dict) -> Tuple[List[float], List[float]]:
    bboxes = list(cell_bboxes.values())
    if not bboxes:
        return [0, 0, 0], [0, 0, 0]
    gmin = [min(b[i] for b in bboxes) for i in (0, 2, 4)]
    gmax = [max(b[i] for b in bboxes) for i in (1, 3, 5)]
    return gmin, gmax


def flat_bounds(gmin: Sequence[float], gmax: Sequence[float]) -> List[float]:
    return [gmin[0], gmax[0], gmin[1], gmax[1], gmin[2], gmax[2]]
