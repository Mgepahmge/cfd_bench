"""Derived artifacts needed by benchmark workloads after HDF5 conversion."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from .model import CanonicalFrame, CanonicalMesh


def max_neighbor_diffs(mesh: CanonicalMesh, values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) != mesh.cell_count:
        raise ValueError(f"values={len(arr)} but mesh cells={mesh.cell_count}")
    maximum = 0.0
    found = False
    for cid, neighbors in enumerate(mesh.cell_adjacency):
        for nid in neighbors:
            if nid <= cid:
                continue
            a, b = arr[cid], arr[nid]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            maximum = max(maximum, abs(float(a) - float(b)))
            found = True
    if found:
        return float(maximum)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return 0.0
    # A mesh with no adjacency (e.g. disconnected elements) still gets a safe
    # non-negative search width instead of making W3 fail at file lookup time.
    return float(np.max(finite) - np.min(finite))


def write_max_diff_files(
    output_dir: str,
    dataset_key: str,
    mesh: CanonicalMesh,
    frames: Sequence[CanonicalFrame],
) -> Sequence[str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for frame in frames:
        path = root / f"{dataset_key}_{frame.timestep}_max_diffs.csv"
        rows = [
            (var, max_neighbor_diffs(mesh, values))
            for var, values in sorted(frame.cell_scalars.items())
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["variable", "max_diff"])
            writer.writerows(rows)
        written.append(str(path))
    return written
