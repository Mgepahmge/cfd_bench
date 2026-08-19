"""Runtime-only nodal projection helpers for legacy Tecplot CFD datasets.

Legacy CFD DAT files store X/Y/Z at nodes but U/V/W/P/K/E at cell centers.
W11 needs point extrema across frames.  This module derives point values at
query time by averaging incident cell-centered values.  Nothing is persisted,
so the CFD ingest contract remains unchanged and the frozen H5 path is not
involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class NodeCellCSR:
    """Compact reverse node -> incident-cell mapping."""

    node_count: int
    offsets: np.ndarray
    cell_ids: np.ndarray

    def incident_cells(self, dense_node_id: int) -> np.ndarray:
        nid = int(dense_node_id)
        if nid < 0 or nid >= int(self.node_count):
            return np.zeros((0,), dtype=np.int32)
        start = int(self.offsets[nid])
        end = int(self.offsets[nid + 1])
        return self.cell_ids[start:end]


def build_node_cell_csr(
    cell_nodes: Mapping[int, Sequence[int]], node_count: int
) -> NodeCellCSR:
    """Invert cell connectivity without retaining a Python node->cells dict."""

    node_count = max(0, int(node_count))
    if not cell_nodes or node_count <= 0:
        return NodeCellCSR(
            node_count=node_count,
            offsets=np.zeros((node_count + 1,), dtype=np.int64),
            cell_ids=np.zeros((0,), dtype=np.int32),
        )

    # Preserve mapping iteration order; reverse connectivity does not depend
    # on cell ordering, so avoid an unnecessary O(C log C) sort here.
    items = [(int(cid), nodes) for cid, nodes in cell_nodes.items()]
    lengths = np.fromiter((len(nodes) for _, nodes in items), dtype=np.int64, count=len(items))
    total = int(lengths.sum())
    if total <= 0:
        return NodeCellCSR(
            node_count=node_count,
            offsets=np.zeros((node_count + 1,), dtype=np.int64),
            cell_ids=np.zeros((0,), dtype=np.int32),
        )

    dense_nodes = np.empty((total,), dtype=np.int64)
    dense_cells = np.empty((total,), dtype=np.int32)
    pos = 0
    for (cid, nodes), length in zip(items, lengths):
        n = int(length)
        if n <= 0:
            continue
        dense_nodes[pos : pos + n] = np.asarray(nodes, dtype=np.int64)
        dense_cells[pos : pos + n] = int(cid)
        pos += n
    if pos != total:
        dense_nodes = dense_nodes[:pos]
        dense_cells = dense_cells[:pos]

    return build_node_cell_csr_from_incidence(dense_nodes, dense_cells, node_count)


def build_node_cell_csr_from_incidence(
    node_ids: Sequence[int] | np.ndarray,
    cell_ids: Sequence[int] | np.ndarray,
    node_count: int,
) -> NodeCellCSR:
    """Build CSR from flat ``(node_id, cell_id)`` incidence arrays."""

    node_count = max(0, int(node_count))
    dense_nodes = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    dense_cells = np.asarray(cell_ids, dtype=np.int32).reshape(-1)
    if dense_nodes.size != dense_cells.size:
        raise ValueError("node/cell incidence arrays must have the same length")
    valid = (dense_nodes >= 0) & (dense_nodes < node_count)
    dense_nodes = dense_nodes[valid]
    dense_cells = dense_cells[valid]
    if dense_nodes.size == 0:
        return NodeCellCSR(
            node_count=node_count,
            offsets=np.zeros((node_count + 1,), dtype=np.int64),
            cell_ids=np.zeros((0,), dtype=np.int32),
        )

    order = np.argsort(dense_nodes, kind="stable")
    dense_nodes = dense_nodes[order]
    dense_cells = dense_cells[order]
    counts = np.bincount(dense_nodes, minlength=node_count).astype(np.int64, copy=False)
    offsets = np.empty((node_count + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return NodeCellCSR(node_count=node_count, offsets=offsets, cell_ids=dense_cells)


def point_frame_extrema_from_cell_values(
    csr: NodeCellCSR,
    source_point_ids: Sequence[int],
    steps: Sequence[int],
    fetch_values: Callable[[int, np.ndarray], np.ndarray],
) -> Dict[int, Tuple[float, float]]:
    """Project cell values to Tecplot points and compute extrema across frames.

    Tecplot node IDs are implicit and one-based.  The canonical runtime stores
    dense zero-based node IDs, so source ID ``n`` maps to dense ID ``n-1``.
    ``fetch_values(step, cell_ids)`` must return values aligned to ``cell_ids``.
    """

    requested = []
    seen = set()
    for raw in source_point_ids:
        source_id = int(raw)
        if source_id in seen:
            continue
        seen.add(source_id)
        dense_id = source_id - 1
        if 0 <= dense_id < int(csr.node_count):
            requested.append((source_id, dense_id))
    if not requested:
        return {}

    edge_nodes = []
    edge_cells = []
    active_source_ids = []
    for source_id, dense_id in requested:
        cells = csr.incident_cells(dense_id)
        if cells.size == 0:
            continue
        active_idx = len(active_source_ids)
        active_source_ids.append(source_id)
        edge_nodes.append(np.full(cells.size, active_idx, dtype=np.int32))
        edge_cells.append(np.asarray(cells, dtype=np.int32))
    if not edge_cells:
        return {}

    edge_node_idx = np.concatenate(edge_nodes)
    edge_cell_ids = np.concatenate(edge_cells)
    unique_cells, edge_cell_pos = np.unique(edge_cell_ids, return_inverse=True)
    mins = np.full((len(active_source_ids),), np.inf, dtype=np.float64)
    maxs = np.full((len(active_source_ids),), -np.inf, dtype=np.float64)

    for step in sorted(set(int(s) for s in steps)):
        values = np.asarray(fetch_values(step, unique_cells), dtype=np.float64).reshape(-1)
        if values.size != unique_cells.size:
            raise ValueError(
                f"cell value fetch returned {values.size} values for {unique_cells.size} cells"
            )
        edge_values = values[edge_cell_pos]
        finite = np.isfinite(edge_values)
        if not np.any(finite):
            continue
        node_idx = edge_node_idx[finite]
        sums = np.bincount(
            node_idx,
            weights=edge_values[finite],
            minlength=len(active_source_ids),
        )
        counts = np.bincount(node_idx, minlength=len(active_source_ids))
        valid_nodes = counts > 0
        projected = np.full((len(active_source_ids),), np.nan, dtype=np.float64)
        projected[valid_nodes] = sums[valid_nodes] / counts[valid_nodes]
        mins[valid_nodes] = np.minimum(mins[valid_nodes], projected[valid_nodes])
        maxs[valid_nodes] = np.maximum(maxs[valid_nodes], projected[valid_nodes])

    return {
        int(source_id): (float(mins[i]), float(maxs[i]))
        for i, source_id in enumerate(active_source_ids)
        if np.isfinite(mins[i]) and np.isfinite(maxs[i])
    }
