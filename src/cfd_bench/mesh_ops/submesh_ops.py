from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from cfd_bench.core.runtime_mesh import RuntimeMeshData
from cfd_bench.core.types import LiteMesh


def iotdb_extract_submesh(data: RuntimeMeshData, cell_indexes: Sequence[int]) -> LiteMesh:
    picked = [int(c) for c in cell_indexes if int(c) in data.cells]
    picked_set = set(picked)

    node_ids = set()
    for cid in picked:
        for nid in data.cell_nodes.get(cid, []):
            node_ids.add(int(nid))

    node_xyz = {nid: data.nodes[nid] for nid in node_ids if nid in data.nodes}
    cell_nodes = {cid: [int(n) for n in data.cell_nodes.get(cid, [])] for cid in picked}
    cell_bbox = {cid: data.cell_bbox[cid] for cid in picked if cid in data.cell_bbox}

    return LiteMesh(
        cell_ids=np.array(sorted(picked_set), dtype=np.int32),
        node_xyz=node_xyz,
        cell_nodes=cell_nodes,
        cell_bbox=cell_bbox,
    )
