"""Modern logical data domains shared across IoTDB, TileDB, and PostgreSQL."""

from __future__ import annotations

from typing import FrozenSet, Tuple

# mesh_static/{zone}/ — topology (W1 geometry, W3 submesh, W6 normals, W7 gradient)
MESH_STATIC_LEAVES: Tuple[str, ...] = (
    "mesh_meta",
    "nodes",
    "cells",
    "cell_nodes",
    "cell_adjacency",
    "face_planes",
    "boundary_faces",
)

# post_processing/step_{t}/ — per-timestep scalars (W1–W5, W8)
POST_PROCESSING_LEAVES: Tuple[str, ...] = (
    "cell_vars",
    "cell_vars_hull",
    "node_vars",
)

# derived/step_{t}/ — materialized derived fields (W7 online Q, W8 vortex)
DERIVED_LEAVES: Tuple[str, ...] = (
    "cell_qcriterion",
    "cell_gradient",
)

MODERN_CAPS: FrozenSet[str] = frozenset(
    {"mesh_static", "cell_vars", "node_vars", "cell_qcriterion", "cell_gradient"}
)
