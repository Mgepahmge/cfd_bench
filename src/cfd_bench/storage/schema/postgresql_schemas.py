"""PostgreSQL table and column names for modern CFD-Bench schema."""

from __future__ import annotations

MESH_NODES = "mesh_nodes"
MESH_CELLS = "mesh_cells"
MESH_CELL_NODES = "mesh_cell_nodes"
MESH_CELL_ADJACENCY = "mesh_cell_adjacency"
MESH_FACE_PLANES = "mesh_face_planes"
MESH_BOUNDARY_FACES = "mesh_boundary_faces"

CELL_SCALAR = "cell_scalar"
NODE_SCALAR = "node_scalar"
CELL_QCRITERION = "cell_qcriterion"
CELL_GRADIENT = "cell_gradient"

# PostGIS spatial acceleration
CELL_GEOM_FULL = "cell_geom_full"
POINT_LOCATOR_GRID = "point_locator_grid"
BOUNDARY_FACE_GEOM = "boundary_face_geom"

# Legacy mesh tables (benchmark_1 compatible)
MESH_METADATA = "mesh_metadata"
CELL_CENTROID = "cell_centroid"
CELL_FACE_PLANE = "cell_face_plane"
CELL_ADJACENCY_LEGACY = "cell_adjacency"
NODE_COORDINATES = "node_coordinates"
CELL_NODES_LEGACY = "cell_nodes"
