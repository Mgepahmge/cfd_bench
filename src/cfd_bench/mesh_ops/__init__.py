from cfd_bench.mesh_ops.geometry_ops import (
    iotdb_line_intersection,
    iotdb_plane_intersection,
    iotdb_point_intersection,
)
from cfd_bench.mesh_ops.isosurface_ops import iotdb_isosurface_extraction
from cfd_bench.mesh_ops.normals import iotdb_surface_norm, iotdb_surface_norm_from_mesh
from cfd_bench.mesh_ops.submesh_ops import iotdb_extract_submesh
from cfd_bench.mesh_ops.gradient_ops import compute_qcriterion_roi

# Backend-agnostic aliases (TileDB clients use same algorithms)
tiledb_point_intersection = iotdb_point_intersection
tiledb_line_intersection = iotdb_line_intersection
tiledb_plane_intersection = iotdb_plane_intersection
tiledb_extract_submesh = iotdb_extract_submesh
tiledb_isosurface_extraction = iotdb_isosurface_extraction
tiledb_surface_norm = iotdb_surface_norm
tiledb_surface_norm_from_mesh = iotdb_surface_norm_from_mesh

__all__ = [
    "iotdb_point_intersection",
    "iotdb_line_intersection",
    "iotdb_plane_intersection",
    "iotdb_extract_submesh",
    "iotdb_isosurface_extraction",
    "iotdb_surface_norm",
    "iotdb_surface_norm_from_mesh",
    "tiledb_point_intersection",
    "tiledb_line_intersection",
    "tiledb_plane_intersection",
    "tiledb_extract_submesh",
    "tiledb_isosurface_extraction",
    "tiledb_surface_norm",
    "tiledb_surface_norm_from_mesh",
    "compute_qcriterion_roi",
]
