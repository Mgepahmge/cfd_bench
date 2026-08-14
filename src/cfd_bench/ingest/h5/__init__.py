"""ODB-like HDF5 ingest support."""

from .artifacts import max_neighbor_diffs, write_max_diff_files
from .canonical import (
    BENCHMARK_SCALAR_VARS,
    BENCHMARK_VECTOR_VARS,
    available_mapping,
    build_canonical_frame,
    field_to_cells,
)
from .model import CanonicalFrame, CanonicalMesh, ElementBlock, FieldInfo, FrameInfo, H5IngestPlan
from .reader import OdbH5Reader

__all__ = [
    "BENCHMARK_SCALAR_VARS",
    "BENCHMARK_VECTOR_VARS",
    "CanonicalFrame",
    "CanonicalMesh",
    "ElementBlock",
    "FieldInfo",
    "FrameInfo",
    "H5IngestPlan",
    "OdbH5Reader",
    "available_mapping",
    "build_canonical_frame",
    "field_to_cells",
    "max_neighbor_diffs",
    "write_max_diff_files",
]
