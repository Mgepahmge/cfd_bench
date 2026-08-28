"""Structure-to-CFD coupling feature."""

from .alignment import AlignmentDiagnostics, SimilarityTransform, estimate_similarity_alignment
from .engine import CouplingSummary, StructureCfdCouplingEngine
from .output import (
    STATUS_CODES,
    STATUS_INTERPOLATION_FAILED,
    STATUS_NO_CONTAINING_CELL,
    STATUS_OUTSIDE_MESH,
    STATUS_PASS,
)

__all__ = [
    "AlignmentDiagnostics",
    "SimilarityTransform",
    "estimate_similarity_alignment",
    "CouplingSummary",
    "StructureCfdCouplingEngine",
    "STATUS_CODES",
    "STATUS_PASS",
    "STATUS_OUTSIDE_MESH",
    "STATUS_NO_CONTAINING_CELL",
    "STATUS_INTERPOLATION_FAILED",
]
