"""Unified mesh client API protocol."""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from cfd_bench.core.context import MeshContext
from cfd_bench.core.types import LiteMesh, LitePolyData


@runtime_checkable
class MeshClient(Protocol):
    """Common method set for IoTDB, PostgreSQL, TileDB, and VTK baseline backends."""

    def connect(
        self,
        dataset_key: str,
        step: int,
        zone: str = "0_Fluid",
        **kwargs,
    ) -> MeshContext:
        ...

    def point_query(
        self,
        cell_indexes: Sequence[int],
        attribute_name: str,
        step: Optional[int] = None,
    ) -> NDArray[np.float64]:
        ...

    def range_query_var(
        self,
        lower_bound: float,
        upper_bound: float,
        attribute_name: str,
        step: Optional[int] = None,
    ) -> NDArray[np.int32]:
        ...

    def range_query_coord(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
    ) -> NDArray[np.int32]:
        ...

    def point_intersection(self, points: NDArray[np.float64]) -> NDArray[np.int32]:
        ...

    def line_intersection(
        self,
        line_start: Sequence[float],
        line_end: Sequence[float],
    ) -> NDArray[np.int32]:
        ...

    def plane_intersection(
        self,
        plane_origin: Sequence[float],
        plane_norm: Sequence[float],
    ) -> NDArray[np.int32]:
        ...

    def extract_submesh(
        self,
        cell_indexes: Sequence[int],
        mesh_handle=None,
    ) -> LiteMesh:
        ...

    def isosurface_extraction(
        self,
        variable_name: str,
        iso_value: float,
        step: Optional[int] = None,
    ) -> LitePolyData:
        ...

    def surface_norm(self, mesh_handle=None) -> NDArray[np.float64]:
        ...

    def compute_qcriterion_roi(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        tau: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
        ...


# Workload capability requirements (W1–W8)
WORKLOAD_CAPS = {
    "w1": {"mesh_static", "cell_vars"},
    "w2": {"cell_vars"},
    "w3": {"mesh_static", "cell_vars"},
    "w4": {"cell_vars"},
    "w5": {"cell_vars"},
    "w6": {"mesh_static"},
    "w7": {"mesh_static", "cell_vars"},
    "w8": {"cell_vars", "cell_qcriterion"},
}
