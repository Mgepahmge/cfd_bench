"""Canonical in-memory model used by the HDF5 ingest path.

The HDF5 files handled by this package are organized like an Abaqus ODB:
mesh topology lives under ``Parts`` while field output lives under
``Steps/<step>/Frames/<frame>`` and is keyed by assembly instance.

This module deliberately keeps that source-specific metadata out of the
PostgreSQL schema used by the benchmark.  The PostgreSQL adapter consumes the
canonical arrays below and assigns dense, zero-based node/cell identifiers so
that the existing W1-W8 code can address cells by ``range(n_cells)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ElementBlock:
    """One source element class inside a Part."""

    name: str
    element_type: str
    source_labels: np.ndarray
    connectivity: np.ndarray
    cell_ids: np.ndarray
    section_category: str = ""

    @property
    def count(self) -> int:
        return int(len(self.cell_ids))


@dataclass
class CanonicalMesh:
    """Dense benchmark mesh for one assembly instance."""

    part_name: str
    instance_name: str
    node_ids: np.ndarray
    source_node_labels: np.ndarray
    node_coordinates: np.ndarray
    element_blocks: List[ElementBlock]
    cell_ids: np.ndarray
    source_cell_labels: np.ndarray
    cell_node_ids: List[np.ndarray]
    cell_element_types: List[str]
    cell_centroids: np.ndarray
    cell_adjacency: List[List[int]]
    block_cell_ids: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return int(len(self.node_ids))

    @property
    def cell_count(self) -> int:
        return int(len(self.cell_ids))


@dataclass(frozen=True)
class FrameInfo:
    step_name: str
    step_index: int
    step_domain: Optional[int]
    frame_name: str
    frame_index: int
    inc_or_mode: Optional[int]
    time_or_frequency: Optional[float]
    description: str = ""
    load_case: str = ""


@dataclass(frozen=True)
class FieldInfo:
    name: str
    components: Tuple[str, ...]
    position_code: Optional[int]
    type_code: Optional[int]
    instances: Tuple[str, ...]


@dataclass
class CanonicalFrame:
    """Benchmark scalars derived from one HDF5 frame.

    ``cell_scalars`` keeps the representation consumed by W1-W10.  When the
    source HDF5 field is genuinely nodal, ``node_scalars`` preserves the
    original point-level values as well so H5-only workloads such as W11 do
    not have to reconstruct them from cell averages.
    """

    info: FrameInfo
    timestep: int
    cell_scalars: Dict[str, np.ndarray]
    node_scalars: Dict[str, np.ndarray] = field(default_factory=dict)
    source_fields: Dict[str, str] = field(default_factory=dict)

    def validate(self, cell_count: int, node_count: Optional[int] = None) -> None:
        for name, values in self.cell_scalars.items():
            arr = np.asarray(values)
            if arr.shape != (cell_count,):
                raise ValueError(
                    f"field {name!r} has shape {arr.shape}, expected {(cell_count,)}"
                )
        if node_count is not None:
            for name, values in self.node_scalars.items():
                arr = np.asarray(values)
                if arr.shape != (node_count,):
                    raise ValueError(
                        f"nodal field {name!r} has shape {arr.shape}, expected {(node_count,)}"
                    )


@dataclass(frozen=True)
class H5IngestPlan:
    """Summary of what can be loaded before a database connection is opened."""

    h5_path: str
    part_name: str
    instance_name: str
    node_count: int
    cell_count: int
    element_types: Tuple[str, ...]
    frame_count: int
    mapped_timesteps: Tuple[int, ...]
    mapped_variables: Tuple[str, ...]
    mapped_node_variables: Tuple[str, ...] = ()
    skipped_frames: Tuple[str, ...] = ()
