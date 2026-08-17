"""TileDB adapter for the canonical ODB-like HDF5 model.

The adapter intentionally writes the existing TileDB W1-W8 layout and adds
H5-only metadata/source-label arrays needed by W9-W11. PostgreSQL and IoTDB
paths are not involved.
"""

from __future__ import annotations

import os
import shutil
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from cfd_bench.ingest.h5.artifacts import max_neighbor_diffs
from cfd_bench.ingest.h5.model import CanonicalFrame, CanonicalMesh, H5IngestPlan
from cfd_bench.ingest.h5.postgresql import build_ingest_plan


def _cell_vars_leaf(zone: str) -> str:
    z = str(zone or "0_Fluid").strip().lower()
    if "hull" in z or "wall" in z or z in {"1_hull", "hull"}:
        return "cell_vars_hull"
    return "cell_vars"


def _fixed_matrix(rows, width: int, *, label: str) -> np.ndarray:
    if any(len(row) > width for row in rows):
        found = max(len(row) for row in rows)
        raise ValueError(f"TileDB {label} supports <= {width} values per cell; found {found}")
    out = np.full((len(rows), width), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        vals = np.asarray(row, dtype=np.int64).reshape(-1)
        if vals.size:
            out[i, : vals.size] = vals.astype(np.int32)
    return out


def _mesh_payload(mesh: CanonicalMesh):
    if mesh.node_count:
        mins = np.min(mesh.node_coordinates, axis=0)
        maxs = np.max(mesh.node_coordinates, axis=0)
    else:
        mins = maxs = np.zeros(3, dtype=np.float64)

    cell_rows = {
        "cx": np.asarray(mesh.cell_centroids[:, 0], dtype=np.float32),
        "cy": np.asarray(mesh.cell_centroids[:, 1], dtype=np.float32),
        "cz": np.asarray(mesh.cell_centroids[:, 2], dtype=np.float32),
        "xmin": np.empty(mesh.cell_count, dtype=np.float32),
        "xmax": np.empty(mesh.cell_count, dtype=np.float32),
        "ymin": np.empty(mesh.cell_count, dtype=np.float32),
        "ymax": np.empty(mesh.cell_count, dtype=np.float32),
        "zmin": np.empty(mesh.cell_count, dtype=np.float32),
        "zmax": np.empty(mesh.cell_count, dtype=np.float32),
        # Keep the legacy numeric cell_type contract. The lossless H5 type is
        # stored separately in cell_source.tdb.
        "cell_type": np.asarray([len(x) for x in mesh.cell_node_ids], dtype=np.int32),
    }
    for cid, node_ids in enumerate(mesh.cell_node_ids):
        pts = mesh.node_coordinates[np.asarray(node_ids, dtype=np.int64)]
        lo = np.min(pts, axis=0)
        hi = np.max(pts, axis=0)
        cell_rows["xmin"][cid], cell_rows["xmax"][cid] = lo[0], hi[0]
        cell_rows["ymin"][cid], cell_rows["ymax"][cid] = lo[1], hi[1]
        cell_rows["zmin"][cid], cell_rows["zmax"][cid] = lo[2], hi[2]

    meta = {
        "node_count": mesh.node_count,
        "cell_count": mesh.cell_count,
        "face_count": 0,
        "bbox_min_x": float(mins[0]),
        "bbox_max_x": float(maxs[0]),
        "bbox_min_y": float(mins[1]),
        "bbox_max_y": float(maxs[1]),
        "bbox_min_z": float(mins[2]),
        "bbox_max_z": float(maxs[2]),
    }
    # Existing TileDB schemas have 16 fixed slots. B33 and C3D10 both fit.
    cell_nodes = _fixed_matrix(mesh.cell_node_ids, 16, label="cell_nodes")
    adjacency = _fixed_matrix(mesh.cell_adjacency, 16, label="cell_adjacency")
    return meta, cell_rows, cell_nodes, adjacency


def _frame_metadata(frame: CanonicalFrame) -> dict:
    info = frame.info
    return {
        "timestep": int(frame.timestep),
        "step_name": str(info.step_name),
        "step_index": int(info.step_index),
        "step_domain": None if info.step_domain is None else int(info.step_domain),
        "frame_name": str(info.frame_name),
        "frame_index": int(info.frame_index),
        "inc_or_mode": None if info.inc_or_mode is None else int(info.inc_or_mode),
        "time_or_frequency": None if info.time_or_frequency is None else float(info.time_or_frequency),
        "description": str(info.description or ""),
        "load_case": str(info.load_case or ""),
        "source_fields": dict(frame.source_fields),
    }


def write_plan_to_tiledb(
    dataset_key: str,
    zone_type: str,
    root_path: str,
    plan: H5IngestPlan,
    mesh: CanonicalMesh,
    frames: Sequence[CanonicalFrame],
    *,
    overwrite: bool = True,
    io_module=None,
    ctx=None,
) -> None:
    """Write a prepared canonical H5 plan to TileDB.

    ``io_module`` is injectable for tests so parsing/mapping can be validated
    without requiring the optional TileDB package in the test environment.
    """
    if io_module is None:
        from cfd_bench.ingest.tiledb import io as io_module

    dataset_dir = os.path.join(root_path, dataset_key)
    if overwrite and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    meta, cell_rows, cell_nodes, adjacency = _mesh_payload(mesh)
    print(
        f"[ingest-h5][tiledb] mesh nodes={mesh.node_count} cells={mesh.cell_count} "
        f"types={sorted(set(mesh.cell_element_types))}"
    )
    io_module.write_mesh_meta(
        io_module.mesh_static_uri(root_path, dataset_key, zone_type, "mesh_meta"),
        meta, ctx=ctx, overwrite=overwrite,
    )
    io_module.write_nodes(
        io_module.mesh_static_uri(root_path, dataset_key, zone_type, "nodes"),
        mesh.node_coordinates[:, 0], mesh.node_coordinates[:, 1], mesh.node_coordinates[:, 2],
        ctx=ctx, overwrite=overwrite,
    )
    io_module.write_cells(
        io_module.mesh_static_uri(root_path, dataset_key, zone_type, "cells"),
        cell_rows, mesh.cell_count, ctx=ctx, overwrite=overwrite,
    )
    io_module.write_cell_nodes(
        io_module.mesh_static_uri(root_path, dataset_key, zone_type, "cell_nodes"),
        cell_nodes, mesh.cell_count, ctx=ctx, overwrite=overwrite,
    )
    io_module.write_cell_adjacency(
        io_module.mesh_static_uri(root_path, dataset_key, zone_type, "cell_adjacency"),
        adjacency, mesh.cell_count, ctx=ctx, overwrite=overwrite,
    )
    io_module.write_source_labels(
        io_module.mesh_static_uri(root_path, dataset_key, zone_type, "node_source"),
        mesh.source_node_labels, dim_name="node_id", ctx=ctx, overwrite=overwrite,
    )
    io_module.write_cell_source(
        io_module.mesh_static_uri(root_path, dataset_key, zone_type, "cell_source"),
        mesh.source_cell_labels, mesh.cell_element_types, ctx=ctx, overwrite=overwrite,
    )

    common_cell = sorted(set.intersection(*(set(f.cell_scalars) for f in frames))) if frames else []
    common_node = sorted(set.intersection(*(set(f.node_scalars) for f in frames))) if frames else []
    h5_meta = {
        "is_h5": True,
        "zone": str(zone_type),
        "part_name": str(mesh.part_name),
        "instance_name": str(mesh.instance_name),
        "variables_csv": ",".join(plan.mapped_variables),
        "nodal_variables_csv": ",".join(plan.mapped_node_variables),
        "common_variables_csv": ",".join(common_cell),
        "common_nodal_variables_csv": ",".join(common_node),
        "element_types_csv": ",".join(plan.element_types),
        "timesteps_csv": ",".join(str(x) for x in plan.mapped_timesteps),
        "node_count": mesh.node_count,
        "cell_count": mesh.cell_count,
        "frames": [_frame_metadata(frame) for frame in frames],
    }
    io_module.write_h5_dataset_meta(
        io_module.h5_metadata_uri(root_path, dataset_key, "dataset_meta"),
        h5_meta, ctx=ctx, overwrite=overwrite,
    )

    print(f"[ingest-h5][tiledb] frames={len(frames)} vars={list(plan.mapped_variables)}")
    for frame in frames:
        step = int(frame.timestep)
        if frame.cell_scalars:
            vars_ = sorted(frame.cell_scalars)
            io_module.write_cell_vars(
                io_module.post_uri(root_path, dataset_key, step, _cell_vars_leaf(zone_type)),
                {v: np.asarray(frame.cell_scalars[v], dtype=np.float32) for v in vars_},
                mesh.cell_count, vars_, ctx=ctx, overwrite=overwrite,
            )
            max_diffs = {
                v: max_neighbor_diffs(mesh, np.asarray(frame.cell_scalars[v], dtype=np.float64))
                for v in vars_
            }
            io_module.write_max_diffs(
                io_module.derived_uri(root_path, dataset_key, step, "max_diff"),
                max_diffs, ctx=ctx, overwrite=overwrite,
            )
        if frame.node_scalars:
            vars_ = sorted(frame.node_scalars)
            io_module.write_node_vars(
                io_module.post_uri(root_path, dataset_key, step, "node_vars"),
                {v: np.asarray(frame.node_scalars[v], dtype=np.float32) for v in vars_},
                mesh.node_count, vars_, ctx=ctx, overwrite=overwrite,
            )


def load_h5_to_tiledb(
    h5_path: str,
    dataset_key: str,
    *,
    root_path: str = "TileDB_Instances",
    instance_name: Optional[str] = None,
    zone_type: str = "0_Fluid",
    step_names: Optional[Sequence[str]] = None,
    vector_field: Optional[str] = None,
    scalar_fields: Optional[Sequence[str]] = None,
    explicit_mapping: Optional[Mapping[str, Tuple[str, Optional[str]]]] = None,
    timestep_mode: str = "sequence",
    include_empty_frames: bool = False,
    overwrite: bool = True,
    ctx=None,
) -> H5IngestPlan:
    if ctx is None:
        try:
            import tiledb
        except ImportError as exc:
            raise RuntimeError(
                "TileDB H5 ingest requires tiledb. "
                "Install with: pip install 'cfd_bench[tiledb]'"
            ) from exc
        ctx = tiledb.Ctx()

    plan, mesh, frames = build_ingest_plan(
        h5_path,
        instance_name=instance_name,
        step_names=step_names,
        vector_field=vector_field,
        scalar_fields=scalar_fields,
        explicit_mapping=explicit_mapping,
        timestep_mode=timestep_mode,
        include_empty_frames=include_empty_frames,
    )
    write_plan_to_tiledb(
        dataset_key, zone_type, root_path, plan, mesh, frames,
        overwrite=overwrite, ctx=ctx,
    )
    return plan
