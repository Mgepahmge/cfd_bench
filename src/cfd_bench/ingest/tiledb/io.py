"""TileDB write helpers for ingest scripts."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import tiledb

from cfd_bench.storage.schema import tiledb_schemas as schema


def mesh_static_uri(root: str, dataset_key: str, zone: str, leaf: str) -> str:
    return os.path.join(root, dataset_key, "mesh_static", zone, f"{leaf}.tdb")


def post_uri(root: str, dataset_key: str, step: int, leaf: str) -> str:
    return os.path.join(root, dataset_key, "post_processing", f"step_{int(step)}", f"{leaf}.tdb")


def derived_uri(root: str, dataset_key: str, step: int, leaf: str) -> str:
    return os.path.join(root, dataset_key, "derived", f"step_{int(step)}", f"{leaf}.tdb")


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_mesh_meta(uri: str, meta: dict, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_mesh_meta(ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(
        uri,
        {
            "node_count": np.array([int(meta["node_count"])], dtype=np.int32),
            "cell_count": np.array([int(meta["cell_count"])], dtype=np.int32),
            "face_count": np.array([int(meta["face_count"])], dtype=np.int32),
            "bbox_min_x": np.array([float(meta["bbox_min_x"])], dtype=np.float32),
            "bbox_max_x": np.array([float(meta["bbox_max_x"])], dtype=np.float32),
            "bbox_min_y": np.array([float(meta["bbox_min_y"])], dtype=np.float32),
            "bbox_max_y": np.array([float(meta["bbox_max_y"])], dtype=np.float32),
            "bbox_min_z": np.array([float(meta["bbox_min_z"])], dtype=np.float32),
            "bbox_max_z": np.array([float(meta["bbox_max_z"])], dtype=np.float32),
        },
        "meta_id",
        ctx=ctx,
    )


def write_nodes(uri: str, x, y, z, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    n = len(x)
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_nodes(n, ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(
        uri,
        {
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "z": np.asarray(z, dtype=np.float32),
        },
        "node_id",
        ctx=ctx,
    )


def write_cells(uri: str, rows: dict, cell_count: int, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_cells(cell_count, ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(
        uri,
        {k: np.asarray(v, dtype=np.float32 if k != "cell_type" else np.int32) for k, v in rows.items()},
        "cell_id",
        ctx=ctx,
    )


def write_cell_nodes(uri: str, matrix: np.ndarray, cell_count: int, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_cell_nodes(cell_count, ctx), overwrite=overwrite, ctx=ctx)
    data = {f"node_id_{i}": matrix[:, i].astype(np.int32) for i in range(matrix.shape[1])}
    schema.write_dense_by_id(uri, data, "cell_id", ctx=ctx)


def write_cell_adjacency(uri: str, matrix: np.ndarray, cell_count: int, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_cell_adjacency(cell_count, ctx), overwrite=overwrite, ctx=ctx)
    data = {f"neighbor_id_{i}": matrix[:, i].astype(np.int32) for i in range(matrix.shape[1])}
    schema.write_dense_by_id(uri, data, "cell_id", ctx=ctx)


def write_face_planes(uri: str, rows: dict, face_count: int, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    if face_count <= 0:
        return
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_face_planes(face_count, ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(
        uri,
        {k: np.asarray(v, dtype=np.float32 if k not in ("cell_id", "neighbor_id") else np.int32) for k, v in rows.items()},
        "face_row_id",
        ctx=ctx,
    )


def write_boundary_faces(uri: str, rows: dict, face_count: int, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    if face_count <= 0:
        return
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_boundary_faces(face_count, ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(
        uri,
        {k: np.asarray(v, dtype=np.float32 if k not in ("cell_id",) else np.int32) for k, v in rows.items()},
        "boundary_row_id",
        ctx=ctx,
    )


def write_cell_vars(uri: str, var_data: dict, cell_count: int, var_names: Sequence[str], ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_cell_vars(cell_count, var_names, ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(uri, {k: np.asarray(v, dtype=np.float32) for k, v in var_data.items()}, "cell_id", ctx=ctx)
