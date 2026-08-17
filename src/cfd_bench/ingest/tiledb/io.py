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
    schema.create_dense_array(uri, schema.schema_cell_nodes(cell_count, ctx=ctx), overwrite=overwrite, ctx=ctx)
    data = {f"node_id_{i}": matrix[:, i].astype(np.int32) for i in range(matrix.shape[1])}
    schema.write_dense_by_id(uri, data, "cell_id", ctx=ctx)


def write_cell_adjacency(uri: str, matrix: np.ndarray, cell_count: int, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_cell_adjacency(cell_count, ctx=ctx), overwrite=overwrite, ctx=ctx)
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


def h5_metadata_uri(root: str, dataset_key: str, leaf: str = "dataset_meta") -> str:
    return os.path.join(root, dataset_key, "h5_metadata", f"{leaf}.tdb")


def write_node_vars(uri: str, var_data: dict, node_count: int, var_names: Sequence[str], ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_node_vars(node_count, var_names, ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(uri, {k: np.asarray(v, dtype=np.float32) for k, v in var_data.items()}, "node_id", ctx=ctx)


def write_source_labels(uri: str, labels, *, dim_name: str, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    values = np.asarray(labels, dtype=np.int64)
    ensure_parent(uri)
    schema.create_dense_array(
        uri,
        schema.schema_source_labels(len(values), dim_name, ctx),
        overwrite=overwrite,
        ctx=ctx,
    )
    schema.write_dense_by_id(uri, {"source_label": values}, dim_name, ctx=ctx)


def write_cell_source(uri: str, labels, element_types, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    import json

    labels = np.asarray(labels, dtype=np.int64)
    types = [str(x) for x in element_types]
    unique = sorted(set(types))
    type_to_code = {name: i for i, name in enumerate(unique)}
    codes = np.asarray([type_to_code[name] for name in types], dtype=np.int32)
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_cell_source(len(labels), ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(
        uri,
        {"source_label": labels, "element_type_code": codes},
        "cell_id",
        ctx=ctx,
    )
    c = ctx if ctx is not None else tiledb.Ctx()
    with tiledb.open(uri, mode="w", ctx=c) as A:
        A.meta["element_types_json"] = json.dumps(unique, ensure_ascii=True)


def write_h5_dataset_meta(uri: str, meta: dict, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    """Persist H5 discovery metadata in a TileDB anchor array.

    TileDB Array metadata is used for strings/lists so this stays compatible
    across TileDB-Py versions without depending on variable-length string attrs.
    """
    import json

    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_h5_dataset_meta(ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(
        uri,
        {
            "is_h5": np.asarray([1 if meta.get("is_h5", True) else 0], dtype=np.uint8),
            "node_count": np.asarray([int(meta.get("node_count", 0))], dtype=np.int64),
            "cell_count": np.asarray([int(meta.get("cell_count", 0))], dtype=np.int64),
        },
        "meta_id",
        ctx=ctx,
    )
    c = ctx if ctx is not None else tiledb.Ctx()
    with tiledb.open(uri, mode="w", ctx=c) as A:
        for key in (
            "zone", "part_name", "instance_name", "variables_csv",
            "nodal_variables_csv", "common_variables_csv",
            "common_nodal_variables_csv", "element_types_csv", "timesteps_csv",
        ):
            A.meta[key] = str(meta.get(key, ""))
        A.meta["frames_json"] = json.dumps(meta.get("frames", []), ensure_ascii=True)


def write_max_diffs(uri: str, values: dict, ctx: Optional[tiledb.Ctx] = None, overwrite: bool = True):
    names = [str(v).upper() for v in sorted(values)]
    if not names:
        return
    ensure_parent(uri)
    schema.create_dense_array(uri, schema.schema_max_diff(names, ctx), overwrite=overwrite, ctx=ctx)
    schema.write_dense_by_id(
        uri,
        {name: np.asarray([float(values[name])], dtype=np.float64) for name in names},
        "meta_id",
        ctx=ctx,
    )
