from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import tiledb


def _ctx(ctx: Optional[tiledb.Ctx]) -> tiledb.Ctx:
    return ctx if ctx is not None else tiledb.Ctx()


def _tile(count: int, preferred: int = 10000) -> int:
    return max(1, min(preferred, count))


def schema_mesh_meta(ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    dom = tiledb.Domain(tiledb.Dim("meta_id", (0, 0), tile=1, dtype=np.int32, ctx=c), ctx=c)
    attrs = [
        tiledb.Attr("node_count", dtype=np.int32, ctx=c),
        tiledb.Attr("cell_count", dtype=np.int32, ctx=c),
        tiledb.Attr("face_count", dtype=np.int32, ctx=c),
        tiledb.Attr("bbox_min_x", dtype=np.float32, ctx=c),
        tiledb.Attr("bbox_max_x", dtype=np.float32, ctx=c),
        tiledb.Attr("bbox_min_y", dtype=np.float32, ctx=c),
        tiledb.Attr("bbox_max_y", dtype=np.float32, ctx=c),
        tiledb.Attr("bbox_min_z", dtype=np.float32, ctx=c),
        tiledb.Attr("bbox_max_z", dtype=np.float32, ctx=c),
    ]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_nodes(node_count: int, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(node_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("node_id", (0, n), tile=_tile(node_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [
        tiledb.Attr("x", dtype=np.float32, ctx=c),
        tiledb.Attr("y", dtype=np.float32, ctx=c),
        tiledb.Attr("z", dtype=np.float32, ctx=c),
    ]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_cells(cell_count: int, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(cell_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("cell_id", (0, n), tile=_tile(cell_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [
        tiledb.Attr("cx", dtype=np.float32, ctx=c),
        tiledb.Attr("cy", dtype=np.float32, ctx=c),
        tiledb.Attr("cz", dtype=np.float32, ctx=c),
        tiledb.Attr("xmin", dtype=np.float32, ctx=c),
        tiledb.Attr("xmax", dtype=np.float32, ctx=c),
        tiledb.Attr("ymin", dtype=np.float32, ctx=c),
        tiledb.Attr("ymax", dtype=np.float32, ctx=c),
        tiledb.Attr("zmin", dtype=np.float32, ctx=c),
        tiledb.Attr("zmax", dtype=np.float32, ctx=c),
        tiledb.Attr("cell_type", dtype=np.int32, ctx=c),
    ]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_cell_nodes(cell_count: int, max_nodes: int = 16, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(cell_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("cell_id", (0, n), tile=_tile(cell_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [tiledb.Attr(f"node_id_{i}", dtype=np.int32, ctx=c) for i in range(max_nodes)]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_cell_adjacency(cell_count: int, max_neighbors: int = 16, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(cell_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("cell_id", (0, n), tile=_tile(cell_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [tiledb.Attr(f"neighbor_id_{i}", dtype=np.int32, ctx=c) for i in range(max_neighbors)]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_face_planes(face_count: int, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(face_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("face_row_id", (0, n), tile=_tile(face_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [
        tiledb.Attr("cell_id", dtype=np.int32, ctx=c),
        tiledb.Attr("neighbor_id", dtype=np.int32, ctx=c),
        tiledb.Attr("nx", dtype=np.float32, ctx=c),
        tiledb.Attr("ny", dtype=np.float32, ctx=c),
        tiledb.Attr("nz", dtype=np.float32, ctx=c),
        tiledb.Attr("d", dtype=np.float32, ctx=c),
        tiledb.Attr("face_area", dtype=np.float32, ctx=c),
        tiledb.Attr("face_cx", dtype=np.float32, ctx=c),
        tiledb.Attr("face_cy", dtype=np.float32, ctx=c),
        tiledb.Attr("face_cz", dtype=np.float32, ctx=c),
    ]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_boundary_faces(face_count: int, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(face_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("boundary_row_id", (0, n), tile=_tile(face_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [
        tiledb.Attr("cell_id", dtype=np.int32, ctx=c),
        tiledb.Attr("patch_code", dtype=np.float32, ctx=c),
        tiledb.Attr("nx", dtype=np.float32, ctx=c),
        tiledb.Attr("ny", dtype=np.float32, ctx=c),
        tiledb.Attr("nz", dtype=np.float32, ctx=c),
        tiledb.Attr("area", dtype=np.float32, ctx=c),
        tiledb.Attr("cx", dtype=np.float32, ctx=c),
        tiledb.Attr("cy", dtype=np.float32, ctx=c),
        tiledb.Attr("cz", dtype=np.float32, ctx=c),
    ]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_cell_vars(cell_count: int, var_names: Sequence[str], ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(cell_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("cell_id", (0, n), tile=_tile(cell_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [tiledb.Attr(str(v), dtype=np.float32, ctx=c) for v in var_names]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_node_vars(node_count: int, var_names: Sequence[str], ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(node_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("node_id", (0, n), tile=_tile(node_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [tiledb.Attr(str(v), dtype=np.float32, ctx=c) for v in var_names]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_cell_qcriterion(cell_count: int, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(cell_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("cell_id", (0, n), tile=_tile(cell_count), dtype=np.int32, ctx=c), ctx=c)
    attrs = [tiledb.Attr("q", dtype=np.float32, ctx=c)]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_cell_gradient(cell_count: int, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    n = max(0, int(cell_count) - 1)
    dom = tiledb.Domain(tiledb.Dim("cell_id", (0, n), tile=_tile(cell_count), dtype=np.int32, ctx=c), ctx=c)
    names = [
        "du_dx", "du_dy", "du_dz",
        "dv_dx", "dv_dy", "dv_dz",
        "dw_dx", "dw_dy", "dw_dz",
    ]
    attrs = [tiledb.Attr(name, dtype=np.float32, ctx=c) for name in names]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def create_dense_array(uri: str, schema: tiledb.ArraySchema, overwrite: bool = False, ctx: Optional[tiledb.Ctx] = None):
    c = _ctx(ctx)
    if tiledb.array_exists(uri, ctx=c):
        if overwrite:
            tiledb.remove(uri, ctx=c)
        else:
            return
    tiledb.Array.create(uri, schema, ctx=c)


def write_dense_by_id(uri: str, data: dict, dim_name: str, ctx: Optional[tiledb.Ctx] = None):
    """Write dense array where keys in data are attribute names and values are numpy arrays indexed 0..N-1."""
    c = _ctx(ctx)
    n = None
    for arr in data.values():
        n = len(arr)
        break
    if n is None:
        return
    with tiledb.open(uri, mode="w", ctx=c) as A:
        # Full-domain dense write.  Using a NumPy coordinate vector here relies
        # on fancy indexing that is not supported consistently for DenseArray.
        A[:] = data


def schema_source_labels(count: int, dim_name: str, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    """Dense source-label mapping for H5 node/cell ids."""
    c = _ctx(ctx)
    n = max(0, int(count) - 1)
    dom = tiledb.Domain(
        tiledb.Dim(dim_name, (0, n), tile=_tile(count), dtype=np.int32, ctx=c),
        ctx=c,
    )
    attrs = [tiledb.Attr("source_label", dtype=np.int64, ctx=c)]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_cell_source(cell_count: int, ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    """H5 source element label plus compact element-type code."""
    c = _ctx(ctx)
    n = max(0, int(cell_count) - 1)
    dom = tiledb.Domain(
        tiledb.Dim("cell_id", (0, n), tile=_tile(cell_count), dtype=np.int32, ctx=c),
        ctx=c,
    )
    attrs = [
        tiledb.Attr("source_label", dtype=np.int64, ctx=c),
        tiledb.Attr("element_type_code", dtype=np.int32, ctx=c),
    ]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_h5_dataset_meta(ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    """Small numeric anchor array; string/list metadata is stored in Array.meta."""
    c = _ctx(ctx)
    dom = tiledb.Domain(tiledb.Dim("meta_id", (0, 0), tile=1, dtype=np.int32, ctx=c), ctx=c)
    attrs = [
        tiledb.Attr("is_h5", dtype=np.uint8, ctx=c),
        tiledb.Attr("node_count", dtype=np.int64, ctx=c),
        tiledb.Attr("cell_count", dtype=np.int64, ctx=c),
    ]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)


def schema_max_diff(var_names: Sequence[str], ctx: Optional[tiledb.Ctx] = None) -> tiledb.ArraySchema:
    c = _ctx(ctx)
    dom = tiledb.Domain(tiledb.Dim("meta_id", (0, 0), tile=1, dtype=np.int32, ctx=c), ctx=c)
    attrs = [tiledb.Attr(str(v), dtype=np.float64, ctx=c) for v in var_names]
    return tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=c)
