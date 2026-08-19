"""High-performance VTK dataset builders for CFD-Bench.

The VTK backend stores one ``.vtu`` file per dataset/zone/frame.  Each file
contains the mesh plus frame-local cell/point arrays, while ``manifest.json``
keeps discovery metadata and derived W3 max-diffs.  Builders here accept the
same canonical CFD/H5 payloads consumed by the frozen database backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


def _vtk_modules():
    try:
        import vtk
        from vtkmodules.util import numpy_support
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "VTK backend requires vtk. Install with: pip install 'cfd_bench[vtk]'"
        ) from exc
    return vtk, numpy_support


def _append_array(dataset, values: np.ndarray, name: str, *, point: bool = False) -> None:
    vtk, numpy_support = _vtk_modules()
    arr = np.asarray(values)
    if arr.dtype.kind in {"U", "S", "O"}:
        vtk_arr = vtk.vtkStringArray()
        vtk_arr.SetName(str(name))
        vtk_arr.SetNumberOfValues(int(arr.size))
        for i, value in enumerate(arr.reshape(-1).tolist()):
            vtk_arr.SetValue(i, str(value))
    else:
        vtk_arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(arr), deep=True)
        vtk_arr.SetName(str(name))
    target = dataset.GetPointData() if point else dataset.GetCellData()
    target.AddArray(vtk_arr)


def _cell_type_for_h5(element_type: str, n: int) -> int:
    vtk, _ = _vtk_modules()
    et = str(element_type or "").upper()
    if et.startswith("B") or et.startswith("T3D"):
        if n == 3:
            return vtk.VTK_QUADRATIC_EDGE
        return vtk.VTK_LINE if n >= 2 else vtk.VTK_VERTEX
    if et.startswith(("S", "M3D", "CPS", "CPE")):
        if n == 3:
            return vtk.VTK_TRIANGLE
        if n == 4:
            return vtk.VTK_QUAD
        if n == 6:
            return vtk.VTK_QUADRATIC_TRIANGLE
        if n == 8:
            return vtk.VTK_QUADRATIC_QUAD
        return vtk.VTK_POLYGON if n >= 3 else vtk.VTK_LINE
    if et.startswith("C3D"):
        if n == 4:
            return vtk.VTK_TETRA
        if n == 5:
            return vtk.VTK_PYRAMID
        if n == 6:
            return vtk.VTK_WEDGE
        if n == 8:
            return vtk.VTK_HEXAHEDRON
        if n == 10:
            return vtk.VTK_QUADRATIC_TETRA
        if n == 20:
            return vtk.VTK_QUADRATIC_HEXAHEDRON
        return vtk.VTK_CONVEX_POINT_SET
    # Generic fallback by node count.  It is intentionally conservative and
    # keeps arbitrary structural elements queryable rather than rejecting them.
    if n == 1:
        return vtk.VTK_VERTEX
    if n == 2:
        return vtk.VTK_LINE
    if n == 3:
        return vtk.VTK_TRIANGLE
    if n == 4:
        return vtk.VTK_TETRA
    if n == 8:
        return vtk.VTK_HEXAHEDRON
    return vtk.VTK_CONVEX_POINT_SET


def _cell_type_for_cfd(zone_type: str, n: int) -> int:
    vtk, _ = _vtk_modules()
    zt = str(zone_type or "").upper()
    if "POLYGON" in zt and "POLYHEDRON" not in zt:
        if n == 1:
            return vtk.VTK_VERTEX
        if n == 2:
            return vtk.VTK_LINE
        if n == 3:
            return vtk.VTK_TRIANGLE
        if n == 4:
            return vtk.VTK_QUAD
        return vtk.VTK_POLYGON
    # FEPolyhedron datasets produced by the benchmark cases are dominated by
    # ordinary tet/wedge/hex cells.  VTK_CONVEX_POINT_SET remains a safe
    # geometry fallback for uncommon arbitrary polyhedra when face lists are
    # not part of the canonical backend-neutral payload.
    if n == 4:
        return vtk.VTK_TETRA
    if n == 5:
        return vtk.VTK_PYRAMID
    if n == 6:
        return vtk.VTK_WEDGE
    if n == 8:
        return vtk.VTK_HEXAHEDRON
    if n == 3:
        return vtk.VTK_TRIANGLE
    if n == 2:
        return vtk.VTK_LINE
    return vtk.VTK_CONVEX_POINT_SET


def _connectivity_arrays(cell_nodes: Sequence[Sequence[int]]) -> tuple:
    """Flatten canonical connectivity once without per-cell NumPy allocations."""
    from itertools import chain

    n = len(cell_nodes)
    lengths = np.fromiter((len(row) for row in cell_nodes), dtype=np.int64, count=n)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1]) if offsets.size else 0
    conn = np.fromiter(
        chain.from_iterable(cell_nodes), dtype=np.int64, count=total
    ) if total else np.zeros((0,), dtype=np.int64)
    return lengths, offsets, conn


def _cfd_cell_types(zone_type: str, lengths: np.ndarray) -> np.ndarray:
    """Vectorized cell-type mapping for the high-volume CFD path."""
    vtk, _ = _vtk_modules()
    zt = str(zone_type or "").upper()
    out = np.full(lengths.shape, vtk.VTK_CONVEX_POINT_SET, dtype=np.uint8)
    if "POLYGON" in zt and "POLYHEDRON" not in zt:
        mapping = {
            1: vtk.VTK_VERTEX, 2: vtk.VTK_LINE, 3: vtk.VTK_TRIANGLE, 4: vtk.VTK_QUAD
        }
        out.fill(vtk.VTK_POLYGON)
    else:
        mapping = {
            2: vtk.VTK_LINE, 3: vtk.VTK_TRIANGLE, 4: vtk.VTK_TETRA,
            5: vtk.VTK_PYRAMID, 6: vtk.VTK_WEDGE, 8: vtk.VTK_HEXAHEDRON,
        }
    for width, vtk_type in mapping.items():
        out[lengths == int(width)] = int(vtk_type)
    return out


def _h5_cell_types(element_types: Sequence[str], lengths: np.ndarray) -> np.ndarray:
    """Map structural element classes while caching repeated type/width pairs."""
    if len(element_types) != len(lengths):
        raise ValueError("cell_element_types length does not match cell topology")
    out = np.empty(len(lengths), dtype=np.uint8)
    cache = {}
    for i, (element_type, width) in enumerate(zip(element_types, lengths)):
        key = (str(element_type).upper(), int(width))
        vtk_type = cache.get(key)
        if vtk_type is None:
            vtk_type = int(_cell_type_for_h5(key[0], key[1]))
            cache[key] = vtk_type
        out[i] = vtk_type
    return out


def _set_cells(grid, offsets: np.ndarray, conn: np.ndarray, cell_types: np.ndarray) -> None:
    vtk, numpy_support = _vtk_modules()
    ca = vtk.vtkCellArray()
    ca.SetData(
        numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_support.numpy_to_vtkIdTypeArray(conn, deep=True),
    )
    vtk_types = numpy_support.numpy_to_vtk(
        np.asarray(cell_types, dtype=np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    grid.SetCells(vtk_types, ca)


def build_grid(
    node_xyz: np.ndarray,
    cell_nodes: Sequence[Sequence[int]],
    *,
    cell_element_types: Optional[Sequence[str]] = None,
    cfd_zone_type: Optional[str] = None,
    source_node_ids: Optional[Sequence[int]] = None,
    source_cell_ids: Optional[Sequence[int]] = None,
    cell_fields: Optional[Mapping[str, np.ndarray]] = None,
    point_fields: Optional[Mapping[str, np.ndarray]] = None,
):
    """Build a ``vtkUnstructuredGrid`` from canonical dense topology."""
    vtk, numpy_support = _vtk_modules()
    xyz = np.ascontiguousarray(node_xyz, dtype=np.float64).reshape(-1, 3)
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(xyz, deep=True))

    n_cells = len(cell_nodes)
    lengths, offsets, conn = _connectivity_arrays(cell_nodes)
    if cell_element_types is not None:
        types = _h5_cell_types(cell_element_types, lengths)
    else:
        types = _cfd_cell_types(cfd_zone_type or "FEPOLYHEDRON", lengths)

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    _set_cells(grid, offsets, conn, types)

    dense_node_ids = np.arange(xyz.shape[0], dtype=np.int64)
    dense_cell_ids = np.arange(n_cells, dtype=np.int64)
    _append_array(grid, dense_node_ids, "dense_node_id", point=True)
    _append_array(grid, dense_cell_ids, "dense_cell_id")
    _append_array(
        grid,
        np.asarray(source_node_ids if source_node_ids is not None else dense_node_ids + 1, dtype=np.int64),
        "source_node_id",
        point=True,
    )
    _append_array(
        grid,
        np.asarray(source_cell_ids if source_cell_ids is not None else dense_cell_ids + 1, dtype=np.int64),
        "source_cell_id",
    )
    if cell_element_types is not None:
        _append_array(grid, np.asarray(cell_element_types, dtype=object), "element_type")

    for name, values in (cell_fields or {}).items():
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size != n_cells:
            raise ValueError(f"cell field {name}: {arr.size} values for {n_cells} cells")
        _append_array(grid, arr, str(name).upper())
    for name, values in (point_fields or {}).items():
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size != xyz.shape[0]:
            raise ValueError(f"point field {name}: {arr.size} values for {xyz.shape[0]} points")
        _append_array(grid, arr, str(name).upper(), point=True)

    # Velocity is a workload-time derived field.  Persist only canonical input
    # arrays so large CFD frames do not duplicate U/V/W at every mesh point.
    return grid


def ensure_point_velocity(grid) -> None:
    """Attach a point ``Velocity`` vector from U/V/W, deriving points if needed."""
    vtk, numpy_support = _vtk_modules()
    point_data = grid.GetPointData()
    if point_data.GetArray("Velocity") is not None:
        return
    point_components = [point_data.GetArray(v) for v in ("U", "V", "W")]
    if not all(point_components):
        cell_data = grid.GetCellData()
        if all(cell_data.GetArray(v) is not None for v in ("U", "V", "W")):
            conv = vtk.vtkCellDataToPointData()
            conv.SetInputData(grid)
            conv.PassCellDataOn()
            conv.Update()
            converted = vtk.vtkUnstructuredGrid()
            converted.ShallowCopy(conv.GetOutput())
            grid.ShallowCopy(converted)
            point_data = grid.GetPointData()
            point_components = [point_data.GetArray(v) for v in ("U", "V", "W")]
    if not all(point_components):
        return
    comps = [numpy_support.vtk_to_numpy(a).reshape(-1) for a in point_components]
    velocity = np.column_stack(comps).astype(np.float64, copy=False)
    vtk_arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(velocity), deep=True)
    vtk_arr.SetName("Velocity")
    vtk_arr.SetNumberOfComponents(3)
    grid.GetPointData().AddArray(vtk_arr)


def write_vtu(grid, path: str) -> str:
    vtk, _ = _vtk_modules()
    out = str(Path(path))
    if not out.lower().endswith(".vtu"):
        out += ".vtu"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(out)
    writer.SetInputData(grid)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()
    if hasattr(writer, "SetCompressorTypeToLZ4"):
        writer.SetCompressorTypeToLZ4()
    ok = int(writer.Write())
    if ok != 1:
        raise IOError(f"VTK writer failed for {out}")
    return out


def read_vtk_grid(path: str):
    vtk, _ = _vtk_modules()
    file = str(path)
    if file.lower().endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif file.lower().endswith(".vtk"):
        reader = vtk.vtkUnstructuredGridReader()
    else:
        raise ValueError(f"unsupported VTK file: {file}")
    reader.SetFileName(file)
    reader.Update()
    grid = reader.GetOutput()
    if grid is None:
        raise IOError(f"failed to read VTK grid: {file}")
    return grid


# ---------------------------------------------------------------------------
# Legacy helper names retained for compatibility with direct module users.
# New backend ingest does not use these paths.


def write_vtk_mesh(mesh, base_path: str) -> str:
    vtk, _ = _vtk_modules()
    out = str(base_path)
    if not out.lower().endswith(".vtk"):
        out += ".vtk"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(out)
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()
    if int(writer.Write()) != 1:
        raise IOError(f"VTK writer failed for {out}")
    return out
