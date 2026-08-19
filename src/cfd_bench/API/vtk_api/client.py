from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from cfd_bench.core.cfd_nodal_projection import (
    NodeCellCSR,
    build_node_cell_csr_from_incidence,
    point_frame_extrema_from_cell_values,
)
from cfd_bench.core.context import MeshContext
from cfd_bench.core.observability import timed_stage
from cfd_bench.infra.vtk.storage import dataset_dir, read_manifest
from cfd_bench.ingest.vtk.mesh_builder import read_vtk_grid


class VTKMeshClient:
    """Unified VTK backend for CFD and H5 datasets.

    New ingests use ``<vtk_root>/<dataset>/manifest.json`` plus per-zone/frame
    ``.vtu`` files.  The old direct ``vtk_file=...`` mode remains readable for
    compatibility, but the benchmark backend uses the manifest layout.
    """

    def __init__(self, root_path: Optional[str] = None):
        self.root_path = root_path
        self.vtk_mesh = None
        self.ctx: Optional[MeshContext] = None
        self._manifest: Dict = {}
        self._legacy_file: Optional[str] = None
        self._grids: Dict[Tuple[str, int], object] = {}
        self._arrays: Dict[Tuple[str, int, str, str], np.ndarray] = {}
        self._bounds_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._locator_cache: Dict[Tuple[str, int], object] = {}
        self._surface_cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._source_index_cache: Dict[Tuple[str, int, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._cfd_node_cell_csr: Optional[NodeCellCSR] = None

    def close(self):
        self.vtk_mesh = None
        self.ctx = None
        self._manifest = {}
        self._legacy_file = None
        self._grids.clear()
        self._arrays.clear()
        self._bounds_cache.clear()
        self._locator_cache.clear()
        self._surface_cache.clear()
        self._source_index_cache.clear()
        self._cfd_node_cell_csr = None

    def connect(
        self,
        dataset_key: str,
        step: int,
        zone: str = "0_Fluid",
        vtk_file: Optional[str] = None,
        root_path: Optional[str] = None,
        **kwargs,
    ) -> MeshContext:
        root = root_path or self.root_path
        self._legacy_file = None
        if root is not None:
            try:
                self._manifest = read_manifest(root, dataset_key)
            except FileNotFoundError:
                self._manifest = {}
        if self._manifest:
            zones = self._manifest.get("zones", {})
            if zone not in zones:
                if zone == "0_Fluid" and self._manifest.get("primary_zone") in zones:
                    zone = str(self._manifest["primary_zone"])
                else:
                    raise FileNotFoundError(
                        f"VTK zone={zone!r} not found for {dataset_key}; available={list(zones)}"
                    )
            grid = self._grid(str(zone), int(step))
        else:
            if vtk_file is None:
                raise FileNotFoundError(
                    f"VTK dataset manifest not found for {dataset_key!r} under {root!r}"
                )
            self._legacy_file = str(vtk_file)
            grid = read_vtk_grid(vtk_file)
            self._grids[(str(zone), int(step))] = grid
            self._manifest = {
                "schema_version": 1,
                "dataset_key": dataset_key,
                "dataset_type": "legacy",
                "is_h5": False,
                "primary_zone": str(zone),
                "steps": [int(step)],
                "variables": self._cell_array_names(grid),
                "zones": {
                    str(zone): {
                        "steps": [int(step)],
                        "variables": self._cell_array_names(grid),
                        "nodal_variables": [],
                        "node_count": int(grid.GetNumberOfPoints()),
                        "cell_count": int(grid.GetNumberOfCells()),
                        "files": {},
                        "max_diff": {},
                    }
                },
            }
        self.vtk_mesh = grid
        ctx = MeshContext(dataset_key=dataset_key, step=int(step), zone=str(zone))
        ctx.available_caps.update({"mesh_static", "cell_vars"})
        if self.is_h5_dataset():
            ctx.available_caps.update({"h5_metadata", "node_vars"})
        self.ctx = ctx
        return ctx

    def _require_ctx(self) -> MeshContext:
        if self.ctx is None:
            raise RuntimeError("VTK client is not connected")
        return self.ctx

    @staticmethod
    def _cell_array_names(grid) -> List[str]:
        data = grid.GetCellData()
        out = []
        for i in range(int(data.GetNumberOfArrays())):
            name = data.GetArrayName(i)
            if name and name not in {"dense_cell_id", "source_cell_id", "cell_ids"}:
                out.append(str(name).upper())
        return sorted(set(out))

    def _grid_path(self, zone: str, step: int) -> str:
        zmeta = (self._manifest.get("zones") or {}).get(str(zone), {})
        rel = (zmeta.get("files") or {}).get(str(int(step)))
        if not rel:
            raise FileNotFoundError(
                f"VTK frame not found dataset={self._manifest.get('dataset_key')} zone={zone} step={step}"
            )
        return str(dataset_dir(self.root_path or "", self._manifest["dataset_key"]) / rel)

    def _grid(self, zone: str, step: int):
        key = (str(zone), int(step))
        if key in self._grids:
            return self._grids[key]
        path = self._grid_path(*key)
        with timed_stage("VTK", f"load frame zone={zone} step={step}"):
            grid = read_vtk_grid(path)
        self._grids[key] = grid
        return grid

    def _grid_for(self, step: Optional[int] = None, zone: Optional[str] = None):
        ctx = self._require_ctx()
        z = str(zone or ctx.zone)
        ts = int(ctx.step if step is None else step)
        if self._legacy_file is not None:
            return self._grids[(ctx.zone, ctx.step)]
        return self._grid(z, ts)

    def _numpy_array(self, name: str, *, point: bool = False, step: Optional[int] = None, zone: Optional[str] = None):
        from vtkmodules.util import numpy_support

        ctx = self._require_ctx()
        z = str(zone or ctx.zone)
        ts = int(ctx.step if step is None else step)
        assoc = "point" if point else "cell"
        key = (z, ts, assoc, str(name).upper())
        if key in self._arrays:
            return self._arrays[key]
        grid = self._grid_for(ts, z)
        data = grid.GetPointData() if point else grid.GetCellData()
        arr = data.GetArray(str(name))
        if arr is None:
            arr = data.GetArray(str(name).upper())
        if arr is None:
            return None
        out = np.asarray(numpy_support.vtk_to_numpy(arr))
        self._arrays[key] = out
        return out

    def get_mesh_bounds(self):
        grid = self._grid_for()
        b = grid.GetBounds()
        if b is None:
            return None
        return [float(x) for x in b]

    def get_cell_count(self) -> int:
        return int(self._grid_for().GetNumberOfCells())

    def is_h5_dataset(self) -> bool:
        return bool(self._manifest.get("is_h5") or self._manifest.get("dataset_type") == "h5")

    def list_zones(self) -> List[str]:
        return list((self._manifest.get("zones") or {}).keys())

    def variables_for_zone(self, zone: Optional[str] = None) -> Tuple[str, ...]:
        ctx = self._require_ctx()
        zmeta = (self._manifest.get("zones") or {}).get(str(zone or ctx.zone), {})
        return tuple(str(v).upper() for v in zmeta.get("variables", ()))

    def resolve_w6_scalar(self, candidates: Sequence[str]) -> str:
        available = set(self.variables_for_zone())
        for raw in candidates:
            name = str(raw).upper()
            if name in available:
                return name
        raise RuntimeError(
            f"no usable VTK scalar in zone={self._require_ctx().zone}; available={sorted(available)}"
        )

    def w6_zone_candidates(
        self, dataset_key: Optional[str] = None, preferred_zone: Optional[str] = None, hull_hint: Optional[str] = None
    ) -> List[str]:
        zones = self.list_zones()
        if not zones:
            return []
        ordered: List[str] = []

        def add(z):
            if z and z in zones and z not in ordered:
                ordered.append(z)

        add(hull_hint)
        for keyword in ("hull", "wall", "symmetry"):
            for z in zones:
                if keyword in z.lower():
                    add(z)
        for z in zones:
            if "fluid" not in z.lower():
                add(z)
        add(preferred_zone)
        add(self._manifest.get("primary_zone"))
        for z in zones:
            add(z)
        return ordered

    def get_max_diffs(self, step: Optional[int] = None) -> Dict[str, float]:
        ctx = self._require_ctx()
        ts = int(ctx.step if step is None else step)
        zmeta = (self._manifest.get("zones") or {}).get(ctx.zone, {})
        raw = (zmeta.get("max_diff") or {}).get(str(ts), {})
        return {str(k).upper(): float(v) for k, v in raw.items()}

    def point_query(self, cell_indexes, attribute_name: str, step: Optional[int] = None) -> NDArray[np.float64]:
        arr = self._numpy_array(attribute_name, step=step)
        if arr is None:
            return np.zeros((0,), dtype=np.float64)
        ids = np.asarray([int(x) for x in cell_indexes], dtype=np.int64)
        valid = (ids >= 0) & (ids < arr.shape[0])
        return np.asarray(arr[ids[valid]], dtype=np.float64).reshape(-1)

    def velocity_query(self, cell_indexes: Sequence[int], step: Optional[int] = None) -> NDArray[np.float64]:
        ids = np.asarray([int(x) for x in cell_indexes], dtype=np.int64)
        if ids.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        cols = [self._numpy_array(v, step=step) for v in ("U", "V", "W")]
        if any(a is None for a in cols):
            return np.full((ids.size, 3), np.nan, dtype=np.float64)
        n = min(int(a.shape[0]) for a in cols)
        valid = (ids >= 0) & (ids < n)
        out = np.full((ids.size, 3), np.nan, dtype=np.float64)
        if np.any(valid):
            out[valid] = np.column_stack([np.asarray(a[ids[valid]], dtype=np.float64) for a in cols])
        return out

    def var_value_range(self, attribute_name: str, step: Optional[int] = None) -> Tuple[float, float]:
        arr = self._numpy_array(attribute_name, step=step)
        if arr is None or arr.size == 0:
            return 0.0, 1.0
        finite = np.asarray(arr, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.min(finite)), float(np.max(finite))

    def range_query_var(
        self, lower_bound: float, upper_bound: float, attribute_name: str, step: Optional[int] = None
    ) -> NDArray[np.int32]:
        arr = self._numpy_array(attribute_name, step=step)
        if arr is None:
            return np.zeros((0,), dtype=np.int32)
        vals = np.asarray(arr, dtype=np.float64).reshape(-1)
        lo, hi = sorted((float(lower_bound), float(upper_bound)))
        return np.flatnonzero(np.isfinite(vals) & (vals >= lo) & (vals <= hi)).astype(np.int32, copy=False)

    def _cell_bounds_arrays(self, step: Optional[int] = None, zone: Optional[str] = None):
        ctx = self._require_ctx()
        z = str(zone or ctx.zone)
        ts = int(ctx.step if step is None else step)
        key = (z, ts)
        if key in self._bounds_cache:
            return self._bounds_cache[key]
        grid = self._grid_for(ts, z)
        n = int(grid.GetNumberOfCells())
        mins = np.empty((n, 3), dtype=np.float64)
        maxs = np.empty((n, 3), dtype=np.float64)
        centers = np.empty((n, 3), dtype=np.float64)
        with timed_stage("VTK", f"build cell bounds/centroids zone={z} step={ts} cells={n}"):
            for cid in range(n):
                b = grid.GetCell(cid).GetBounds()
                mins[cid] = (b[0], b[2], b[4])
                maxs[cid] = (b[1], b[3], b[5])
                centers[cid] = 0.5 * (mins[cid] + maxs[cid])
        self._bounds_cache[key] = (mins, maxs, centers)
        return mins, maxs, centers

    def range_query_coord(self, lower_bound: Sequence[float], upper_bound: Sequence[float]) -> NDArray[np.int32]:
        mins, maxs, _ = self._cell_bounds_arrays()
        lo = np.minimum(np.asarray(lower_bound, dtype=np.float64), np.asarray(upper_bound, dtype=np.float64))
        hi = np.maximum(np.asarray(lower_bound, dtype=np.float64), np.asarray(upper_bound, dtype=np.float64))
        mask = np.all(mins >= lo, axis=1) & np.all(maxs <= hi, axis=1)
        return np.flatnonzero(mask).astype(np.int32, copy=False)

    def _locator(self):
        ctx = self._require_ctx()
        key = (ctx.zone, ctx.step)
        if key in self._locator_cache:
            return self._locator_cache[key]
        import vtk

        with timed_stage("VTK", f"build static cell locator zone={ctx.zone} step={ctx.step}"):
            locator = vtk.vtkStaticCellLocator()
            locator.SetDataSet(self._grid_for())
            locator.BuildLocator()
        self._locator_cache[key] = locator
        return locator

    def point_intersection(self, points: NDArray[np.float64]) -> NDArray[np.int32]:
        locator = self._locator()
        n = int(self._grid_for().GetNumberOfCells())
        out = []
        for point in np.asarray(points, dtype=np.float64).reshape(-1, 3):
            cid = int(locator.FindCell(point))
            if 0 <= cid < n:
                out.append(cid)
        return np.asarray(out, dtype=np.int32)

    def line_intersection(self, line_start: Sequence[float], line_end: Sequence[float]) -> NDArray[np.int32]:
        import vtk

        locator = self._locator()
        tmp_points = vtk.vtkPoints()
        tmp_cells = vtk.vtkIdList()
        locator.IntersectWithLine(line_start, line_end, 1e-6, tmp_points, tmp_cells)
        return np.fromiter(
            (int(tmp_cells.GetId(i)) for i in range(tmp_cells.GetNumberOfIds())),
            dtype=np.int32,
            count=tmp_cells.GetNumberOfIds(),
        )

    def plane_intersection(self, plane_origin: Sequence[float], plane_norm: Sequence[float]) -> NDArray[np.int32]:
        mins, maxs, centers = self._cell_bounds_arrays()
        n = np.asarray(plane_norm, dtype=np.float64)
        norm = float(np.linalg.norm(n))
        if norm <= 1e-15:
            return np.zeros((0,), dtype=np.int32)
        n /= norm
        origin = np.asarray(plane_origin, dtype=np.float64)
        extents = 0.5 * (maxs - mins)
        signed = (centers - origin) @ n
        radius = extents @ np.abs(n)
        return np.flatnonzero(np.abs(signed) <= radius + 1e-9).astype(np.int32, copy=False)

    def extract_submesh(self, cell_indexes: Sequence[int], mesh_handle=None):
        import vtk

        ids = np.asarray(sorted(set(int(x) for x in cell_indexes if int(x) >= 0)), dtype=np.int64)
        if ids.size == 0:
            return None
        source = self._grid_for() if mesh_handle is None else mesh_handle
        id_array = vtk.vtkIdTypeArray()
        id_array.SetNumberOfValues(int(ids.size))
        for i, cid in enumerate(ids.tolist()):
            id_array.SetValue(i, int(cid))
        node = vtk.vtkSelectionNode()
        node.SetFieldType(vtk.vtkSelectionNode.CELL)
        node.SetContentType(vtk.vtkSelectionNode.INDICES)
        node.SetSelectionList(id_array)
        sel = vtk.vtkSelection()
        sel.AddNode(node)
        extract = vtk.vtkExtractSelection()
        extract.SetInputData(0, source)
        extract.SetInputData(1, sel)
        extract.Update()
        out = vtk.vtkUnstructuredGrid()
        out.ShallowCopy(extract.GetOutput())
        return out

    @staticmethod
    def _contour(mesh, variable_name: str, iso_value: float):
        import vtk

        if mesh is None or int(mesh.GetNumberOfCells()) == 0:
            return None
        source = mesh
        if source.GetPointData().GetArray(str(variable_name).upper()) is None:
            conv = vtk.vtkCellDataToPointData()
            conv.SetInputData(source)
            conv.PassCellDataOn()
            conv.Update()
            source = conv.GetOutput()
        contour = vtk.vtkContourFilter()
        contour.SetInputData(source)
        contour.SetValue(0, float(iso_value))
        contour.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, str(variable_name).upper())
        contour.Update()
        return contour.GetOutput()

    def isosurface_extraction(self, variable_name: str, iso_value: float, step: Optional[int] = None):
        return self._contour(self._grid_for(step=step), variable_name, iso_value)

    def isosurface_from_submesh(self, mesh, variable_name: str, iso_value: float, step: Optional[int] = None):
        return self._contour(mesh, variable_name, iso_value)

    @staticmethod
    def _stable_normal(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(pts) >= 3:
            p0 = pts[0]
            for i in range(1, len(pts) - 1):
                for j in range(i + 1, len(pts)):
                    n = np.cross(pts[i] - p0, pts[j] - p0)
                    length = float(np.linalg.norm(n))
                    if length > 1e-15:
                        return n / length
        if len(pts) >= 2:
            tangent = pts[1] - pts[0]
            length = float(np.linalg.norm(tangent))
            if length > 1e-15:
                tangent /= length
                axes = np.eye(3)
                axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
                n = np.cross(tangent, axis)
                nlen = float(np.linalg.norm(n))
                if nlen > 1e-15:
                    return n / nlen
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def surface_cells_and_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        ctx = self._require_ctx()
        key = (ctx.zone, ctx.step)
        if key in self._surface_cache:
            return self._surface_cache[key]
        grid = self._grid_for()
        n_cells = int(grid.GetNumberOfCells())
        normals = np.empty((n_cells, 3), dtype=np.float64)
        with timed_stage("VTK", f"build surface normals zone={ctx.zone} step={ctx.step} cells={n_cells}"):
            for cid in range(n_cells):
                cell = grid.GetCell(cid)
                pts = np.asarray([grid.GetPoint(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())])
                normals[cid] = self._stable_normal(pts)
        result = (np.arange(n_cells, dtype=np.int32), normals)
        self._surface_cache[key] = result
        return result

    def surface_norm(self, mesh_handle=None) -> NDArray[np.float64]:
        if mesh_handle is None:
            return self.surface_cells_and_normals()[1]
        grid = mesh_handle
        n_cells = int(grid.GetNumberOfCells())
        normals = np.empty((n_cells, 3), dtype=np.float64)
        for cid in range(n_cells):
            cell = grid.GetCell(cid)
            pts = np.asarray([grid.GetPoint(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())])
            normals[cid] = self._stable_normal(pts)
        return normals

    def compute_qcriterion_roi(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        tau: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
        import vtk
        from vtkmodules.util import numpy_support

        cells = self.range_query_coord(lower_bound, upper_bound)
        if cells.size == 0:
            return cells, np.zeros((0,), dtype=np.float64)
        sub = self.extract_submesh(cells)
        if sub is None:
            return cells, np.zeros((0,), dtype=np.float64)
        from cfd_bench.ingest.vtk.mesh_builder import ensure_point_velocity

        ensure_point_velocity(sub)
        vel = sub.GetPointData().GetArray("Velocity")
        if vel is None:
            return cells, np.zeros((len(cells),), dtype=np.float64)
        gf = vtk.vtkGradientFilter()
        gf.SetInputData(sub)
        gf.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Velocity")
        gf.SetComputeQCriterion(True)
        gf.SetQCriterionArrayName("Q-criterion")
        gf.Update()
        out = gf.GetOutput()
        q = out.GetPointData().GetArray("Q-criterion")
        if q is None:
            q = out.GetCellData().GetArray("Q-criterion")
        if q is None:
            return cells, np.zeros((len(cells),), dtype=np.float64)
        vals = np.asarray(numpy_support.vtk_to_numpy(q), dtype=np.float64).reshape(-1)
        return cells, vals

    # ------------------------------------------------------------------ W9
    def _source_cell_ids(self, step: Optional[int] = None) -> np.ndarray:
        arr = self._numpy_array("source_cell_id", step=step)
        if arr is None:
            return np.arange(1, int(self._grid_for(step=step).GetNumberOfCells()) + 1, dtype=np.int64)
        return np.asarray(arr, dtype=np.int64).reshape(-1)

    def _source_node_ids(self, step: Optional[int] = None) -> np.ndarray:
        arr = self._numpy_array("source_node_id", point=True, step=step)
        if arr is None:
            return np.arange(1, int(self._grid_for(step=step).GetNumberOfPoints()) + 1, dtype=np.int64)
        return np.asarray(arr, dtype=np.int64).reshape(-1)

    def _element_ids_in_coordinate_range(self, lower_bound, upper_bound) -> np.ndarray:
        _, _, centers = self._cell_bounds_arrays()
        lo = np.minimum(np.asarray(lower_bound, dtype=np.float64), np.asarray(upper_bound, dtype=np.float64))
        hi = np.maximum(np.asarray(lower_bound, dtype=np.float64), np.asarray(upper_bound, dtype=np.float64))
        mask = np.all(centers >= lo, axis=1) & np.all(centers <= hi, axis=1)
        return self._source_cell_ids()[mask].astype(np.int64, copy=False)

    def h5_element_ids_in_coordinate_range(self, lower_bound, upper_bound) -> NDArray[np.int64]:
        return self._element_ids_in_coordinate_range(lower_bound, upper_bound)

    def cfd_element_ids_in_coordinate_range(self, lower_bound, upper_bound) -> NDArray[np.int64]:
        return self._element_ids_in_coordinate_range(lower_bound, upper_bound)

    # ----------------------------------------------------------------- W10
    @staticmethod
    def _stats(values: np.ndarray, position: str) -> Dict[str, object]:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ValueError("no finite values")
        return {
            "position": str(position),
            "count": int(vals.size),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "stddev": float(np.std(vals)),
        }

    def frame_statistics(self, attribute_name: Optional[str] = None, step: Optional[int] = None):
        ctx = self._require_ctx()
        ts = int(ctx.step if step is None else step)
        zmeta = (self._manifest.get("zones") or {}).get(ctx.zone, {})
        frame_cell = (zmeta.get("frame_cell_variables") or {}).get(str(ts), zmeta.get("variables", ()))
        cell_vars = [str(v).upper() for v in frame_cell]
        frame_node = (zmeta.get("frame_node_variables") or {}).get(str(ts), zmeta.get("nodal_variables", ()))
        node_vars = {str(v).upper() for v in frame_node}
        wanted = [str(attribute_name).upper()] if attribute_name is not None else cell_vars
        result = {}
        for var in wanted:
            if var in node_vars:
                arr = self._numpy_array(var, point=True, step=ts)
                if arr is not None:
                    result[var] = self._stats(arr, "node")
                    continue
            arr = self._numpy_array(var, step=ts)
            if arr is not None:
                result[var] = self._stats(arr, "cell")
        if not result:
            raise ValueError(f"no values for frame={ts}")
        return result

    def cfd_frame_statistics(self, attribute_name: Optional[str] = None, step: Optional[int] = None):
        ctx = self._require_ctx()
        ts = int(ctx.step if step is None else step)
        variables = list(self.variables_for_zone())
        wanted = [str(attribute_name).upper()] if attribute_name is not None else variables
        result = {}
        for var in wanted:
            arr = self._numpy_array(var, step=ts)
            if arr is not None:
                result[var] = self._stats(arr, "cell")
        if not result:
            raise ValueError(f"no CFD values for frame={ts}")
        return result

    # ----------------------------------------------------------------- W11
    def h5_nodal_variables(self) -> Tuple[str, ...]:
        return tuple(str(v).upper() for v in self._manifest.get("nodal_variables", ()))

    def h5_point_ids(self) -> np.ndarray:
        return self._source_node_ids().astype(np.int64, copy=False)

    def _source_to_dense(self, source_ids: Sequence[int], *, step: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        ctx = self._require_ctx()
        ts = int(ctx.step if step is None else step)
        key = (ctx.zone, ts, "source_node")
        if key not in self._source_index_cache:
            src = self._source_node_ids(step=ts)
            order = np.argsort(src, kind="stable")
            self._source_index_cache[key] = (src[order], order.astype(np.int64, copy=False))
        sorted_src, dense_order = self._source_index_cache[key]
        req = np.asarray([int(x) for x in source_ids], dtype=np.int64)
        pos = np.searchsorted(sorted_src, req)
        valid = (pos >= 0) & (pos < sorted_src.size)
        clipped = np.clip(pos, 0, max(sorted_src.size - 1, 0))
        if sorted_src.size:
            valid &= sorted_src[clipped] == req
        dense = np.full(req.shape, -1, dtype=np.int64)
        if sorted_src.size:
            dense[valid] = dense_order[pos[valid]]
        return req, dense

    def h5_point_frame_extrema(self, point_ids: Sequence[int], attribute_name: str):
        var = str(attribute_name).upper()
        if var not in self.h5_nodal_variables():
            raise ValueError(f"VTK H5 variable {var} is not available as genuine nodal data")
        steps = [int(x) for x in self._manifest.get("steps", ())]
        req, dense = self._source_to_dense(point_ids, step=steps[0] if steps else None)
        valid = dense >= 0
        if not np.any(valid):
            return {}
        mins = np.full(req.shape, np.inf, dtype=np.float64)
        maxs = np.full(req.shape, -np.inf, dtype=np.float64)
        for ts in steps:
            arr = self._numpy_array(var, point=True, step=ts)
            if arr is None:
                continue
            vals = np.asarray(arr[dense[valid]], dtype=np.float64)
            idx = np.flatnonzero(valid)
            finite = np.isfinite(vals)
            mins[idx[finite]] = np.minimum(mins[idx[finite]], vals[finite])
            maxs[idx[finite]] = np.maximum(maxs[idx[finite]], vals[finite])
        return {
            int(req[i]): (float(mins[i]), float(maxs[i]))
            for i in range(req.size)
            if np.isfinite(mins[i]) and np.isfinite(maxs[i])
        }

    def cfd_variables(self) -> Tuple[str, ...]:
        return self.variables_for_zone()

    def cfd_point_ids(self) -> range:
        ctx = self._require_ctx()
        zmeta = (self._manifest.get("zones") or {}).get(ctx.zone, {})
        count = int(zmeta.get("node_count") or self._grid_for().GetNumberOfPoints())
        return range(1, count + 1)

    def _ensure_cfd_node_cell_csr(self) -> NodeCellCSR:
        if self._cfd_node_cell_csr is not None:
            return self._cfd_node_cell_csr
        from vtkmodules.util import numpy_support

        grid = self._grid_for()
        cells = grid.GetCells()
        offsets = np.asarray(numpy_support.vtk_to_numpy(cells.GetOffsetsArray()), dtype=np.int64)
        conn = np.asarray(numpy_support.vtk_to_numpy(cells.GetConnectivityArray()), dtype=np.int64)
        lengths = np.diff(offsets)
        dense_cells = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        with timed_stage(
            "VTK W11",
            f"build runtime node-to-cell projection dataset={self._require_ctx().dataset_key} zone={self._require_ctx().zone}",
        ):
            self._cfd_node_cell_csr = build_node_cell_csr_from_incidence(
                conn,
                dense_cells,
                int(grid.GetNumberOfPoints()),
            )
        return self._cfd_node_cell_csr

    def prepare_cfd_point_queries(self) -> None:
        self._ensure_cfd_node_cell_csr()

    def cfd_point_frame_extrema(self, point_ids: Sequence[int], attribute_name: str):
        csr = self._ensure_cfd_node_cell_csr()
        steps = [int(x) for x in self._manifest.get("steps", ())]
        var = str(attribute_name).upper()
        return point_frame_extrema_from_cell_values(
            csr,
            point_ids,
            steps,
            lambda step, cell_ids: self.point_query(cell_ids, var, step=step),
        )
