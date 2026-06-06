from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from cfd_bench.core.context import MeshContext
from cfd_bench.core.types import LiteMesh, LitePolyData


class VTKMeshClient:
    """VTK baseline client — behavior identical to VTK_Interface."""

    def __init__(self):
        self.vtk_mesh = None
        self.ctx: Optional[MeshContext] = None

    def close(self):
        self.vtk_mesh = None
        self.ctx = None

    def connect(
        self,
        dataset_key: str,
        step: int,
        zone: str = "0_Fluid",
        vtk_file: Optional[str] = None,
        **kwargs,
    ) -> MeshContext:
        if vtk_file is None:
            raise ValueError("VTKMeshClient.connect requires vtk_file=...")
        self.vtk_mesh = self._vtk_connect(vtk_file)
        ctx = MeshContext(dataset_key=dataset_key, step=int(step), zone=zone)
        ctx.available_caps.update({"mesh_static", "cell_vars"})
        self.ctx = ctx
        return ctx

    def _vtk_connect(self, file):
        from vtk import vtkUnstructuredGridReader

        if file.lower().endswith(".vtk"):
            reader = vtkUnstructuredGridReader()
            reader.SetFileName(file)
            reader.Update()
            return reader.GetOutput()
        raise ValueError(f"Error, file {file} is not a .vtk file.")

    def point_query(self, cell_indexes, attribute_name: str, step: Optional[int] = None) -> NDArray[np.float64]:
        array = self.vtk_mesh.GetCellData().GetArray(attribute_name)
        if not array:
            return np.array([], dtype=np.float64)
        n = int(array.GetNumberOfTuples())
        vals = []
        for cell_id in cell_indexes:
            cid = int(cell_id)
            if 0 <= cid < n:
                vals.append(array.GetValue(cid))
        return np.array(vals, dtype=np.float64)

    def range_query_var(
        self, lower_bound: float, upper_bound: float, attribute_name: str, step: Optional[int] = None
    ) -> NDArray[np.int32]:
        array = self.vtk_mesh.GetCellData().GetArray(attribute_name)
        if not array:
            return np.array([], dtype=np.int32)
        cell_indexes = []
        for i in range(array.GetNumberOfTuples()):
            value = array.GetValue(i)
            if lower_bound <= value <= upper_bound:
                cell_indexes.append(i)
        return np.array(cell_indexes, dtype=np.int32)

    def range_query_coord(self, lower_bound: Sequence[float], upper_bound: Sequence[float]) -> NDArray[np.int32]:
        cell_indexes = []
        for cell_id in range(self.vtk_mesh.GetNumberOfCells()):
            cell = self.vtk_mesh.GetCell(cell_id)
            points = cell.GetPoints()
            inside = True
            for i in range(points.GetNumberOfPoints()):
                point = points.GetPoint(i)
                if not (
                    lower_bound[0] <= point[0] <= upper_bound[0]
                    and lower_bound[1] <= point[1] <= upper_bound[1]
                    and lower_bound[2] <= point[2] <= upper_bound[2]
                ):
                    inside = False
                    break
            if inside:
                cell_indexes.append(cell_id)
        return np.array(cell_indexes, dtype=np.int32)

    def point_intersection(self, points: NDArray[np.float64]) -> NDArray[np.int32]:
        from vtk import vtkCellLocator

        cell_locator = vtkCellLocator()
        cell_locator.SetDataSet(self.vtk_mesh)
        cell_locator.BuildLocator()
        n_cells = int(self.vtk_mesh.GetNumberOfCells())
        cell_indexes = []
        for point in points:
            cid = int(cell_locator.FindCell(point))
            if 0 <= cid < n_cells:
                cell_indexes.append(cid)
        return np.array(cell_indexes, dtype=np.int32)

    def line_intersection(self, line_start: Sequence[float], line_end: Sequence[float]) -> NDArray[np.int32]:
        from vtk import vtkCellLocator, vtkIdList, vtkPoints

        tmp_points = vtkPoints()
        tmp_cells = vtkIdList()
        tolerance = 1e-6
        cell_locator = vtkCellLocator()
        cell_locator.SetDataSet(self.vtk_mesh)
        cell_locator.BuildLocator()
        cell_locator.IntersectWithLine(line_start, line_end, tolerance, tmp_points, tmp_cells)
        return np.array([tmp_cells.GetId(i) for i in range(tmp_cells.GetNumberOfIds())], dtype=np.int32)

    def plane_intersection(self, plane_origin: Sequence[float], plane_norm: Sequence[float]) -> NDArray[np.int32]:
        from vtk import vtkDataObject, vtkExtractGeometry, vtkIntArray, vtkPlane

        if not self.vtk_mesh.GetCellData().GetArray("cell_ids"):
            cell_ids_array = vtkIntArray()
            cell_ids_array.SetName("cell_ids")
            cell_ids_array.SetNumberOfComponents(1)
            cell_ids_array.SetNumberOfTuples(self.vtk_mesh.GetNumberOfCells())
            for i in range(self.vtk_mesh.GetNumberOfCells()):
                cell_ids_array.SetValue(i, i)
            self.vtk_mesh.GetCellData().AddArray(cell_ids_array)

        plane = vtkPlane()
        plane.SetOrigin(plane_origin)
        plane.SetNormal(plane_norm)
        extractor = vtkExtractGeometry()
        extractor.SetInputData(self.vtk_mesh)
        extractor.SetImplicitFunction(plane)
        extractor.SetExtractInside(0)
        extractor.SetExtractOnlyBoundaryCells(1)
        extractor.Update()
        intersected_cells = extractor.GetOutput()
        original_ids = intersected_cells.GetCellData().GetArray("cell_ids")
        if original_ids:
            return np.array([original_ids.GetValue(i) for i in range(original_ids.GetNumberOfTuples())], dtype=np.int32)
        return np.array([], dtype=np.int32)

    def extract_submesh(self, cell_indexes: Sequence[int], mesh_handle=None):
        from vtk import vtkExtractCells, vtkIdList

        if len(cell_indexes) == 0:
            return None
        extract_cells = vtkExtractCells()
        extract_cells.SetInputData(self.vtk_mesh)
        cell_ids_vtk = vtkIdList()
        for cid in cell_indexes:
            cell_ids_vtk.InsertNextId(int(cid))
        extract_cells.SetCellList(cell_ids_vtk)
        extract_cells.Update()
        return extract_cells.GetOutput()

    def isosurface_extraction(self, variable_name: str, iso_value: float, step: Optional[int] = None):
        from vtk import vtkContourFilter, vtkDataObject

        contour = vtkContourFilter()
        contour.SetInputData(self.vtk_mesh)
        contour.SetValue(0, iso_value)
        contour.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, variable_name)
        try:
            contour.Update()
            return contour.GetOutput()
        except Exception:
            return None

    def surface_norm(self, mesh_handle=None) -> NDArray[np.float64]:
        mesh = mesh_handle if mesh_handle is not None else self.vtk_mesh
        normals = []
        for cellid in range(mesh.GetNumberOfCells()):
            cell = mesh.GetCell(cellid)
            if cell.GetNumberOfPoints() >= 3:
                p0 = mesh.GetPoint(cell.GetPointId(0))
                p1 = mesh.GetPoint(cell.GetPointId(1))
                p2 = mesh.GetPoint(cell.GetPointId(2))
                v1 = [p1[i] - p0[i] for i in range(3)]
                v2 = [p2[i] - p0[i] for i in range(3)]
                normal = [
                    v1[1] * v2[2] - v1[2] * v2[1],
                    v1[2] * v2[0] - v1[0] * v2[2],
                    v1[0] * v2[1] - v1[1] * v2[0],
                ]
                normals.append(normal)
            else:
                normals.append([0.0, 0.0, 0.0])
        return np.array(normals, dtype=np.float64)

    def compute_qcriterion_roi(
        self,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        tau: Optional[float] = None,
        step: Optional[int] = None,
    ) -> Tuple[NDArray[np.int32], NDArray[np.float64]]:
        raise NotImplementedError("VTK compute_qcriterion_roi: use TileDB/IoTDB derived data")
