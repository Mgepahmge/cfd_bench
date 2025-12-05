from vtk import vtkCellLocator, vtkPlane, vtkPolyData, vtkUnstructuredGrid, vtkIdList, vtkUnstructuredGridReader, vtkOBBTree, vtkIntArray, vtkExtractGeometry, vtkExtractCells, vtkContourFilter, vtkDataObject, vtkPoints
import numpy as np
from numpy.typing import NDArray

class VTK_Interface:

    def vtk_connect(self, file) -> vtkUnstructuredGrid:
        if file.lower().endswith('.vtk'):
            # Create a reader for legacy VTK files
            reader = vtkUnstructuredGridReader()  # For unstructured grids
            # reader = vtk.vtkDataSetReader()        # For general VTK files (structured/unstructured)
            # Set the filename
            reader.SetFileName(file)
            # Read the file
            reader.Update()  # Triggers the reading process
            # Get the output data
            vtk_mesh = reader.GetOutput()  # For unstructured grids
            # Access key properties
            print(f"Number of points: {vtk_mesh.GetNumberOfPoints()}")
            print(f"Number of cells: {vtk_mesh.GetNumberOfCells()}")
            return vtk_mesh
        else:
            print(f"Error, file {file} is not a .vtk file.")
            return -1

    def point_query(self, vtk_mesh:vtkUnstructuredGrid, cell_indexes:np.array, attribute_name:str) -> NDArray[np.float64]:
        # 数据在 Cell Data
        array = vtk_mesh.GetCellData().GetArray(attribute_name)
        if not array:
            return []
        attribute_values = [array.GetValue(cell_id) for cell_id in cell_indexes]
        return attribute_values

    def range_query_var(self, vtk_mesh:vtkUnstructuredGrid, lower_bound:float, upper_bound:float ,attribute_name:str): # ranges is a 2-D array, with len(ranges = len(attribute_names))
        array = vtk_mesh.GetCellData().GetArray(attribute_name)
        if not array:
            return []
        cell_indexes = []
        for i in range(array.GetNumberOfTuples()):
            value = array.GetValue(i)
            if lower_bound <= value <= upper_bound:
                cell_indexes.append(i)
        return np.array(cell_indexes, dtype=np.int32)
    
    def range_query_coord(self, vtk_mesh:vtkUnstructuredGrid, lower_bound:NDArray[np.float64], upper_bound:NDArray[np.float64]):
        cell_indexes = []
        for cell_id in range(vtk_mesh.GetNumberOfCells()):
            cell = vtk_mesh.GetCell(cell_id)
            points = cell.GetPoints()
            inside = True
            for i in range(points.GetNumberOfPoints()):
                point = points.GetPoint(i)
                if not (lower_bound[0] <= point[0] <= upper_bound[0] and
                        lower_bound[1] <= point[1] <= upper_bound[1] and
                        lower_bound[2] <= point[2] <= upper_bound[2]):
                    inside = False
                    break
            if inside:
                cell_indexes.append(cell_id)
        return np.array(cell_indexes, dtype=np.int32)

    def vtk_point_intersection(self, vtk_mesh:vtkUnstructuredGrid, points:NDArray[np.float64]):
        cell_locator = vtkCellLocator()
        cell_locator.SetDataSet(vtk_mesh)
        cell_locator.BuildLocator()
        
        cell_indexes = []
        for point in points:
            cell_id = cell_locator.FindCell(point)
            cell_indexes.append(cell_id)
        
        return np.array(cell_indexes, dtype=np.int32)

    def vtk_line_intersection(self, vtk_mesh:vtkUnstructuredGrid, line_start:NDArray[np.float64], line_end:NDArray[np.float64]):
        # Variables to store intersection results
        tmp_points = vtkPoints()  # To store intersection points
        tmp_cells = vtkIdList()  # To store intersected cell IDs
        tolerance = 1e-6  # Tolerance for intersection
        
        cell_locator = vtkCellLocator()
        cell_locator.SetDataSet(vtk_mesh)
        cell_locator.BuildLocator()  # Preprocess to build the octree
        
        # Compute intersections
        cell_locator.IntersectWithLine(line_start, line_end, tolerance, tmp_points, tmp_cells)
        
        # Extract intersection points and corresponding cell IDs
        
        intersection_cell_count = tmp_cells.GetNumberOfIds()

        cell_indexes = []
            
        for i in range(0, intersection_cell_count):
            cell_indexes.append(tmp_cells.GetId(i))
        
        return np.array(cell_indexes, dtype=np.int32)
        
    def vtk_plane_intersection(self, vtk_mesh:vtkUnstructuredGrid, plane_origin:NDArray[np.float64], plane_norm:NDArray[np.float64]):
        if not vtk_mesh.GetCellData().GetArray("cell_ids"):
            cell_ids_array = vtkIntArray()
            cell_ids_array.SetName("cell_ids")
            cell_ids_array.SetNumberOfComponents(1)
            cell_ids_array.SetNumberOfTuples(vtk_mesh.GetNumberOfCells())
            for i in range(vtk_mesh.GetNumberOfCells()):
                cell_ids_array.SetValue(i, i)  # 假设 ID 为单元格索引
            vtk_mesh.GetCellData().AddArray(cell_ids_array)

        plane = vtkPlane()
        plane.SetOrigin(plane_origin)
        plane.SetNormal(plane_norm)

        # Use vtkExtractGeometry to get intersected cells
        extractor = vtkExtractGeometry()
        extractor.SetInputData(vtk_mesh)
        extractor.SetImplicitFunction(plane)
        extractor.SetExtractInside(0)  # 0 = extract boundary (intersected) cells
        extractor.SetExtractOnlyBoundaryCells(1)  # Only cells intersected by plane
        extractor.Update()

        # Get the output containing intersected cells
        intersected_cells = extractor.GetOutput()
        
        # Extract cell IDs from the original grid
        original_ids = intersected_cells.GetCellData().GetArray("cell_ids")

        if original_ids:
            cell_indexes = []
            for i in range(original_ids.GetNumberOfTuples()):
                cell_indexes.append(original_ids.GetValue(i))
            print(f"Extracted cell Successfully, total {len(cell_indexes)} cells intersected with the plane.")
        else:
            print("No 'cell_indexes' array found in intersected cells.")

        return np.array(cell_indexes, dtype=np.int32)
        

    def vtk_extract_submesh(cell_indexes: NDArray[np.int32], vtk_mesh:vtkUnstructuredGrid) -> vtkUnstructuredGrid:
        # cell_ids: 列表或数组，包含要提取的细胞 ID
        if len(cell_indexes) == 0:
            print("No cell_indexes provided")
            return None
        
        # 使用 vtkExtractCells 提取子网格
        extract_cells = vtkExtractCells()
        extract_cells.SetInputData(vtk_mesh)
        cell_ids_vtk = vtkIdList()
        for cid in cell_indexes:
            cell_ids_vtk.InsertNextId(cid)
        extract_cells.SetCellList(cell_ids_vtk)
        extract_cells.Update()
        sub_mesh = extract_cells.GetOutput()
        
        print(f"子网格：{sub_mesh.GetNumberOfPoints()} 个点，{sub_mesh.GetNumberOfCells()} 个单元格")
        return sub_mesh

    def vtk_isosurface_extraction(vtk_mesh:vtkUnstructuredGrid, variable_name:str, iso_value:float) -> vtkPolyData:
        if vtk_mesh is None:
            print("网格为空，无法提取等值面")
            return None
        contour = vtkContourFilter()
        contour.SetInputData(vtk_mesh)
        contour.SetValue(0, iso_value)
        contour.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, variable_name)
        try:
            contour.Update()
            iso_surface = contour.GetOutput()
            print(f"等值面：{iso_surface.GetNumberOfPoints()} 个点，{iso_surface.GetNumberOfCells()} 个单元格")
        except Exception as e:
            print(f"Contour filter failed: {e}")
            return None
        return iso_surface

    def vtk_surface_norm(vtk_mesh:vtkUnstructuredGrid) -> NDArray[np.float64]: # returns a 2-D array
        normals = []
        for cellid in range(vtk_mesh.GetNumberOfCells()):
            cell = vtk_mesh.GetCell(cellid)
            # 手动计算法向量（使用前3个点）
            if cell.GetNumberOfPoints() >= 3:
                p0 = vtk_mesh.GetPoint(cell.GetPointId(0))
                p1 = vtk_mesh.GetPoint(cell.GetPointId(1))
                p2 = vtk_mesh.GetPoint(cell.GetPointId(2))
                v1 = [p1[i] - p0[i] for i in range(3)]
                v2 = [p2[i] - p0[i] for i in range(3)]
                normal = [
                    v1[1]*v2[2] - v1[2]*v2[1],
                    v1[2]*v2[0] - v1[0]*v2[2],
                    v1[0]*v2[1] - v1[1]*v2[0]
                ]
                normals.append(normal)
            else:
                normals.append(None)
        return np.array(normals, dtype=np.float64)
