
from tkinter import Variable
import vtk
import numpy as np
from vtkmodules.util import numpy_support
import os
from .Dat_Data_Decoder import CAE_Decoder
from .Zone import Zone_3D
from tqdm import tqdm
from vtk import vtkCell, vtkCellLocator, vtkCellTypes, vtkPlane, vtkPolyData, vtkIdTypeArray, vtkUnstructuredGrid, vtkIdList
import time
from collections import defaultdict
from pathlib import Path
import sys
# import pyvista as pv

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkCommonColor import vtkNamedColors

from cfd_bench.core.paths import resolve_vtk_dir, resolve_vtk_hull_dir

DEBUG = False
SAVE_VTK = True
DATA_SOURCE = 0 # 0 for .dat and 1 for .vtk

def main(input_path = None):


    if DATA_SOURCE == 0:
        if input_path is not None:
            path = input_path
        else:
            project_root = Path(__file__).resolve().parents[4]
        path = project_root / "data" / "Kvlcc2_351k" / "Postprocessing"
        print("Decoding Post-processing data")
        if not os.path.exists(path):
            print(f"Error, path does not exist: {path}")
            return
        elif not os.path.isdir(path):
            print(f"Error, please input the directory to which the .dat files are located: {path}")
            
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            if os.path.isfile(filepath):  # Process Files only
                # Check if the filename ends with .dat
                if file.lower().endswith('.dat'):

                    fluid_mesh, hull_mesh = load_unstructured_grid_from_dat(filepath)

                    if SAVE_VTK == True:
                        ''' 
                            Write unstructured grid to .vtk file.
                        '''
                        output_fluid_file_dir = resolve_vtk_dir()
                        output_hull_file_dir = resolve_vtk_hull_dir()
                        os.makedirs(output_fluid_file_dir, exist_ok=True)
                        os.makedirs(output_hull_file_dir, exist_ok=True)

                        file_prefix_fluid = "Suboff_3258k_GEO_"
                        file_prefix_hull = "Suboff_3258k_hull_"

                        filename_without_suffix = Path(file).with_suffix('')

                        filename_fluid_with_prefix = file_prefix_fluid + os.path.basename(filename_without_suffix)
                        filename_hull_with_prefix = file_prefix_hull + os.path.basename(filename_without_suffix)
                        output_file_fluid = os.path.join(output_fluid_file_dir, filename_fluid_with_prefix)
                        
                        output_file_hull = os.path.join(output_hull_file_dir, filename_hull_with_prefix)
                    
                        write_unstructured_grid_to_vtk_file(fluid_mesh, output_file_fluid)
                        write_unstructured_grid_to_vtk_file(hull_mesh, output_file_hull)
                        
                    
    elif DATA_SOURCE == 1:
        if input_path is not None:
            path = input_path
        else:
            path = 'vtk\\'
        print("Decoding .vtk data")
        if not os.path.exists(path):
            print(f"Error, path does not exist: {path}")
            return
        elif not os.path.isdir(path):
            print(f"Error, please input the directory to which the .dat files are located: {path}")

        for file in os.listdir(path):
            start = time.time()
            filepath = os.path.join(path, file)
            if os.path.isfile(filepath):  # Process Files on
                mesh = load_unstructured_grid_from_vtk_file(filepath)
                end = time.time()
                # print(f"Constructing unstructured grid from .vtk time: {end - start:.6f} seconds...")
                # # Create the cell locator
                # start = end
                # cell_locator = vtk.vtkCellLocator()
                # cell_locator.SetDataSet(mesh)
                # cell_locator.BuildLocator()

                # end = time.time()
                # print(f"Constructing cell locator (Octree) time: {end - start:.6f} seconds...")

                # # Threshold filtering
                # start = end
                # query_variables = ['P']
                # query_thresholds = [[-1.0, 0.0]]
                # workload_threshold_filtering(mesh, query_variables, query_thresholds)
                # end = time.time()
                # print(f"Threshold filtering time: {end - start:.6f} seconds...")
                

                # # Locating cells by points
                # start = end
                # ids = [5000,50000,600000]
                # points = []
                # for i in ids:
                #     central_coordinate = np.zeros(6, dtype = np.float64)
                #     mesh.GetCell(i).GetBounds(central_coordinate)
                #     p = [(central_coordinate[0] + central_coordinate[1])/2, (central_coordinate[2] + central_coordinate[3])/2, (central_coordinate[4] + central_coordinate[5])/2]
                #     points.append(p)
                    
                # workload_locate_cells_by_points(cell_locator, points)
                # end = time.time()
                # print(f"Locating cell by points time: {end - start:.6f} seconds...")
                


def mesh_plane_intersection(mesh: vtkUnstructuredGrid, plane: vtkPlane):

    # Assuming `mesh` is your vtkPolyData/vtkUnstructuredGrid
    locator = vtkCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()  # Preprocess to build the octree
   

    # Find cells intersecting the plane
    cell_ids = vtkIdList()
    locator.FindCellsIntersectingPlane(plane.GetNormal(), plane.GetOrigin(), 1e-6, cell_ids)  # Tolerance for numerical stability

    result_cell_ids = []
    for i in range(cell_ids.GetNumberOfIds()):
        cell_id = cell_ids.GetId(i)
        # cell = mesh.GetCell(cell_id)
        result_cell_ids.append(cell_id)

    return result_cell_ids



def construct_vtk_mesh(zone: Zone_3D):
    start = time.time()
    WRITE_GEO = True
    '''
    This process requires calling function 
    void vtkUnstructuredGrid::SetPolyhedralCells{
        vtkUnsignedCharArray * 	cellTypes,
        vtkCellArray * 	cells,
        vtkCellArray * 	faceLocations,
        vtkCellArray * 	faces 
    }	
    For the sake of efficiency, we need to construct all input arrays manually
    '''
    
    size_cells = zone.Element_count
    size_cell_types = zone.Element_count
    size_face_locations = zone.Element_count
    size_faces = zone.Face_count


    print(f"Estimated cells size: {size_cells}")
    print(f"Estimated cellTypes size: {size_cell_types}")
    print(f"Estimated faces size: {size_faces}")
    print(f"Estimated faceLocations size: {size_face_locations}")

    # Inserting points...
    points = np.stack([zone.Node_Coordinates[0], zone.Node_Coordinates[1], zone.Node_Coordinates[2]]).T
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=0))
    # for p in points:
    #      vtk_points.InsertNextPoint(p)

    # constructing faces & face locations
    print("Constructing faces...")
    vtk_faces = vtk.vtkCellArray()
    vtk_faces.SetNumberOfCells(size_faces)
    
    vtk_face_locations = vtk.vtkCellArray()
    vtk_face_locations.SetNumberOfCells(size_face_locations)
    
    # Constructing faces
    for f in tqdm(range(0, zone.Face_count)):
        tmp_face_nodes = zone.FN[f]
        vtk_faces.InsertNextCell(len(tmp_face_nodes), list(tmp_face_nodes))
        

    # Constructing face locations
    for e in tqdm(range(0, zone.Element_count)):
        # Insert numFaces
        tmp_element_faces = zone.EF[e]
        vtk_face_locations.InsertNextCell(len(tmp_element_faces), list(tmp_element_faces))


    # constructing cells
    vtk_cell_types = vtk.vtkUnsignedCharArray()
    
    print("Constructing cells...")
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetNumberOfCells(size_cells)
    for e in tqdm(range(0, zone.Element_count)):
        vtk_cell_types.InsertNextValue(vtk.VTK_POLYHEDRON)
        tmp_EN = zone.EN[e]
        vtk_cells.InsertNextCell(len(tmp_EN), list(tmp_EN))


    # So far all the inputs for SetCells() are constructed


    mesh = vtk.vtkUnstructuredGrid()
    mesh.SetPoints(vtk_points)
    mesh.SetPolyhedralCells(vtk_cell_types, vtk_cells, vtk_face_locations, vtk_faces)
    end = time.time()
    print(f"Time: {end - start:.6f} seconds...")
    
    # colors = vtkNamedColors()
    # # Create a mapper and actor
    # mapper = vtkDataSetMapper()
    # mapper.SetInputData(mesh)

    # actor = vtkActor()
    # actor.SetMapper(mapper)
    # actor.GetProperty().SetColor(
    #     colors.GetColor3d('Silver'))

    # # Visualize
    # renderer = vtkRenderer()
    # renderWindow = vtkRenderWindow()
    # renderWindow.SetWindowName('Polyhedron')
    # renderWindow.AddRenderer(renderer)
    # renderWindowInteractor = vtkRenderWindowInteractor()
    # renderWindowInteractor.SetRenderWindow(renderWindow)

    # renderer.AddActor(actor)
    # renderer.SetBackground(colors.GetColor3d('Salmon'))
    # renderer.ResetCamera()
    # renderer.GetActiveCamera().Azimuth(30)
    # renderer.GetActiveCamera().Elevation(30)
    # renderWindow.Render()
    # renderWindowInteractor.Start()


    return mesh


def compute_intersection_cell_locator(mesh:vtkUnstructuredGrid, line_start, line_end, tolerance):
    """
    Compute intersection points between an unstructured grid and a line using vtkCellLocator.
    
    Args:
        unstructured_grid: vtkUnstructuredGrid input
        line_start: Start point of the line [x, y, z]
        line_end: End point of the line [x, y, z]
    
    Returns:
        List of intersection points and corresponding cell IDs
    """

    # Variables to store intersection results
    tmp_points = vtk.vtkPoints()  # To store intersection points
    tmp_cells = vtk.vtkIdList()  # To store intersected cell IDs
    param_coords = []  # Parametric coordinates (t) along the line
    
    cell_locator = vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()  # Preprocess to build the octree
    
    # Compute intersections
    start = time.time()
    cell_locator.IntersectWithLine(line_start, line_end, tolerance, tmp_points, tmp_cells)
    end = time.time()
    print(f"Line intersection time: {end - start:.6f} seconds...")
    # Extract intersection points and corresponding cell IDs
    
    intersection_point_count = tmp_points.GetNumberOfPoints()
    intersection_cell_count = tmp_cells.GetNumberOfIds()

    intersection_points = []
    intersected_cells = []
    
    for i in range(0, intersection_point_count):
        intersection_points.append(tmp_points.GetPoint(i))
        
    for i in range(0, intersection_cell_count):
        intersected_cells.append(tmp_cells.GetId(i))
    
    return intersection_points, intersected_cells


def load_unstructured_grid_from_dat(file):
    # Check if the filename ends with .dat
    if file.lower().endswith('.dat'):
        start = time.time()
        print(f"Processing file: {file}")
        data = CAE_Decoder(3)
        data.Decode_dat_file(file)
        # Thus far, only the fluid zone is our interest
        fluid_zone:Zone_3D = data.Zones[0]
        hull_zone:Zone_3D = data.Zones[1]

        element_node_counts = defaultdict(set)
        element_node_dict = fluid_zone.EN
        vtk_cell_types_arr = np.zeros(fluid_zone.Element_count)
                
        mesh_fluid = construct_vtk_mesh(fluid_zone)
        mesh_hull = construct_vtk_mesh(hull_zone)

        # Attaching Variables
        variables = fluid_zone.Variables[3:] # Ignoring variables "X, Y, Z", which represents the node coordinates

        cell_ids = np.arange(0, fluid_zone.Element_count, dtype = np.int64)
        mesh_fluid = append_cell_array(mesh_fluid, cell_ids, "cell_ids")

        # write_unstructured_grid_to_vtk_file(mesh_fluid, 'SUBOFF_528k_FLUID')

        for v in range(0, len(variables)):
            v_name = variables[v]
            v_values = fluid_zone.Element_Variables[v]
            v_vtk_array = numpy_support.numpy_to_vtk(v_values, deep=True, array_type=vtk.VTK_DOUBLE)
            v_vtk_array.SetName(v_name)

            mesh_fluid.GetCellData().AddArray(v_vtk_array)

        # --- 创建 Velocity 在网格上 ---
        # 插值 celldata 到 pointdata，保留 celldata
        cell_to_point = vtk.vtkCellDataToPointData()
        cell_to_point.SetInputData(mesh_fluid)
        cell_to_point.Update()
        mesh_fluid = cell_to_point.GetOutput()

        # 避免插值后丢失celldata 再次添加一遍
        for v in range(0, len(variables)):
            v_name = variables[v]
            v_values = fluid_zone.Element_Variables[v]
            v_vtk_array = numpy_support.numpy_to_vtk(v_values, deep=True, array_type=vtk.VTK_DOUBLE)
            v_vtk_array.SetName(v_name)

            mesh_fluid.GetCellData().AddArray(v_vtk_array)

        u_array = mesh_fluid.GetPointData().GetArray("U")
        v_array = mesh_fluid.GetPointData().GetArray("V")
        w_array = mesh_fluid.GetPointData().GetArray("W")

        if u_array and v_array and w_array:
            velocity_array = vtk.vtkDoubleArray()
            velocity_array.SetName("Velocity")
            velocity_array.SetNumberOfComponents(3)
            velocity_array.SetNumberOfTuples(mesh_fluid.GetNumberOfPoints())
            
            for i in range(mesh_fluid.GetNumberOfPoints()):
                velocity_array.InsertNextTuple3(u_array.GetValue(i), v_array.GetValue(i), w_array.GetValue(i))
            
            mesh_fluid.GetPointData().AddArray(velocity_array)

        hull_P = hull_zone.Element_Variables[3]  # Assuming "P" is the 4th variable
        hull_cell_ids = np.arange(0, hull_zone.Element_count, dtype = np.int64)
        mesh_hull = append_cell_array(mesh_hull, hull_cell_ids, "cell_ids")
        mesh_hull = append_cell_array(mesh_hull, hull_P, "P")


        end = time.time()
        # 查看 cellData 包含的数组名称

        # 查看 cellData 包含的数组名称
        cell_data_hull = mesh_hull.GetCellData()
        cell_arrays_hull = [cell_data_hull.GetArrayName(i) for i in range(cell_data_hull.GetNumberOfArrays())]
        print("mesh_hull 的 cellData 数组:", cell_arrays_hull)

        cell_data_fluid = mesh_fluid.GetCellData()
        cell_arrays_fluid = [cell_data_fluid.GetArrayName(i) for i in range(cell_data_fluid.GetNumberOfArrays())]
        print("mesh_fluid 的 cellData 数组:", cell_arrays_fluid)

        point_data_fluid = mesh_fluid.GetPointData()
        point_arrays_fluid = [point_data_fluid.GetArrayName(i) for i in range(point_data_fluid.GetNumberOfArrays())]
        print("mesh_fluid 的 pointData 数组:", point_arrays_fluid)

        # 检查 Velocity 是否成功创建
        if mesh_fluid.GetPointData().HasArray("Velocity"):
            print("Velocity 已成功添加到 pointData 中")
        else:
            print("Velocity 未在 pointData 中找到")






        return mesh_fluid, mesh_hull
        
        # print(f"Number of points: {mesh_fluid.GetNumberOfPoints()}")
        # print(f"Number of cells: {mesh_hull.GetNumberOfCells()}")
        # print(f"Constructing unstructured grid from .dat time: {end - start:.6f} seconds...")
    else:
        print(f"Error, file {file} is not a .dat file.")
        return 

def write_unstructured_grid_to_vtk_file(mesh:vtkUnstructuredGrid, file: str):
    ''' 
    Write unstructured grid to .vtk file.
    '''
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(f"{file}.vtk")
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()  # Or `SetFileTypeToASCII()`
    writer.Write()
    
def load_unstructured_grid_from_vtk_file(file):
    # Check if the filename ends with .vtk
    if file.lower().endswith('.vtk'):
        # Create a reader for legacy VTK files
        reader = vtk.vtkUnstructuredGridReader()  # For unstructured grids
        # reader = vtk.vtkDataSetReader()        # For general VTK files (structured/unstructured)
        # Set the filename
        reader.SetFileName(file)
        # Read the file
        reader.Update()  # Triggers the reading process
        # Get the output data
        mesh = reader.GetOutput()  # For unstructured grids
        # Access key properties
        print(f"Number of points: {mesh.GetNumberOfPoints()}")
        print(f"Number of cells: {mesh.GetNumberOfCells()}")
        return mesh
    else:
        print(f"Error, file {file} is not a .vtk file.")
        return 

def workload_line_intersection_locator(locator: vtkCellLocator, line_start: list, line_end: list):
    # Compute intersections
    tolerance = 0.001
    points, cells = compute_intersection_cell_locator(locator, line_start, line_end, tolerance)

    # Print results
    print(f"Found {len(points)} intersection points:")
    # for i, (point, cell_id) in enumerate(zip(points, cells)):
    #     print(f"  Intersection {i+1}:")
    #     print(f"    Point: {point}")
    #     print(f"    Cell ID: {cell_id}")
        
    return points, cells
        
def workload_threshold_filtering(mesh: vtkUnstructuredGrid, variables: list, constraints: list):
    if len(variables) != len(constraints):
        print("Error, length of variables and constraints does not match.")
        return 
    
    # Get the cell data array by name
    cell_data = mesh.GetCellData()
    for i in range(0, len(variables)):
        variable_name = variables[i]  # Replace with your cell data array name
        values = cell_data.GetArray(variable_name)

        if values is None:
            raise ValueError(f"Cell data array '{variable_name}' not found.")

        # Convert VTK array to NumPy array
        value_array = np.array(values)

        # Define your constraints (example: values between 0.5 and 1.0)
        lower_bound = constraints[i][0]
        upper_bound = constraints[i][1]
        mask = (value_array >= lower_bound) & (value_array <= upper_bound)
        
        selected_indices = np.where(mask)[0]

    print(f"Find {len(selected_indices)} cells.")

    return selected_indices

def workload_locate_cells_by_points(mesh:vtkUnstructuredGrid, points: list):
    # point = [x, y, z]  # Your target coordinate (in the same coordinate system as the grid)
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()
    results = []

    for p in points:
        # Query parameters
        cell_id = -1
        # Find the cell
        cell_id = cell_locator.FindCell(p)
        results.append(cell_id)
        # Check if a cell was found
        if cell_id == -1:
            print(f"Point {p} is outside the grid or no cell found.")
            
    return results

def linear_interpolation_get_coordinate_by_variable(point_coordinates:list, point_variables:list, objective_variable:np.double):
    ''' Checking if linear interpolation can be done...'''
    # Check if the format of inputs are correct
    if len(point_coordinates) != 2 or point_variables != 2:
        print("\nERROR when calling \'interpolation_get_coordinate_by_variable()\': Input sizes does not match or larger than 2...")
        return 
    
    # Check if the desired value can be found between the two input points
    smaller_one = 0;
    larger_one = 1;
    if point_variables[smaller_one] > point_variables[larger_one]:
        smaller_one = 1
        larger_one = 0
        
    if objective_variable < smaller_one or objective_variable > larger_one:
        print("\nERROR when calling \'interpolation_get_coordinate_by_variable()\': Objective variable exceeds range...")
        return 

    ''' Start function after checking...'''
    # Initialize return value
    coordinate = np.zeros(3)
    
    objective_ratio_to_smaller_one = (objective_variable - smaller_one)/(larger_one - smaller_one)
    
    coordinate = point_coordinates[smaller_one] + (objective_ratio_to_smaller_one) * (point_coordinates[larger_one] - point_coordinates[smaller_one])

    return coordinate
        
def workload_plane_intersection_vtk(vtk_unstruct_mesh: vtkUnstructuredGrid, plane_origin: np.array, plane_norm: np.array):
    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_origin)  # Use your values
    plane.SetNormal(plane_norm)       # Use your values

    # Use vtkCutter to perform the intersection
    cutter = vtk.vtkCutter()
    cutter.SetInputData(vtk_unstruct_mesh)
    cutter.SetCutFunction(plane)
    # Optional: Generate triangles for better visualization
    cutter.GenerateTrianglesOn()
    cutter.Update()

    # Get the output from the cutter
    cut_output = cutter.GetOutput()
    print(f"Cutter produced {cut_output.GetNumberOfCells()} intersection lines.")
    

    # The key is to get the "vtkOriginalCellIds" array from the output's cell data
    original_ids = cut_output.GetCellData().GetArray("vtkOriginalCellIds")

    return original_ids

# def workload_point_intersection(pv_unstruct_mesh: pv.UnstructuredGrid, points: list):
#     indices = []
#     for p in points:
#         indices.append(pv_unstruct_mesh.find_containing_cell(p))
    
#     return indices

# def workload_line_intersection(pv_unstruct_mesh: pv.UnstructuredGrid, line_start: np.array, line_end: np.array):
    # 1. Create a cell locator from the mesh
    locator = pv.CellLocator(pv_unstruct_mesh)

    # 2. Find the cell IDs that intersect with the line
    # The tolerance is optional but can be useful for floating-point inaccuracies.
    cell_ids = locator.FindCellsAlongLine(line_start, line_end, tolerance=0.0)
    return cell_ids

def compute_face_normals_vtk_pipeline(grid, face_ids):
    """
    Computes normals by first creating a vtkPolyData surface and then
    using the vtkPolyDataNormals filter. This is the recommended approach.
    """
    # 1. Create the new surface mesh components
    new_points = vtk.vtkPoints()
    new_polys = vtk.vtkCellArray()
    
    point_map = {} # To avoid adding duplicate points

    # 2. Iterate through the faces to build the surface
    for cell_id, face_index in face_ids:
        cell = grid.GetCell(cell_id)
        face = cell.GetFace(face_index)
        if not face:
            continue

        # Get the point IDs for this face in the NEW polydata
        poly_point_ids = []
        for i in range(face.GetNumberOfPoints()):
            original_point = face.GetPoints().GetPoint(i)
            
            # Check if we've already added this point
            if original_point not in point_map:
                new_id = new_points.InsertNextPoint(original_point)
                point_map[original_point] = new_id
            poly_point_ids.append(point_map[original_point])
            
        # Add the polygon (face) to the cell array
        new_polys.InsertNextCell(len(poly_point_ids))
        for pid in poly_point_ids:
            new_polys.InsertCellPoint(pid)

    # 3. Create the PolyData object
    face_surface = vtk.vtkPolyData()
    face_surface.SetPoints(new_points)
    face_surface.SetPolys(new_polys)

    # 4. Use the vtkPolyDataNormals filter
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(face_surface)
    normals_filter.ComputeCellNormalsOn()
    normals_filter.ComputePointNormalsOff() # We only need cell normals
    normals_filter.Update()
    
    # 5. Get the results from the filter's output
    output = normals_filter.GetOutput()
    normals_array = output.GetCellData().GetNormals()
    
    # Convert to a list of lists for easier use in Python
    normals = []
    for i in range(normals_array.GetNumberOfTuples()):
        normal = [0.0, 0.0, 0.0]
        normals_array.GetTuple(i, normal)
        normals.append(normal)
        
    return normals


def append_cell_array(mesh:vtkUnstructuredGrid, arr:np.array, name:str):
    '''
    Description:
        Append new arrays to a unstructured grid, these arrays corresponds to the cells of the grid. 
    Input:
        mesh: The vtk unstructured grid;
        arr: The array to be appended;
        name: The name of the appended array.
    Output:
        None
    '''
    
    # Convert to VTK array
    vtk_array = numpy_support.numpy_to_vtk(arr)
    vtk_array.SetName(name)
    
    # Add to cell data
    mesh.GetCellData().AddArray(vtk_array)
    
    return mesh
    


if __name__ == "__main__":
    main()
