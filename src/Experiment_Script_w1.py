import Experiment_Framework_DBInterfaces as DB_API
import numpy as np
from numpy.typing import NDArray
from vtk import vtkCell, vtkCellLocator, vtkCellTypes, vtkPlane, vtkPolyData, vtkIdTypeArray, vtkUnstructuredGrid, vtkIdList

# A function that represents various simple tasks that should be implemented based on the context
def place_holder():
    
    return None

# In-memory computation for aggregated computation
def aggregation(vals: NDArray[np.float64]) -> np.float64:
    # TODO
    return result



''' Pre-processing '''

pg_entity = DB_API.PG_Interface.pg_connect()

iotdb_entity = DB_API.Iotdb_Interface.iotdb_connect()

tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()

vtk_entity = DB_API.VTK_Interface.vtk_connect() # Load timestep vtk files

vtk_mesh = DB_API.VTK_Interface.vtk_connect # Load the geo structure of the ship



''' Workload 1 '''

attribute_names = place_holder() # TODO: Randomly select one or several attributes from [U,V,W,P,K,E]

''' W 1.1 Point Query '''

coordinates = place_holder() # TODO: Generate a list of valid coordinates as points

''' W 1.1.1 Testing pg... '''
cell_indexes = DB_API.VTK_Interface.vtk_point_intersection(vtk_mesh, coordinates)

vals = DB_API.PG_Interface.point_query(pg_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.1.2 Testing iotdb ... '''

cell_indexes = DB_API.VTK_Interface.vtk_point_intersection(vtk_mesh, coordinates)

vals = DB_API.Iotdb_Interface.point_query(iotdb_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.1.3 Testing tiledb ... '''

cell_indexes = DB_API.VTK_Interface.vtk_point_intersection(vtk_mesh, coordinates)

vals = DB_API.Tiledb_Interface.point_query(tiledb_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.1.4 Testing vtk (as db) ... '''

cell_indexes = DB_API.VTK_Interface.vtk_point_intersection(vtk_mesh, coordinates)

vals = DB_API.VTK_Interface.point_query(vtk_entity, cell_indexes, attribute_names)

result = aggregation(vals)




''' W 1.2 Range Query '''

line_origin, line_direction = place_holder() # TODO: Generate a random line 

''' W 1.2.1 Testing pg... '''
cell_indexes = DB_API.VTK_Interface.vtk_line_intersection(vtk_mesh, line_origin, line_direction)

vals = DB_API.PG_Interface.point_query(pg_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.2.2 Testing iotdb ... '''

cell_indexes = DB_API.VTK_Interface.vtk_line_intersection(vtk_mesh, line_origin, line_direction)

vals = DB_API.Iotdb_Interface.point_query(iotdb_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.2.3 Testing tiledb ... '''

cell_indexes = DB_API.VTK_Interface.vtk_line_intersection(vtk_mesh, line_origin, line_direction)

vals = DB_API.Tiledb_Interface.point_query(tiledb_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.2.4 Testing vtk (as db) ... '''

cell_indexes = DB_API.VTK_Interface.vtk_line_intersection(vtk_mesh, line_origin, line_direction)

vals = DB_API.VTK_Interface.point_query(vtk_entity, cell_indexes, attribute_names)

result = aggregation(vals)




''' W 1.3 Plane Query '''

plane_origin, plane_direction = place_holder() # TODO: Generate a random line 

''' W 1.3.1 Testing pg... '''
cell_indexes = DB_API.VTK_Interface.vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)

vals = DB_API.PG_Interface.point_query(pg_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.3.2 Testing iotdb ... '''

cell_indexes = DB_API.VTK_Interface.vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)

vals = DB_API.Iotdb_Interface.point_query(iotdb_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.3.3 Testing tiledb ... '''

cell_indexes = DB_API.VTK_Interface.vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)

vals = DB_API.Tiledb_Interface.point_query(tiledb_entity, cell_indexes, attribute_names)

result = aggregation(vals)

''' W 1.3.4 Testing vtk (as db) ... '''

cell_indexes = DB_API.VTK_Interface.vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)

vals = DB_API.VTK_Interface.point_query(vtk_entity, cell_indexes, attribute_names)

result = aggregation(vals)