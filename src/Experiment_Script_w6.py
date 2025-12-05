import Experiment_Framework_DBInterfaces as DB_API
import numpy as np
from numpy.typing import NDArray
from vtk import vtkCell, vtkCellLocator, vtkCellTypes, vtkPlane, vtkPolyData, vtkIdTypeArray, vtkUnstructuredGrid, vtkIdList

# A function that represents various simple tasks that should be implemented based on the context
def place_holder():

    return None


''' Pre-processing '''

# Notice, in w6, we are loading the hull_zone data instead of fluid_zone data

pg_entity = DB_API.PG_Interface.pg_connect()

iotdb_entity = DB_API.Iotdb_Interface.iotdb_connect()

tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()

vtk_entity = DB_API.VTK_Interface.vtk_connect() # Load timestep vtk files

vtk_mesh = DB_API.VTK_Interface.vtk_connect # Load the geo structure of the ship

''' Workload 6 '''

hull_zone_length = -1 # TODO: compute the number of elements of hull zone

all_indexes = np.arange(0, hull_zone_length)

''' W 6.1  Testing pg ... '''

norm_vectors = DB_API.VTK_Interface.vtk_surface_norm(vtk_mesh) # load the hull_zone and compute the norms of each element

pressures = DB_API.PG_Interface.point_query(pg_entity, all_indexes, ['P']) # The purpose is to retrieve all pressure values, one can choose to implement in the most efficient way.

aggregated_force = norm_vectors * pressures

# print(aggregated_force[0])

''' W 6.2  Testing iotdb ... '''

norm_vectors = DB_API.VTK_Interface.vtk_surface_norm(vtk_mesh) # load the hull_zone and compute the norms of each element

pressures = DB_API.Iotdb_Interface.point_query(iotdb_entity, all_indexes, ['P']) # The purpose is to retrieve all pressure values, one can choose to implement in the most efficient way.

aggregated_force = norm_vectors * pressures

# print(aggregated_force[0])

''' W 6.3  Testing tiledb ... '''

norm_vectors = DB_API.VTK_Interface.vtk_surface_norm(vtk_mesh) # load the hull_zone and compute the norms of each element

pressures = DB_API.Tiledb_Interface.point_query(tiledb_entity, all_indexes, ['P']) # The purpose is to retrieve all pressure values, one can choose to implement in the most efficient way.

aggregated_force = norm_vectors * pressures

# print(aggregated_force[0])

''' W 6.4  Testing vtk as db ... '''

norm_vectors = DB_API.VTK_Interface.vtk_surface_norm(vtk_mesh) # load the hull_zone and compute the norms of each element

pressures = DB_API.VTK_Interface.point_query(vtk_entity, all_indexes, ['P']) # The purpose is to retrieve all pressure values, one can choose to implement in the most efficient way.

aggregated_force = norm_vectors * pressures

# print(aggregated_force[0])