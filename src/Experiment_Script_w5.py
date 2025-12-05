import Experiment_Framework_DBInterfaces as DB_API
import numpy as np
from numpy.typing import NDArray
from vtk import vtkCell, vtkCellLocator, vtkCellTypes, vtkPlane, vtkPolyData, vtkIdTypeArray, vtkUnstructuredGrid, vtkIdList

# A function that represents various simple tasks that should be implemented based on the context
def place_holder():

    return None


''' Pre-processing '''

pg_entity = DB_API.PG_Interface.pg_connect()

iotdb_entity = DB_API.Iotdb_Interface.iotdb_connect()

tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()

vtk_entity = DB_API.VTK_Interface.vtk_connect() # Load timestep vtk files

vtk_mesh = DB_API.VTK_Interface.vtk_connect # Load the geo structure of the ship

''' Workload 5 '''

COORD = ['X','Y','Z'] # Its just a macro

VELOCITY = ['U','V','W'] # Its another macro

initial_cell_indexes = place_holder() # TODO: Generate one or several indexes as starting cells

''' W 5.1  Testing pg ... '''

coordinates = DB_API.PG_Interface.point_query(pg_entity, initial_cell_indexes, COORD)

trajectory = []

_proceed = True

while _proceed:

    tmp_indexes = DB_API.VTK_Interface.vtk_point_intersection(coordinates)

    if len(tmp_indexes == 0):
        # No interesection found
        _proceed = False

        break

    tmp_vel = DB_API.PG_Interface.point_query(pg_entity, tmp_indexes, VELOCITY)
    
    coordinates = coordinates + tmp_vel

    trajectory.append(coordinates)

# print(trajectory)

''' W 5.2  Testing iotdb ... '''

coordinates = DB_API.Iotdb_Interface.point_query(iotdb_entity, initial_cell_indexes, COORD)

trajectory = []

_proceed = True

while _proceed:

    tmp_indexes = DB_API.VTK_Interface.vtk_point_intersection(coordinates)

    if len(tmp_indexes == 0):
        # No interesection found
        _proceed = False

        break

    tmp_vel = DB_API.Iotdb_Interface.point_query(iotdb_entity, tmp_indexes, VELOCITY)
    
    coordinates = coordinates + tmp_vel

    trajectory.append(coordinates)

# print(trajectory)

''' W 5.3  Testing tiledb ... '''

coordinates = DB_API.Tiledb_Interface.point_query(tiledb_entity, initial_cell_indexes, COORD)

trajectory = []

_proceed = True

while _proceed:

    tmp_indexes = DB_API.VTK_Interface.vtk_point_intersection(coordinates)

    if len(tmp_indexes == 0):
        # No interesection found
        _proceed = False

        break

    tmp_vel = DB_API.Tiledb_Interface.point_query(tiledb_entity, tmp_indexes, VELOCITY)
    
    coordinates = coordinates + tmp_vel

    trajectory.append(coordinates)

# print(trajectory)

''' W 5.4  Testing vtk as db ... '''

coordinates = DB_API.VTK_Interface.point_query(vtk_entity, initial_cell_indexes, COORD)

trajectory = []

_proceed = True

while _proceed:

    tmp_indexes = DB_API.VTK_Interface.vtk_point_intersection(coordinates)

    if len(tmp_indexes == 0):
        # No interesection found
        _proceed = False

        break

    tmp_vel = DB_API.VTK_Interface.point_query(vtk_entity, tmp_indexes, VELOCITY)
    
    coordinates = coordinates + tmp_vel

    trajectory.append(coordinates)

# print(trajectory)