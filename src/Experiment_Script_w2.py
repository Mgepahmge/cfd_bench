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

time_steps = [200,400,...,1800]

''' Workload 2 '''

ranges: NDArray[np.float64] = place_holder() # TODO: Generateing a range in the form of [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

COORD = ['X','Y','Z'] # Its just a macro

desired_attribute = place_holder() # TODO: Select a random attribute from [U,V,W,P,K,E], e.g., ['P']

''' W 2.1  Testing pg ... '''

cell_indexes = DB_API.PG_Interface.range_query(pg_entity, ranges, COORD)

result = []

for t in time_steps:

    pg_entity = DB_API.PG_Interface.pg_connect() # TODO: We are about to read a new timestep, adjust FROM clause accordingly.

    result.append(DB_API.PG_Interface.point_query(pg_entity, cell_indexes, desired_attribute))

aggregation(result)

''' W 2.2  Testing iotdb ... '''

cell_indexes = DB_API.Iotdb_Interface.range_query(iotdb_entity, ranges, COORD)

result = []

for t in time_steps:

    iotdb_entity = DB_API.PG_Interface.iotdb_connect() # TODO: We are about to read a new timestep, adjust FROM clause accordingly.

    result.append(DB_API.Iotdb_Interface.point_query(iotdb_entity, cell_indexes, desired_attribute))

aggregation(result)

''' W 2.3  Testing tiledb ... '''

cell_indexes = DB_API.Tiledb_Interface.range_query(tiledb_entity, ranges, COORD)

result = []

for t in time_steps:

    tiledb_entity = DB_API.PG_Interface.tiledb_connect() # TODO: We are about to read a new timestep, adjust FROM clause accordingly.

    result.append(DB_API.Tiledb_Interface.point_query(tiledb_entity, cell_indexes, desired_attribute))

aggregation(result)

''' W 2.4  Testing vtk as db ... '''

cell_indexes = DB_API.VTK_Interface.range_query(vtk_entity, ranges, COORD)

result = []

for t in time_steps:

    vtk_entity = DB_API.PG_Interface.vtk_connect() # We are about to read a new timestep, adjust FROM clause accordingly.

    result.append(DB_API.VTK_Interface.point_query(vtk_entity, cell_indexes, desired_attribute))

aggregation(result)