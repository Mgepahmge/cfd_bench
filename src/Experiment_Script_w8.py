import Experiment_Framework_DBInterfaces as DB_API
import numpy as np
from numpy.typing import NDArray
from vtk import vtkCell, vtkCellLocator, vtkCellTypes, vtkPlane, vtkPolyData, vtkIdTypeArray, vtkUnstructuredGrid, vtkIdList


''' Pre-processing '''

pg_entity = DB_API.PG_Interface.pg_connect()

iotdb_entity = DB_API.Iotdb_Interface.iotdb_connect()

tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()

vtk_entity = DB_API.VTK_Interface.vtk_connect() # Load timestep vtk files

vtk_mesh = DB_API.VTK_Interface.vtk_connect # Load the geo structure of the ship

''' Workload 7 '''

finfo = np.finfo(np.float64)

''' W 8.1  Testing pg ... '''

# Theoretically, we should compute another variable, the Q-criterion and execute
#   range_query(pg_entity, [0, finfo.max], 'Q-criterion')
# However, we encountered problem computing the correct Q-criterion. 
# Therefore, from the perspective of time complexity, we replace Q-criterion with any other variable, e.g. U

DB_API.PG_Interface.range_query(pg_entity, [0, finfo.max], 'U') 

''' W 8.2 Testing iotdb ... '''

DB_API.Iotdb_Interface.range_query(iotdb_entity, [0, finfo.max], 'U') 

''' W 8.3 Testing tiledb ... '''

DB_API.Tiledb_Interface.range_query(tiledb_entity, [0, finfo.max], 'U') 

''' W 8.4 Testing vtk as db ... '''

DB_API.VTK_Interface.range_query(vtk_entity, [0, finfo.max], 'U') 