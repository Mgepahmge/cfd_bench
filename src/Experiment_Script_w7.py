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

''' Workload 7 '''

ranges: NDArray[np.float64] = place_holder() # TODO: Generateing a range in the form of [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

COORD = ['X','Y','Z'] # Its just a macro

''' W 7.1  Testing pg ... '''

cell_indexes = DB_API.PG_Interface.range_query(pg_entity, ranges, COORD)

sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_entity)

DB_API.VTK_Interface.vtk_Q_criterion(sub_mesh)

''' W 7.2  Testing iotdb ... '''

cell_indexes = DB_API.Iotdb_Interface.range_query(iotdb_entity, ranges, COORD)

sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_entity)

DB_API.VTK_Interface.vtk_Q_criterion(sub_mesh)

''' W 7.3  Testing tiledb ... '''

cell_indexes = DB_API.Tiledb_Interface.range_query(tiledb_entity, ranges, COORD)

sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_entity)

DB_API.VTK_Interface.vtk_Q_criterion(sub_mesh)

''' W 7.4  Testing vtk as db ... '''

cell_indexes = DB_API.VTK_Interface.range_query(vtk_entity, ranges, COORD)

sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_entity)

DB_API.VTK_Interface.vtk_Q_criterion(sub_mesh)