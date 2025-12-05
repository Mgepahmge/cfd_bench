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

''' Workload 3 '''


desired_attribute = place_holder() # TODO: Select a random attribute from [U,V,W,P,K,E], e.g., ['P']

ranges: NDArray[np.float64] = place_holder() # TODO: Generateing a range in the form of [[v_min, v_max]], where v = desired_attribute

attribute_gap = place_holder() # TODO: read the maximum cell-wise gap value of the desired_attribute

''' W 3.1  Testing pg ... '''

cell_indexes = DB_API.PG_Interface.range_query(pg_entity, [[ranges[0][0] - attribute_gap, ranges[0][1] + attribute_gap]], desired_attribute)

sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(vtk_entity, cell_indexes)

DB_API.VTK_Interface.vtk_isosurface_extraction(sub_mesh)

''' W 3.2  Testing iotdb ... '''

cell_indexes = DB_API.Iotdb_Interface.range_query(iotdb_entity, [[ranges[0][0] - attribute_gap, ranges[0][1] + attribute_gap]], desired_attribute)

sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(vtk_entity, cell_indexes)

DB_API.VTK_Interface.vtk_isosurface_extraction(sub_mesh)

''' W 3.3  Testing tiledb ... '''

cell_indexes = DB_API.Tiledb_Interface.range_query(tiledb_entity, [[ranges[0][0] - attribute_gap, ranges[0][1] + attribute_gap]], desired_attribute)

sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(vtk_entity, cell_indexes)

DB_API.VTK_Interface.vtk_isosurface_extraction(sub_mesh)

''' W 3.4  Testing vtk as db ... '''

cell_indexes = DB_API.VTK_Interface.range_query(vtk_entity, [[ranges[0][0] - attribute_gap, ranges[0][1] + attribute_gap]], desired_attribute)

sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(vtk_entity, cell_indexes)

DB_API.VTK_Interface.vtk_isosurface_extraction(sub_mesh)