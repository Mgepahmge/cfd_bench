import DB_Interface_PG as DB_API
import DB_Interface_IoTDB as IoTDB_API
import numpy as np
from numpy.typing import NDArray
from vtk import vtkUnstructuredGrid ,vtkGradientFilter, vtkDataObject
import VTK_Interface as VTK_API
import os
import random
import time

def random_range(vtk_mesh:vtkUnstructuredGrid) -> NDArray[np.float64]:
    bounds = vtk_mesh.GetBounds()
    x1, x2 = sorted([random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])])
    y1, y2 = sorted([random.uniform(bounds[2], bounds[3]), random.uniform(bounds[2], bounds[3])])
    z1, z2 = sorted([random.uniform(bounds[4], bounds[5]), random.uniform(bounds[4], bounds[5])])
    return np.array([[x1, y1, z1], [x2, y2, z2]])

def ComputeQCriterion(vtk_mesh:vtkUnstructuredGrid) -> NDArray[np.float64]:

    #for this function, we assume that the velocity vector is already constructed in the vtk_mesh

    velocity_array = vtk_mesh.GetPointData().GetArray("Velocity")
    if velocity_array:
        gradient_filter = vtkGradientFilter()
        gradient_filter.SetInputData(vtk_mesh)
        gradient_filter.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "Velocity")
        gradient_filter.SetComputeQCriterion(True)
        gradient_filter.Update()
        mesh_with_q = gradient_filter.GetOutput()
        QCriterion_array = mesh_with_q.GetPointData().GetArray("QCriterion")
        return QCriterion_array
        
def main():

    ''' Pre-processing '''

    # pg_entity = DB_API.PG_Interface.pg_connect()
    pg_api = DB_API.PG_Interface()
    pg_entity = pg_api.pg_connect()

    iotdb_api = IoTDB_API.Iotdb_Interface()
    iotdb_entity = iotdb_api.iotdb_connect()

    # tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()

    vtk_api = VTK_API.VTK_Interface()


    ''' Define ship types and time_step for query'''
    SHIP_TYPE_LIST = ["JBC_615k", "JBC_3843k", "Kvlcc2_351k", "Kvlcc2_3709k", "Suboff_3258k"]
    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]
    VTK_MESH_DIR = "../vtk_dir/"
    # VTK_QUERY_DIR = "../vtk_db_dir/"
   
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    # vtk_query_files = [f for f in os.listdir(VTK_QUERY_DIR) if f.endswith(".vtk")]
    


    ''' Workload 7 '''
    for ship_type in SHIP_TYPE_LIST:

        # vtk_file, = [f for f in vtk_geo_files if ship_type in f]
        # vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

        for time_step in TIME_STEP_LIST:

            # 跳过不存在的 2000 时间步
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                # print(f"跳过 {ship_type} 的时间步 2000，该数据集无此时间步")
                continue  # 跳过当前时间步，继续下一个

            vtk_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
            vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

            ''' W 7.1  Testing pg ... '''
            if "_" in ship_type:
                parts = ship_type.split("_")
                pg_api.set_ship_type(parts[0])  # 设置船型
                pg_api.set_ship_scale("_".join(parts[1:]))  # 设置规模
            pg_api.set_zone_type("fluid")

            pg_api.set_time_step(time_step)

            print("\nTesting PostgreSQL workload7 for ship type:", ship_type, " at time step:", time_step)
            pg_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                while True:
                    lower_bound, upper_bound = random_range(vtk_mesh)
                    cell_indexes = pg_api.range_query_coord(pg_entity, lower_bound, upper_bound)
                    if cell_indexes.size > 0:
                        break
                
                sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)

                pg_QCriterion = ComputeQCriterion(sub_mesh)

                pg_transaction += 1

            print("PostgreSQL workload7 transactions for ship type:", ship_type, " at time step:", time_step, " is ", pg_transaction)

            ''' W 7.2  Testing iotdb ... '''

            iotdb_api.set_ship_type(ship_type)
            iotdb_api.set_time_step(time_step)
            print("\nTesting IoTDB workload7 for ship type:", ship_type, " at time step:", time_step)
            iotdb_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                while True:
                    lower_bound, upper_bound = random_range(vtk_mesh)
                    cell_indexes = iotdb_api.range_query_coord(iotdb_entity, lower_bound, upper_bound)
                    if cell_indexes.size > 0:
                        break
                
                sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)

                iotdb_QCriterion = ComputeQCriterion(sub_mesh)

                iotdb_transaction += 1

            print("IoTDB workload7 transactions for ship type:", ship_type, " at time step:", time_step, " is ", iotdb_transaction)


            ''' W 7.3  Testing tiledb ... '''

            # cell_indexes = DB_API.Tiledb_Interface.range_query(tiledb_entity, ranges, COORD)

            # sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_entity)

            # DB_API.VTK_Interface.vtk_Q_criterion(sub_mesh)

            ''' W 7.4  Testing vtk as db ... '''
            print("\nTesting VTK workload7 for ship type:", ship_type, " at time step:", time_step)
            vtk_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                while True:
                    lower_bound, upper_bound = random_range(vtk_mesh)
                    cell_indexes = vtk_api.range_query_coord(vtk_mesh, lower_bound, upper_bound)
                    if cell_indexes.size > 0:
                        break
                
                sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)

                vtk_QCriterion = ComputeQCriterion(sub_mesh)

                vtk_transaction += 1

            print("VTK workload7 transactions for ship type:", ship_type, " at time step:", time_step, " is ", vtk_transaction)

            # cell_indexes = DB_API.VTK_Interface.range_query(vtk_entity, ranges, COORD)

            # sub_mesh = DB_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_entity)

            # DB_API.VTK_Interface.vtk_Q_criterion(sub_mesh)


if __name__ == "__main__":
    main()