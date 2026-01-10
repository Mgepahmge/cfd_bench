import DB_Interface_PG as DB_API
import DB_Interface_IoTDB as IoTDB_API
import TileDB_Interface as TDB_API
import numpy as np
from numpy.typing import NDArray
import VTK_Interface as VTK_API
import os
import random
import time
import re

# A function that represents various simple tasks that should be implemented based on the context
def calculate_force(normals: NDArray[np.float64], pressures: NDArray[np.float64]) -> NDArray[np.float64]:
    total_force = np.array([0.0, 0.0, 0.0])
    for i in range(len(normals)):
        normal = normals[i]
        pressure = pressures[i]
        force = pressure * normal  # Assuming unit area for simplicity
        total_force += force
    return total_force

def main(ship_types=None):

    ''' Pre-processing '''

    # pg_entity = DB_API.PG_Interface.pg_connect()
    pg_api = DB_API.PG_Interface()
    pg_entity = pg_api.pg_connect()

    iotdb_api = IoTDB_API.Iotdb_Interface()
    iotdb_entity = iotdb_api.iotdb_connect()

    # tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()
    tdb_api = TDB_API.TileDB_Interface()

    vtk_api = VTK_API.VTK_Interface()


    ''' Define ship types and time_step for query'''
    if ship_types is None:
        SHIP_TYPE_LIST = ["JBC_615k", "JBC_3843k", "Kvlcc2_351k", "Kvlcc2_3709k", "Suboff_3258k"]
    else:
        SHIP_TYPE_LIST = ship_types
    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]

    VTK_QUERY_DIR = "../vtk_dir/"
    VTK_HULL_DIR = "../vtk_hull_dir/"
    TileDB_DIR = "../TileDB_Instances/"

    vtk_query_files = [f for f in os.listdir(VTK_QUERY_DIR) if f.endswith(".vtk")]
    vtk_hull_files = [f for f in os.listdir(VTK_HULL_DIR) if f.endswith(".vtk")]


    ''' Workload 6 '''
    for ship_type in SHIP_TYPE_LIST:

        for time_step in TIME_STEP_LIST:

            # 跳过不存在的 2000 时间步
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                # print(f"跳过 {ship_type} 的时间步 2000，该数据集无此时间步")
                continue  # 跳过当前时间步，继续下一个

            # vtk_hull_file, = [f for f in vtk_hull_files if ship_type in f]
            vtk_hull_file, = [f for f in vtk_hull_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
            vtk_hull_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_HULL_DIR, vtk_hull_file))

            ''' W 6.1  Testing pg ... '''
            if "_" in ship_type:
                parts = ship_type.split("_")
                pg_api.set_ship_type(parts[0])  # 设置船型
                pg_api.set_ship_scale("_".join(parts[1:]))  # 设置规模
            pg_api.set_zone_type("hull")

            pg_api.set_time_step(time_step)

            print("\nTesting PostgreSQL workload6 for ship type:", ship_type, " at time step:", time_step)

            pg_transaction = 0
            start_time = time.time()

            while time.time() - start_time < 60:
                
                pg_norm_vectors = VTK_API.VTK_Interface.vtk_surface_norm(vtk_hull_mesh) # load the hull_zone and compute the norms of each element

                pressures = pg_api.point_query(pg_entity, np.array(range(vtk_hull_mesh.GetNumberOfCells())), 'P') # The purpose is to retrieve all pressure values, one can choose to implement in the most efficient way.

                aggregated_force = calculate_force(pg_norm_vectors, pressures)

                pg_transaction += 1

            print("PostgreSQL workload6 transactions for ship type:", ship_type, " at time step:", time_step, " is ", pg_transaction)

            ''' W 6.2  Testing iotdb ... '''
            iotdb_api.set_ship_type(ship_type + "hull")
            iotdb_api.set_time_step(time_step)
            print("\nTesting IoTDB workload6 for ship type:", ship_type, " at time step:", time_step)
            iotdb_transcation = 0
            
            start_time = time.time()
            while time.time() - start_time < 60:

                iotdb_norm_vectors = VTK_API.VTK_Interface.vtk_surface_norm(vtk_hull_mesh) # load the hull_zone and compute the norms of each element

                pressures = iotdb_api.point_query(iotdb_entity, np.array(range(vtk_hull_mesh.GetNumberOfCells())), 'P') # The purpose is to retrieve all pressure values, one can choose to implement in the most efficient way.

                aggregated_force = calculate_force(iotdb_norm_vectors, pressures)

                iotdb_transcation += 1

            print("IoTDB workload6 transactions for ship type:", ship_type, " at time step:", time_step, " is ", iotdb_transcation)

            ''' W 6.3  Testing tiledb ... '''
            ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)
            tiledb_file, = [
                f for f in os.listdir(ship_tiledb_dir) 
                if re.match(rf"^{time_step}(?!\d).*hull\.tdb$", f)
            ]
            tdb_entity = tdb_api.Load_TileDB_File(os.path.join(ship_tiledb_dir, tiledb_file))
            print("\nTesting TileDB workload6 for ship type:", ship_type, " at time step:", time_step)
            tiledb_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:

                tiledb_norm_vectors = VTK_API.VTK_Interface.vtk_surface_norm(vtk_hull_mesh) # load the hull_zone and compute the norms of each element

                pressures = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, np.array(range(vtk_hull_mesh.GetNumberOfCells())), 'P') # The purpose is to retrieve all pressure values, one can choose to implement in the most efficient way.

                aggregated_force = calculate_force(tiledb_norm_vectors, pressures)

                tiledb_transaction += 1
            print("TileDB workload6 transactions for ship type:", ship_type, " at time step:", time_step, " is ", tiledb_transaction)

            ''' W 6.4  Testing vtk as db ... '''
            
            print("\nTesting VTK workload6 for ship type:", ship_type, " at time step:", time_step)

            vtk_transaction = 0
            start_time = time.time()

            while time.time() - start_time < 60:
                # vtk_entity = vtk_api.vtk_connect(os.path.join(VTK_HULL_DIR, vtk_hull_file))

                vtk_norm_vectors = VTK_API.VTK_Interface.vtk_surface_norm(vtk_hull_mesh) # load the hull_zone and compute the norms of each element

                pressures = vtk_api.point_query(vtk_hull_mesh, np.array(range(vtk_hull_mesh.GetNumberOfCells())), 'P') # The purpose is to retrieve all pressure values, one can choose to implement in the most efficient way.

                aggregated_force = calculate_force(vtk_norm_vectors, pressures)

                vtk_transaction += 1

            print("VTK workload6 transactions for ship type:", ship_type, " at time step:", time_step, " is ", vtk_transaction)
            
            # 需要每个ship每个timestep vtk_hull 5*10 files

if __name__ == "__main__":
    main()