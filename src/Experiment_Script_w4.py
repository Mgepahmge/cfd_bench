import DB_Interface_PG as PG_API
import DB_Interface_IoTDB as IoTDB_API
import TileDB_Interface as TDB_API
import numpy as np
from numpy.typing import NDArray
import VTK_Interface as VTK_API
import os
import random
import time
import re

#cell_id: np.array of shape (1,1) coordinate: np.array of shape (1,3)
def random_start(vtk_mesh:VTK_API.vtkUnstructuredGrid):
    while True:
        bounds = vtk_mesh.GetBounds()
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[2], bounds[3])
        z = random.uniform(bounds[4], bounds[5])
        cell_id =VTK_API.VTK_Interface().vtk_point_intersection(vtk_mesh, np.array([[x, y, z]], dtype=np.float64))
        if cell_id.size == 1:
            break
    return cell_id, np.array([x, y, z], dtype=np.float64)

def cal_next_point(current_point:NDArray[np.float64], velocity:NDArray[np.float64], delta_t:float=0.01) -> NDArray[np.float64]:

    #Calculate the next point based on current point, velocity and time step
    return current_point + velocity * delta_t


def main():

    ''' Pre-processing '''
    # pg_entity = DB_API.PG_Interface.pg_connect()
    pg_api = PG_API.PG_Interface()
    pg_entity = pg_api.pg_connect()

    iotdb_api = IoTDB_API.Iotdb_Interface()
    iotdb_entity = iotdb_api.iotdb_connect()

    # tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()
    tdb_api = TDB_API.TileDB_Interface()

    vtk_api = VTK_API.VTK_Interface()

    ''' Define ship types and time_step for query'''
    SHIP_TYPE_LIST = ["JBC_615k", "JBC_3843k", "Kvlcc2_351k", "Kvlcc2_3709k", "Suboff_3258k"]
    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]
    VTK_MESH_DIR = "../vtk_dir/"
    TileDB_DIR = "../TileDB_Instances/"
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]

    ''' Workload 4 '''

    for ship_type in SHIP_TYPE_LIST:

        if ship_type in ["JBC_3843k", "Kvlcc2_3709k"]:
            valid_time_steps = [ts for ts in TIME_STEP_LIST if ts != "2000"]
            # print(f"{ship_type} 数据集没有 2000 时间步，使用: {valid_time_steps}")
        else:
            valid_time_steps = TIME_STEP_LIST.copy()

        vtk_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith(f"_200.vtk")]
        vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

        ''' W 4.1  Testing pg ... '''
        if "_" in ship_type:
            parts = ship_type.split("_")
            pg_api.set_ship_type(parts[0])  # 设置船型
            pg_api.set_ship_scale("_".join(parts[1:]))  # 设置规模
        pg_api.set_zone_type("fluid") 

        print("\nTesting PostgreSQL Workload 4 at ship type:", ship_type)
        pg_transaction = 0
        start_time = time.time()
        while time.time() - start_time < 60: 
            
            current_cell_indexe, current_coordinate = random_start(vtk_mesh)

            pg_trajectory = []

            pg_trajectory.append(current_coordinate)

            for time_step in valid_time_steps:
                pg_api.set_time_step(time_step)
                
                u, = pg_api.point_query(pg_entity, current_cell_indexe, "U")
                v, = pg_api.point_query(pg_entity, current_cell_indexe, "V")
                w, = pg_api.point_query(pg_entity, current_cell_indexe, "W")

                velocity = np.array([u, v, w], dtype=np.float64)

                next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=0.01)

                next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])
                if next_cell_indexe.size == 0:
                    break

                pg_trajectory.append(next_coordinate)
                current_cell_indexe = next_cell_indexe
                current_coordinate = next_coordinate

            pg_transaction += 1

        print("PostgreSQL Workload 4 Test at ship type", ship_type, "Completed.")
        print(f"Total PostgreSQL Transactions in 60 seconds: {pg_transaction}")


        ''' W 4.2  Testing iotdb ... '''
        iotdb_api.set_ship_type(ship_type)

        print("\nTesting IoTDB Workload 4 at ship type:", ship_type)
        iotdb_transaction = 0
        start_time = time.time()
        while time.time() - start_time < 60: 
            
            current_cell_indexe, current_coordinate = random_start(vtk_mesh)

            iotdb_trajectory = []

            iotdb_trajectory.append(current_coordinate)

            for time_step in valid_time_steps:
                iotdb_api.set_time_step(time_step)
                
                u, = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "U")
                v, = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "V")
                w, = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "W")

                velocity = np.array([u, v, w], dtype=np.float64)

                next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=0.01)

                next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])
                if next_cell_indexe.size == 0:
                    break

                iotdb_trajectory.append(next_coordinate)
                current_cell_indexe = next_cell_indexe
                current_coordinate = next_coordinate

            iotdb_transaction += 1

        print("IoTDB Workload 4 Test at ship type", ship_type, "Completed.")
        print(f"Total IoTDB Transactions in 60 seconds: {iotdb_transaction}")


        ''' W 4.3  Testing tiledb ... '''
        ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)
        print("\nTesting TileDB Workload 4 at ship type:", ship_type)
        tiledb_transaction = 0
        start_time = time.time()
        while time.time() - start_time < 60:
            
            current_cell_indexe, current_coordinate = random_start(vtk_mesh)

            tiledb_trajectory = []

            tiledb_trajectory.append(current_coordinate)

            for time_step in valid_time_steps:
                
                tiledb_file, = [
                    f for f in os.listdir(ship_tiledb_dir) 
                    if re.match(rf"^{time_step}(?!\d).*fluid\.tdb$", f)
                ]
                tdb_entity = tdb_api.Load_TileDB_File(os.path.join(ship_tiledb_dir, tiledb_file))
                
                u, = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, "U", current_cell_indexe)
                v, = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, "V", current_cell_indexe)
                w, = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, "W", current_cell_indexe)

                velocity = np.array([u, v, w], dtype=np.float64)

                next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=0.01)

                next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])
                if next_cell_indexe.size == 0:
                    break

                tiledb_trajectory.append(next_coordinate)
                current_cell_indexe = next_cell_indexe
                current_coordinate = next_coordinate

            tiledb_transaction += 1
        print("TileDB Workload 4 Test at ship type", ship_type, "Completed.")
        print(f"Total TileDB Transactions in 60 seconds: {tiledb_transaction}")

        ''' W 4.4  Testing vtk as db ... '''        
        print("\nTesting VTK Workload 4 at ship type:", ship_type)

        vtk_transaction = 0
        start_time = time.time()
        while time.time() - start_time < 60:

            current_cell_indexe, current_coordinate = random_start(vtk_mesh)

            vtk_trajectory = []

            for time_step in valid_time_steps:

                vtk_query_file, = [f for f in vtk_geo_files 
                    if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]

                vtk_entity = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_query_file))

                u, = vtk_api.point_query(vtk_entity, current_cell_indexe, "U")
                v, = vtk_api.point_query(vtk_entity, current_cell_indexe, "V")
                w, = vtk_api.point_query(vtk_entity, current_cell_indexe, "W")

                velocity = np.array([u, v, w], dtype=np.float64)
                next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=0.01)
                
                next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])
                if next_cell_indexe.size == 0:
                    break

                vtk_trajectory.append(next_coordinate)
                current_cell_indexe = next_cell_indexe
                current_coordinate = next_coordinate
            vtk_transaction += 1
        print("VTK Workload 4 Test at ship type", ship_type, "Completed.")
        print(f"Total VTK Transactions in 60 seconds: {vtk_transaction}")


if __name__ == "__main__":
    main()