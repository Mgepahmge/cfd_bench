import DB_Interface_PG as PG_API
import DB_Interface_IoTDB as IoTDB_API
import numpy as np
from numpy.typing import NDArray
import VTK_Interface as VTK_API
import os
import random
import time

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

def cal_next_point(current_point:NDArray[np.float64], velocity:NDArray[np.float64], delta_t:float=1.0) -> NDArray[np.float64]:

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

    vtk_api = VTK_API.VTK_Interface()

    ''' Workload 3 '''

    ''' Define ship types and time_step for query'''
    SHIP_TYPE_LIST = ["JBC_615k", "JBC_3843k", "Kvlcc2_351k", "Kvlcc2_3709k", "Suboff_3258k"]
    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]
    VTK_MESH_DIR = "../vtk_dir/"
    VTK_QUERY_DIR = "../vtk_db_dir/"

    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    vtk_query_files = [f for f in os.listdir(VTK_QUERY_DIR) if f.endswith(".vtk")]

    ''' Workload 5 '''
    
    for ship_type in SHIP_TYPE_LIST:

        if ship_type in ["JBC_3843k", "Kvlcc2_3709k"]:
            valid_time_steps = [ts for ts in TIME_STEP_LIST if ts != "2000"]
            # print(f"{ship_type} 数据集没有 2000 时间步，使用: {valid_time_steps}")
        else:
            valid_time_steps = TIME_STEP_LIST.copy()

        if "_" in ship_type:
            parts = ship_type.split("_")
            pg_api.set_ship_type(parts[0])  # 设置船型
            pg_api.set_ship_scale("_".join(parts[1:]))  # 设置规模
        pg_api.set_zone_type("fluid") 

        iotdb_api.set_ship_type(ship_type)

        vtk_file, = [f for f in vtk_geo_files if ship_type in f]
        vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

        for time_step in valid_time_steps:

            ''' W 5.1  Testing pg ... '''
            pg_api.set_time_step(time_step)
            print("\nTesting PostgreSQL Workload 5 at ship type:", ship_type, "time step:", time_step)
            pg_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:

                pg_streamline = []

                current_cell_indexe, current_coordinate = random_start(vtk_mesh)

                pg_streamline.append(current_coordinate)

                while True:

                    u, = pg_api.point_query(pg_entity, current_cell_indexe, "U")
                    v, = pg_api.point_query(pg_entity, current_cell_indexe, "V")
                    w, = pg_api.point_query(pg_entity, current_cell_indexe, "W")

                    velocity = np.array([u, v, w], dtype=np.float64)

                    next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=1.0)

                    next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])

                    if next_cell_indexe.size == 0:
                        break

                    pg_streamline.append(next_coordinate)
                    current_cell_indexe = next_cell_indexe
                    current_coordinate = next_coordinate

                pg_transaction += 1

            print("PostgreSQL Workload 5 Test at ship type", ship_type, "time step", time_step, "Completed.")
            print(f"Total PostgreSQL Transactions in 60 seconds: {pg_transaction}")

            ''' W 5.2  Testing iotdb ... '''

            iotdb_api.set_time_step(time_step)
            print("\nTesting IoTDB Workload 5 at ship type:", ship_type, "time step:", time_step)
            iotdb_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:

                iotdb_streamline = []

                current_cell_indexe, current_coordinate = random_start(vtk_mesh)

                iotdb_streamline.append(current_coordinate)

                while True:

                    u, = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "U")
                    v, = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "V")
                    w, = iotdb_api.point_query(iotdb_entity, current_cell_indexe, "W")

                    velocity = np.array([u, v, w], dtype=np.float64)

                    next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=1.0)

                    next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])

                    if next_cell_indexe.size == 0:
                        break

                    iotdb_streamline.append(next_coordinate)
                    current_cell_indexe = next_cell_indexe
                    current_coordinate = next_coordinate

                iotdb_transaction += 1

            print("IoTDB Workload 5 Test at ship type", ship_type, "time step", time_step, "Completed.")
            print(f"Total IoTDB Transactions in 60 seconds: {iotdb_transaction}")

            ''' W 5.3  Testing tiledb ... '''

            # coordinates = DB_API.Tiledb_Interface.point_query(tiledb_entity, initial_cell_indexes, COORD)

            # trajectory = []

            # _proceed = True

            # while _proceed:

            #     tmp_indexes = DB_API.VTK_Interface.vtk_point_intersection(coordinates)

            #     if len(tmp_indexes == 0):
            #         # No interesection found
            #         _proceed = False

            #         break

            #     tmp_vel = DB_API.Tiledb_Interface.point_query(tiledb_entity, tmp_indexes, VELOCITY)
                
            #     coordinates = coordinates + tmp_vel

            #     trajectory.append(coordinates)

            # print(trajectory)

            ''' W 5.4  Testing vtk as db ... '''
            print("\nTesting VTK Workload 5 at ship type:", ship_type, "time step:", time_step)
            vtk_query_file, = [f for f in vtk_query_files 
            if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]

            vtk_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                vtk_streamline = []

                current_cell_indexe, current_coordinate = random_start(vtk_mesh)

                vtk_streamline.append(current_coordinate)

                while True:

                    vtk_entity = vtk_api.vtk_connect(os.path.join(VTK_QUERY_DIR, vtk_query_file))

                    u, = vtk_api.point_query(vtk_entity, current_cell_indexe, "U")
                    v, = vtk_api.point_query(vtk_entity, current_cell_indexe, "V")
                    w, = vtk_api.point_query(vtk_entity, current_cell_indexe, "W")
                    velocity = np.array([u, v, w], dtype=np.float64)
                    next_coordinate = cal_next_point(current_coordinate, velocity, delta_t=1.0)
                    next_cell_indexe = vtk_api.vtk_point_intersection(vtk_mesh, [next_coordinate])
        
                    if next_cell_indexe.size == 0:
                        break
                    vtk_streamline.append(next_coordinate)

                    current_cell_indexe = next_cell_indexe
                    current_coordinate = next_coordinate
                vtk_transaction += 1
            print("VTK Workload 5 Test at ship type", ship_type, "time step", time_step, "Completed.")
            print(f"Total VTK Transactions in 60 seconds: {vtk_transaction}")


if __name__ == "__main__":
    main()