import DB_Interface_PG as DB_API
import DB_Interface_IoTDB as IoTDB_API
import TileDB_Interface as TDB_API
import numpy as np
from numpy.typing import NDArray
import vtk
from vtk import vtkUnstructuredGrid ,vtkGradientFilter, vtkDataObject
import VTK_Interface as VTK_API
import os
import random
import time
import re
from iotdb.utils.exception import StatementExecutionException

def random_var_range(vtk_mesh: vtk.vtkUnstructuredGrid, attribute_name: str) -> NDArray[np.float64]:
    """
    从 vtk_mesh 中自动获取变量 attribute_name 的取值范围，
    随机生成一个合法的 [lower_bound, upper_bound]
    """

    data_array = vtk_mesh.GetCellData().GetArray(attribute_name)

    if data_array is None:
        raise ValueError(f"Attribute '{attribute_name}' not found in vtk mesh.")

    vmin, vmax = data_array.GetRange()

    lower, upper = sorted([random.uniform(vmin, vmax), random.uniform(vmin, vmax)])

    return np.array([lower, upper], dtype=np.float64)


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
    VARIABLE_LIST = ["U", "V", "W", "P", "K", "E"]
    VTK_MESH_DIR = "../vtk_dir/"   
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    TileDB_DIR = "../TileDB_Instances/"

    ''' Workload 8 '''
    for ship_type in SHIP_TYPE_LIST:

        for time_step in TIME_STEP_LIST:

            # 跳过不存在的 2000 时间步
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                # print(f"跳过 {ship_type} 的时间步 2000，该数据集无此时间步")
                continue  # 跳过当前时间步，继续下一个

            vtk_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
            vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

            ''' W 8.1  Testing pg ... '''
            if "_" in ship_type:
                parts = ship_type.split("_")
                pg_api.set_ship_type(parts[0])  # 设置船型
                pg_api.set_ship_scale("_".join(parts[1:]))  # 设置规模
            pg_api.set_zone_type("fluid")

            pg_api.set_time_step(time_step)

            print("\nTesting PostgreSQL workload8 for ship type:", ship_type, " at time step:", time_step)
            pg_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                
                attribute_name = random.choice(VARIABLE_LIST)

                lower_bound, upper_bound = random_var_range(vtk_mesh, attribute_name)

                cell_indexes = pg_api.range_query_var(pg_entity, lower_bound, upper_bound, attribute_name)

                pg_transaction += 1

            print("PostgreSQL workload8 transactions for ship type:", ship_type, " at time step:", time_step, " is ", pg_transaction)

            ''' W 8.2  Testing iotdb ... '''

            iotdb_api.set_ship_type(ship_type)
            iotdb_api.set_time_step(time_step)
            print("\nTesting IoTDB workload8 for ship type:", ship_type, " at time step:", time_step)
            iotdb_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                try:
                    attribute_name = random.choice(VARIABLE_LIST)

                    lower_bound, upper_bound = random_var_range(vtk_mesh, attribute_name)

                    cell_indexes = iotdb_api.range_query_var(iotdb_entity, lower_bound, upper_bound, attribute_name)
                
                except StatementExecutionException as e:
                    print("IoTDB StatementExecutionException:", e)
                    continue  # Skip this iteration and try again
                else:
                    iotdb_transaction += 1

            print("IoTDB workload8 transactions for ship type:", ship_type, " at time step:", time_step, " is ", iotdb_transaction)


            ''' W 8.3  Testing tiledb ... '''
            ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)
            tiledb_file, = [
                f for f in os.listdir(ship_tiledb_dir) 
                if re.match(rf"^{time_step}(?!\d).*fluid\.tdb$", f)
            ]
            tdb_entity = tdb_api.Load_TileDB_File(os.path.join(ship_tiledb_dir, tiledb_file))

            print("\nTesting TileDB workload8 for ship type:", ship_type, " at time step:", time_step)
            tiledb_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                attribute_name = random.choice(VARIABLE_LIST)

                lower_bound, upper_bound = random_var_range(vtk_mesh, attribute_name)

                cell_indexes = tdb_api.Attribute_Range_Query_TileDB(tdb_entity, attribute_name, lower_bound, upper_bound)

                tiledb_transaction += 1
            print("TileDB workload8 transactions for ship type:", ship_type, " at time step:", time_step, " is ", tiledb_transaction)

            ''' W 8.4  Testing vtk as db ... '''
            print("\nTesting VTK workload8 for ship type:", ship_type, " at time step:", time_step)
            vtk_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:
                attribute_name = random.choice(VARIABLE_LIST)

                lower_bound, upper_bound = random_var_range(vtk_mesh, attribute_name)

                cell_indexes = vtk_api.range_query_var(vtk_mesh, lower_bound, upper_bound, attribute_name)

                vtk_transaction += 1

            print("VTK workload8 transactions for ship type:", ship_type, " at time step:", time_step, " is ", vtk_transaction)

if __name__ == "__main__":
    main()