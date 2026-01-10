import DB_Interface_PG as DB_API
import DB_Interface_IoTDB as IoTDB_API
import TileDB_Interface as TDB_API
import numpy as np
from numpy.typing import NDArray
import csv
import VTK_Interface as VTK_API
import os
import random
import time
import re
from iotdb.utils.exception import StatementExecutionException

# A function that represents various simple tasks that should be implemented based on the context
def read_max_diffs_to_dict(delta_file_path):
    max_diffs = {}
    with open(delta_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) 
        for row in reader:
            if len(row) == 2:
                max_diffs[row[0]] = float(row[1])
    return max_diffs


def main(ship_types=None):

    ''' Pre-processing '''

    pg_api = DB_API.PG_Interface()
    pg_entity = pg_api.pg_connect()

    iotdb_api = IoTDB_API.Iotdb_Interface()
    iotdb_entity = iotdb_api.iotdb_connect()
    
    # tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()
    tdb_api = TDB_API.TileDB_Interface()

    vtk_api = VTK_API.VTK_Interface()

    ''' Workload 3 '''

    ''' Define ship types and time_step for query'''
    if ship_types is None:
        SHIP_TYPE_LIST = ["JBC_615k", "JBC_3843k", "Kvlcc2_351k", "Kvlcc2_3709k", "Suboff_3258k"]
    else:
        SHIP_TYPE_LIST = ship_types
    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]
    VARIABLE_LIST = ["U", "V", "W", "P", "K", "E"]
    VTK_MESH_DIR = "../vtk_dir/"
    MAX_RANGE_DIR = "../Max_Range/"
    TileDB_DIR = "../TileDB_Instances/"
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    delta_files = [f for f in os.listdir(MAX_RANGE_DIR) if f.endswith(".csv")]

    for ship_type in SHIP_TYPE_LIST:

        # pg_api.set_ship_type(ship_type)
        if "_" in ship_type:
            parts = ship_type.split("_")
            pg_api.set_ship_type(parts[0])  # 设置船型
            pg_api.set_ship_scale("_".join(parts[1:]))  # 设置规模
        pg_api.set_zone_type("fluid") 
        
        iotdb_api.set_ship_type(ship_type)

        ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)

        for time_step in TIME_STEP_LIST:

            # 跳过不存在的 2000 时间步
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                print(f"跳过 {ship_type} 的时间步 2000，该数据集无此时间步")
                continue  # 跳过当前时间步，继续下一个

            pg_api.set_time_step(time_step)

            iotdb_api.set_time_step(time_step)

            tiledb_file, = [
                    f for f in os.listdir(os.path.join(TileDB_DIR, ship_type)) 
                    if re.match(rf"^{time_step}(?!\d).*fluid\.tdb$", f)
                ]
            tdb_entity = tdb_api.Load_TileDB_File(os.path.join(TileDB_DIR, ship_type, tiledb_file))
            
            vtk_query_file, = [f for f in vtk_geo_files 
                if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
            vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_query_file))

            delta_file, = [f for f in delta_files if ship_type in f
                           and f.endswith(f"_{time_step}_max_diffs.csv")]
            
            max_diffs_dict = read_max_diffs_to_dict(os.path.join(MAX_RANGE_DIR, delta_file))

            
            ''' W 3.1  Testing pg ... '''
            print("\nTesting PostgreSQL Workload 3 at ship type:", ship_type, "time step:", time_step)

            pg_transaction = 0
            start_time = time.time()

            while time.time() - start_time < 60:  # Run for 60 seconds

                attribute_name = random.choice(VARIABLE_LIST)

                delta = max_diffs_dict[attribute_name]

                pg_iso_value, = pg_api.point_query(pg_entity,[random.randint(1, vtk_mesh.GetNumberOfCells())], attribute_name)
                
                cell_indexes = pg_api.range_query_var(pg_entity, pg_iso_value - delta, pg_iso_value + delta, attribute_name)

                pg_sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh) 

                pg_isosurface = VTK_API.VTK_Interface.vtk_isosurface_extraction(pg_sub_mesh, attribute_name, pg_iso_value)

                pg_transaction += 1

            print(f"PostgreSQL Workload 3 completed {pg_transaction} transactions in 60 seconds.")
            print(f"Total PostgreSQL Transactions in 60 seconds: {pg_transaction}")


            ''' W 3.2  Testing iotdb ... '''
            print("\nTesting IoTDB Workload 3 at ship type:", ship_type, "time step:", time_step)

            iotdb_transaction = 0

            start_time = time.time()
            while time.time() - start_time < 60:  # Run for 60 seconds

                try:

                    attribute_name = random.choice(VARIABLE_LIST)

                    delta = max_diffs_dict[attribute_name]

                    iotdb_iso_value, = iotdb_api.point_query(iotdb_entity,[random.randint(0, vtk_mesh.GetNumberOfCells()-1)], attribute_name)
                    
                    cell_indexes = iotdb_api.range_query_var(iotdb_entity, iotdb_iso_value -delta, iotdb_iso_value + delta, attribute_name)

                    iotdb_sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh) 

                    iotdb_isosurface = VTK_API.VTK_Interface.vtk_isosurface_extraction(iotdb_sub_mesh, attribute_name, iotdb_iso_value)  

                except StatementExecutionException as e:
                    print(f"IoTDB Query 超时，跳过当前事务: {e}")
                    continue  # Skip this iteration and continue with the next
                else:
                    iotdb_transaction += 1

            print(f"IoTDB Workload 3 completed {iotdb_transaction} transactions in 60 seconds.")
            print(f"Total IoTDB Transactions in 60 seconds: {iotdb_transaction}")

            ''' W 3.3  Testing tiledb ... '''
            print("\nTesting TileDB Workload 3 at ship type:", ship_type, "time step:", time_step)
            tiledb_transaction = 0
            start_time = time.time()
            while time.time() - start_time < 60:  # Run for 60 seconds
                
                attribute_name = random.choice(VARIABLE_LIST)

                delta = max_diffs_dict[attribute_name]

                tiledb_iso_value, = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, attribute_name, [random.randint(0, vtk_mesh.GetNumberOfCells()-1)])
                
                cell_indexes = tdb_api.Attribute_Range_Query_TileDB(tdb_entity, attribute_name, tiledb_iso_value - delta, tiledb_iso_value + delta)

                tiledb_sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh) 

                tiledb_isosurface = VTK_API.VTK_Interface.vtk_isosurface_extraction(tiledb_sub_mesh, attribute_name, tiledb_iso_value)  

                tiledb_transaction += 1
            print(f"TileDB Workload 3 completed {tiledb_transaction} transactions in 60 seconds.")
            print(f"Total TileDB Transactions in 60 seconds: {tiledb_transaction}")

            ''' W 3.4  Testing vtk as db ... '''
            print("\nTesting VTK Workload 3 at ship type:", ship_type, "time step:", time_step)
            vtk_transaction = 0
            start_time = time.time()

            while time.time() - start_time < 60:  # Run for 60 second

                attribute_name = random.choice(VARIABLE_LIST)

                delta = max_diffs_dict[attribute_name]

                vtk_entity = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_query_file))

                vtk_iso_value, = vtk_api.point_query(vtk_entity,[random.randint(0, vtk_mesh.GetNumberOfCells()-1)], attribute_name)

                cell_indexes = vtk_api.range_query_var(vtk_entity, vtk_iso_value - delta, vtk_iso_value + delta, attribute_name)

                vtk_sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)

                vtk_isosurface = VTK_API.VTK_Interface.vtk_isosurface_extraction(vtk_sub_mesh, attribute_name, vtk_iso_value)

                vtk_transaction += 1

            print("VTK Workload 3 Completed.")
            print(f"Total VTK Transactions in 60 seconds: {vtk_transaction}")


if __name__ == "__main__":
    main()