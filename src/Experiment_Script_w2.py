import DB_Interface_PG as PG_API
import DB_Interface_IoTDB as IoTDB_API
import TileDB_Interface as TDB_API
import numpy as np
from numpy.typing import NDArray
import VTK_Interface as VTK_API
import random
import os
from vtk import vtkUnstructuredGrid
import time
import re

# A function that represents various simple tasks that should be implemented based on the context
def random_range(vtk_mesh:vtkUnstructuredGrid) -> NDArray[np.float64]:
    bounds = vtk_mesh.GetBounds()
    x1, x2 = sorted([random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])])
    y1, y2 = sorted([random.uniform(bounds[2], bounds[3]), random.uniform(bounds[2], bounds[3])])
    z1, z2 = sorted([random.uniform(bounds[4], bounds[5]), random.uniform(bounds[4], bounds[5])])
    return np.array([[x1, y1, z1], [x2, y2, z2]])

# In-memory computation for aggregated computation
def aggregation(vals: NDArray[np.float64]) -> np.float64:
    
    mean_result = np.mean(vals)

    max_result = np.max(vals)

    min_result = np.min(vals)
    
    return mean_result, max_result, min_result

def main(ship_types=None):

    ''' Pre-processing '''

    pg_api = PG_API.PG_Interface()
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

    for ship_type in SHIP_TYPE_LIST:

        if ship_type in ["JBC_3843k", "Kvlcc2_3709k"]:
            valid_time_steps = [ts for ts in TIME_STEP_LIST if ts != "2000"]
            # print(f"{ship_type} 数据集没有 2000 时间步，使用: {valid_time_steps}")
        else:
            valid_time_steps = TIME_STEP_LIST.copy()

        # ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)

        vtk_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith(f"_200.vtk")]
        vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

        ''' Workload 2 '''
        # ranges: NDArray[np.float64] = place_holder() # TODO: Generateing a range in the form of [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        # COORD = ['X','Y','Z'] # Its just a macro

        # desired_attribute = place_holder() # TODO: Select a random attribute from [U,V,W,P,K,E], e.g., ['P']

        print(f"\nship_type: {ship_type}")

        # ''' W 2.1  Testing pg ... '''
        print("\nTesting PostgreSQL Workload 2...")
        if "_" in ship_type:
            parts = ship_type.split("_")
            pg_api.set_ship_type(parts[0])  # 设置船型
            pg_api.set_ship_scale("_".join(parts[1:]))  # 设置规模
        pg_api.set_zone_type("fluid") 

        pg_transaction = 0
        start = time.time()
        while time.time() - start < 60:
            attribute_name = random.choice(VARIABLE_LIST)

            # Generate a random range until we find some cells within the range
            while True:
                lower_bound, upper_bound = random_range(vtk_mesh)
                cell_indexes = pg_api.range_query_coord(pg_entity, lower_bound, upper_bound)
                if cell_indexes.size > 0:
                    break

            pg_result = []

            for time_step in valid_time_steps:

                pg_api.set_time_step(time_step)

                pg_vals = pg_api.point_query(pg_entity, cell_indexes, attribute_name)
                pg_result.extend(pg_vals)

            pg_aggvals = aggregation(pg_result)
            pg_transaction += 1
        print("PostgreSQL Workload 2 Test Completed.")
        print(f"Total PostgreSQL Transactions in 60 seconds: {pg_transaction}")

        ''' W 2.2  Testing iotdb ... '''
        print("\nTesting IoTDB Workload 2...")
        iotdb_api.set_ship_type(ship_type)

        iotdb_transaction = 0
        start = time.time()

        while time.time() - start < 60:
            attribute_name = random.choice(VARIABLE_LIST)
            # Generate a random range until we find some cells within the range
            while True:
                lower_bound, upper_bound = random_range(vtk_mesh)
                cell_indexes = iotdb_api.range_query_coord(iotdb_entity, lower_bound, upper_bound)
                if cell_indexes.size > 0:
                    break

            iotdb_result = []

            for time_step in valid_time_steps:

                iotdb_api.set_time_step(time_step)

                iotdb_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)
                iotdb_result.extend(iotdb_vals)

            iotdb_aggvals = aggregation(iotdb_result)

            iotdb_transaction += 1
        print("IoTDB Workload 2 Test Completed.")
        print(f"Total IoTDB Transactions in 60 seconds: {iotdb_transaction}")

        ''' W 2.3  Testing tiledb ... '''
        ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)
        print("\nTesting TileDB Workload 2...")
        tiledb_transaction = 0
        start = time.time()
        while time.time() - start < 60:
            attribute_name = random.choice(VARIABLE_LIST)

            # Generate a random range until we find some cells within the range
            while True:
                lower_bound, upper_bound = random_range(vtk_mesh)
                cell_indexes = tdb_api.Spatial_Range_Query_TileDB(tdb_entity, lower_bound, upper_bound)
                if cell_indexes.size > 0:
                    break

            tiledb_result = []

            for time_step in valid_time_steps:

                tiledb_file, = [
                    f for f in os.listdir(os.path.join(TileDB_DIR, ship_type)) 
                    if re.match(rf"^{time_step}(?!\d).*fluid\.tdb$", f)
                ]
                tdb_entity = tdb_api.Load_TileDB_File(os.path.join(TileDB_DIR, ship_type, tiledb_file))

                tiledb_vals = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, cell_indexes, attribute_name)

                tiledb_result.extend(tiledb_vals)

            tiledb_aggvals = aggregation(tiledb_result)
            tiledb_transaction += 1
        print("TileDB Workload 2 Test Completed.")
        print(f"Total TileDB Transactions in 60 seconds: {tiledb_transaction}")

        ''' W 2.4  Testing vtk as db ... '''
        print("\nTesting VTK Workload 2...")
        vtk_transaction = 0
        start = time.time()
        while time.time() - start < 60:
            attribute_name = random.choice(VARIABLE_LIST)

            while True:
                lower_bound, upper_bound = random_range(vtk_mesh)
                cell_indexes = vtk_api.range_query_coord(vtk_mesh, lower_bound, upper_bound)
                if cell_indexes.size > 0:
                    break
    
            vtk_result = []

            for time_step in valid_time_steps:

                vtk_query_file, = [f for f in vtk_geo_files 
                    if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]

                vtk_entity = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_query_file))

                vtk_vals = vtk_api.point_query(vtk_entity, cell_indexes, attribute_name)

                vtk_result.extend(vtk_vals)

            vtk_aggvals = aggregation(vtk_result)
            vtk_transaction += 1
        print("VTK Workload 2 Test Completed.")
        print(f"Total VTK Transactions in 60 seconds: {vtk_transaction}")

if __name__ == "__main__":
    main()
        

        