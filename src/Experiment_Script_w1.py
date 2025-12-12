import DB_Interface_PG as PG_API
import DB_Interface_IoTDB as IoTDB_API
import numpy as np
from numpy.typing import NDArray
from vtk import vtkUnstructuredGrid
import VTK_Interface as VTK_API
import random
import os
import time

# Functions that represent various simple tasks that should be implemented based on the context

def random_points(vtk_mesh:vtkUnstructuredGrid)-> NDArray[np.float64]:
    bounds = vtk_mesh.GetBounds()
    num_points = random.randint(1, 100)
    points = np.random.rand(num_points, 3)
    points[:, 0] = points[:, 0] * (bounds[1] - bounds[0]) + bounds[0]
    points[:, 1] = points[:, 1] * (bounds[3] - bounds[2]) + bounds[2]
    points[:, 2] = points[:, 2] * (bounds[5] - bounds[4]) + bounds[4]
    return points

def random_line(vtk_mesh:vtkUnstructuredGrid):
    bounds = vtk_mesh.GetBounds()
    start = [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
    end = [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
    return np.array(start), np.array(end)

def random_plane(vtk_mesh:vtkUnstructuredGrid):
    bounds = vtk_mesh.GetBounds()
    origin = [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
    normal = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
    return np.array(origin), np.array(normal)

# In-memory computation for aggregated computation
def aggregation(vals: NDArray[np.float64]) -> np.float64:
    operations = ['sum', 'mean', 'max', 'min']
    op = random.choice(operations)

    if op == 'sum':
        result = np.sum(vals)
    elif op == 'mean':
        result = np.mean(vals)
    elif op == 'max':
        result = np.max(vals)
    elif op == 'min':
        result = np.min(vals)
    
    return result

def main():

    ''' Pre-processing '''

    pg_api = PG_API.PG_Interface()
    pg_entity = pg_api.pg_connect()

    iotdb_api = IoTDB_API.Iotdb_Interface()
    iotdb_entity = iotdb_api.iotdb_connect()

    # tiledb_entity = DB_API.Tiledb_Interface.tiledb_connect()

    vtk_api = VTK_API.VTK_Interface()
    
    ''' Define ship types and time_step for query'''
    SHIP_TYPE_LIST = ["JBC_615k", "JBC_3843k", "Kvlcc2_351k", "Kvlcc2_3709k", "Suboff_3258k"]
    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]
    VARIABLE_LIST = ["U", "V", "W", "P", "K", "E"]
    VTK_MESH_DIR = "../vtk_dir/"
    vtk_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    VTK_DB_DIR = "../vtk_db_dir/"
    
    for ship_type in SHIP_TYPE_LIST:

        # pg_api.set_ship_type(ship_type)
        if "_" in ship_type:
            parts = ship_type.split("_")
            pg_api.set_ship_type(parts[0])  # 设置船型
            pg_api.set_ship_scale("_".join(parts[1:]))  # 设置规模
        pg_api.set_zone_type("fluid") 

        # tiledb_api.set_ship_type(ship_type)

        iotdb_api.set_ship_type(ship_type)

        #load VTK mesh corresponding to current zone
        vtk_file, = [f for f in vtk_files if ship_type in f]
        vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))
        
        for time_step in TIME_STEP_LIST:

            # 跳过不存在的 2000 时间步
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                print(f"跳过 {ship_type} 的时间步 2000，该数据集无此时间步")
                continue  # 跳过当前时间步，继续下一个

            pg_api.set_time_step(time_step)
            
            iotdb_api.set_time_step(time_step)
 
            vtk_db_file_name = f"{ship_type}_GEO_{time_step}.vtk"
            vtk_db_file_path = os.path.join(VTK_DB_DIR, vtk_db_file_name)
            vtk_entity = vtk_api.vtk_connect(vtk_db_file_path)
 
            ''' Workload 1 '''

            ''' W 1.1 Point Intersection Query '''
            print("\nship_type:", ship_type, " time_step:", time_step)
            ''' W 1.1.1 Testing pg... '''
            print("\nTesting PG Point Intersection Query...")
            start_time = time.time()
            pg_point_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second

                attribute_name = random.choice(VARIABLE_LIST)

                while True:
                    coordinates = random_points(vtk_mesh) # TODO: Generate a list of valid coordinates as points

                    cell_indexes = VTK_API.VTK_Interface().vtk_point_intersection(vtk_mesh, coordinates)
                    if cell_indexes.size > 0:
                        break

                pg_point_vals = pg_api.point_query(pg_entity, cell_indexes, attribute_name)

                pg_point_result = aggregation(pg_point_vals)

                pg_point_transaction += 1

            print("PG Point Intersection Query Test Completed.")
            print(f"Total PG Point Transactions in 60 seconds: {pg_point_transaction}")

            ''' W 1.1.2 Testing iotdb ... '''
            print("\nTesting IoTDB Point Intersection Query...")
            start_time = time.time()
            iotdb_point_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second
                
                attribute_name = random.choice(VARIABLE_LIST)

                while True:
                    coordinates = random_points(vtk_mesh) # TODO: Generate a list of valid coordinates as points    

                    cell_indexes = VTK_API.VTK_Interface().vtk_point_intersection(vtk_mesh, coordinates)
                    if cell_indexes.size > 0:
                        break

                iotdb_point_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)

                iotdb_point_result = aggregation(iotdb_point_vals)

                iotdb_point_transaction += 1

            print("IoTDB Point Intersection Query Test Completed.")
            print(f"Total IoTDB Point Transactions in 60 seconds: {iotdb_point_transaction}")

            # ''' W 1.1.3 Testing tiledb ... '''

            # cell_indexes = DB_API.VTK_Interface.vtk_point_intersection(vtk_mesh, coordinates)

            # vals = DB_API.Tiledb_Interface.point_query(tiledb_entity, cell_indexes, attribute_names)

            # result = aggregation(vals)

            ''' W 1.1.4 Testing vtk (as db) ... '''
            print("\nTesting VTK Point Intersection Query...")
            start_time = time.time()
            vtk_point_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second
                
                attribute_name = random.choice(VARIABLE_LIST)

                while True:
                    coordinates = random_points(vtk_mesh) # TODO: Generate a list of valid coordinates as points

                    cell_indexes = VTK_API.VTK_Interface().vtk_point_intersection(vtk_mesh, coordinates)
                    if cell_indexes.size > 0:
                        break

                vtk_point_vals = vtk_api.point_query(vtk_entity, cell_indexes, attribute_name)

                vtk_point_result = aggregation(vtk_point_vals)

                vtk_point_transaction += 1

            print("VTK Point Intersection Query Test Completed.")
            print(f"Total VTK Point Transactions in 60 seconds: {vtk_point_transaction}")


            ''' W 1.2 Range Query ''' 

            ''' W 1.2.1 Testing pg... '''
            print("\nTesting PG Line Intersection Query...")
            start_time = time.time()
            pg_line_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second                

                attribute_name = random.choice(VARIABLE_LIST)
                
                while True:
                    line_start, line_end = random_line(vtk_mesh) # TODO: #Generate a random line

                    cell_indexes = VTK_API.VTK_Interface().vtk_line_intersection(vtk_mesh, line_start, line_end)
                    if cell_indexes.size > 0:
                        break

                pg_line_vals = pg_api.point_query(pg_entity, cell_indexes, attribute_name)

                pg_line_result = aggregation(pg_line_vals)

                pg_line_transaction += 1
            
            print("PG Line Intersection Query Test Completed.")
            print(f"Total PG Line Transactions in 60 seconds: {pg_line_transaction}")

            ''' W 1.2.2 Testing iotdb ... '''
            print("\nTesting IoTDB Line Intersection Query...")
            start_time = time.time()
            iotdb_line_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second
                
                attribute_name = random.choice(VARIABLE_LIST)

                while True:
                    line_start, line_end = random_line(vtk_mesh) # TODO: #Generate a random line

                    cell_indexes = VTK_API.VTK_Interface().vtk_line_intersection(vtk_mesh, line_start, line_end)
                    if cell_indexes.size > 0:
                        break

                iotdb_line_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)

                iotdb_line_result = aggregation(iotdb_line_vals)

                iotdb_line_transaction += 1

            print("IoTDB Line Intersection Query Test Completed.")
            print(f"Total IoTDB Line Transactions in 60 seconds: {iotdb_line_transaction}")

            ''' W 1.2.3 Testing tiledb ... '''

            # cell_indexes = DB_API.VTK_Interface.vtk_line_intersection(vtk_mesh, line_origin, line_direction)

            # vals = DB_API.Tiledb_Interface.point_query(tiledb_entity, cell_indexes, attribute_names)

            # result = aggregation(vals)

            ''' W 1.2.4 Testing vtk (as db) ... '''
            print("\nTesting VTK Line Intersection Query...")
            start_time = time.time()
            vtk_line_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second
                
                attribute_name = random.choice(VARIABLE_LIST)

                while True:
                    line_start, line_end = random_line(vtk_mesh) # TODO: #Generate a random line

                    cell_indexes = VTK_API.VTK_Interface().vtk_line_intersection(vtk_mesh, line_start, line_end)
                    if cell_indexes.size > 0:
                        break

                vtk_line_vals = vtk_api.point_query(vtk_entity, cell_indexes, attribute_name)

                vtk_line_result = aggregation(vtk_line_vals)

                vtk_line_transaction += 1

            print("VTK Line Intersection Query Test Completed.")
            print(f"Total VTK Line Transactions in 60 seconds: {vtk_line_transaction}")

            ''' W 1.3 Plane Query '''

            # plane_origin, plane_direction = place_holder() # TODO: Generate a random line 

            ''' W 1.3.1 Testing pg... '''
            print("\nTesting PG Plane Intersection Query...")
            start_time = time.time()
            pg_plane_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second                

                attribute_name = random.choice(VARIABLE_LIST)

                while True:
                    plane_origin, plane_direction = random_plane(vtk_mesh) # TODO: Generate a random plane 

                    cell_indexes = VTK_API.VTK_Interface().vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)
                    if cell_indexes.size > 0:
                        break

                pg_plane_vals = pg_api.point_query(pg_entity, cell_indexes, attribute_name)

                pg_plane_result = aggregation(pg_plane_vals)

                pg_plane_transaction += 1

            print("PG Plane Intersection Query Test Completed.")
            print(f"Total PG Plane Transactions in 60 seconds: {pg_plane_transaction}")

            # cell_indexes = DB_API.VTK_Interface.vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)

            # vals = DB_API.PG_Interface.point_query(pg_entity, cell_indexes, attribute_names)

            # result = aggregation(vals)

            ''' W 1.3.2 Testing iotdb ... '''
            print("\nTesting IoTDB Plane Intersection Query...")
            start_time = time.time()
            iotdb_plane_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second
                
                attribute_name = random.choice(VARIABLE_LIST)

                while True:
                    plane_origin, plane_direction = random_plane(vtk_mesh) # TODO: Generate a random plane 

                    cell_indexes = VTK_API.VTK_Interface().vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)
                    if cell_indexes.size > 0:
                        break

                iotdb_plane_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)

                iotdb_plane_result = aggregation(iotdb_plane_vals)

                iotdb_plane_transaction += 1
            
            print("IoTDB Plane Intersection Query Test Completed.")
            print(f"Total IoTDB Plane Transactions in 60 seconds: {iotdb_plane_transaction}")

            ''' W 1.3.3 Testing tiledb ... '''

            # cell_indexes = DB_API.VTK_Interface.vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)

            # vals = DB_API.Tiledb_Interface.point_query(tiledb_entity, cell_indexes, attribute_names)

            # result = aggregation(vals)

            ''' W 1.3.4 Testing vtk (as db) ... '''
            print("\nTesting VTK Plane Intersection Query...")
            start_time = time.time()
            vtk_plane_transaction = 0
            while time.time() - start_time < 60:  # Run for 60 second
                
                attribute_name = random.choice(VARIABLE_LIST)

                while True:
                    plane_origin, plane_direction = random_plane(vtk_mesh) # TODO: Generate a random plane 

                    cell_indexes = VTK_API.VTK_Interface().vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)
                    if cell_indexes.size > 0:
                        break

                vtk_plane_vals = vtk_api.point_query(vtk_entity, cell_indexes, attribute_name)

                vtk_plane_result = aggregation(vtk_plane_vals)

                vtk_plane_transaction += 1

            print("VTK Plane Intersection Query Test Completed.")
            print(f"Total VTK Plane Transactions in 60 seconds: {vtk_plane_transaction}")

if __name__ == "__main__":
    main()