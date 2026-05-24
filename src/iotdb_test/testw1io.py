## import DB_Interface_PG as PG_API  # IoTDB-only mode
from cfd_bench.infra.iotdb import legacy_variables_client as IoTDB_API
## import TileDB_Interface as TDB_API  # IoTDB-only mode
import numpy as np
from numpy.typing import NDArray
from vtk import vtkUnstructuredGrid
import vtk
import VTK_Interface as VTK_API
import random
import os
import time
import re
import psutil  # 用于监测资源
from iotdb.utils.exception import StatementExecutionException

# --- 综合资源监测类 ---
class UnifiedResourceMonitor:
    def __init__(self, label, target_process_names=None):
        self.label = label
        self.target_names = target_process_names
        self.main_process = psutil.Process(os.getpid())
        
    def _get_total_stats(self):
        # 统计当前 Python 进程
        total_mem = self.main_process.memory_info().rss
        total_read = self.main_process.io_counters().read_bytes
        
        # 累加指定数据库服务进程的消耗
        if self.target_names:
            for proc in psutil.process_iter(['name', 'pid']):
                try:
                    if any(name in proc.info['name'].lower() for name in self.target_names):
                        p = psutil.Process(proc.info['pid'])
                        total_mem += p.memory_info().rss
                        total_read += p.io_counters().read_bytes
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return total_mem / 1024 / 1024, total_read / 1024 / 1024

    def __enter__(self):
        self.start_time = time.time()
        self.start_mem, self.start_read = self._get_total_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_mem, end_read = self._get_total_stats()
        print(f"\n>>>> [综合性能报告 - {self.label}] <<<<")
        print(f"  耗时: {end_time - self.start_time:.2f} s")
        print(f"  内存变化: {end_mem - self.start_mem:+.2f} MB (总占用: {end_mem:.2f} MB)")
        print(f"  磁盘读取: {max(0, end_read - self.start_read):.2f} MB")
        print("-" * 55)

# --- 辅助函数 ---
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

def has_empty_values(vals: NDArray[np.float64]) -> bool:
    return vals is None or len(vals) == 0

def main():
    vtk.vtkObject.GlobalWarningDisplayOff()
    ''' Pre-processing '''
    iotdb_api = IoTDB_API.Iotdb_Interface(os.getenv("SHIP_TO_RUN", "JBC_615k"), os.getenv("STEP_TO_RUN", "200"))
    iotdb_entity = iotdb_api.iotdb_connect()
    vtk_api = VTK_API.VTK_Interface()
    
    # 优先从环境变量读取，适配自动化采集脚本
    env_ship = os.getenv("SHIP_TO_RUN")
    env_step = os.getenv("STEP_TO_RUN")
    SHIP_TYPE_LIST = [env_ship] if env_ship else ["JBC_615k"]
    TIME_STEP_LIST = [env_step] if env_step else ["200"]

    VARIABLE_LIST = ["U", "V", "W", "P", "K", "E"]
    VTK_MESH_DIR = "../../vtk_dir/"
    vtk_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    TileDB_DIR = "../../TileDB_Instances/"
    
    for ship_type in SHIP_TYPE_LIST:
        iotdb_api.set_ship_type(ship_type)

        for time_step in TIME_STEP_LIST:
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                continue

            iotdb_api.set_time_step(time_step)
            vtk_file, = [f for f in vtk_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
            vtk_mesh = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))
            vtk_entity = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

            # IoTDB-only mode: run IoTDB point/line/plane sections and skip PG/TileDB/VTK.

            with UnifiedResourceMonitor(f"IoTDB_Point_{ship_type}_{time_step}", ["iotdb", "java"]):
                iotdb_point_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        coordinates = random_points(vtk_mesh)
                        cell_indexes = vtk_api.vtk_point_intersection(vtk_mesh, coordinates)
                        if cell_indexes.size > 0:
                            break
                    iotdb_point_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)
                    if has_empty_values(iotdb_point_vals):
                        print(f"SKIP_STAGE_EMPTY: Ship={ship_type} Step={time_step} Stage=Point")
                        break
                    aggregation(iotdb_point_vals)
                    iotdb_point_transaction += 1
                # print(f"Total IoTDB Point Transactions ({ship_type}_{time_step}): {iotdb_point_transaction}")

            with UnifiedResourceMonitor(f"IoTDB_Line_{ship_type}_{time_step}", ["iotdb", "java"]):
                iotdb_line_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        line_start, line_end = random_line(vtk_mesh)
                        cell_indexes = vtk_api.vtk_line_intersection(vtk_mesh, line_start, line_end)
                        if cell_indexes.size > 0:
                            break
                    iotdb_line_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)
                    if has_empty_values(iotdb_line_vals):
                        print(f"SKIP_STAGE_EMPTY: Ship={ship_type} Step={time_step} Stage=Line")
                        break
                    aggregation(iotdb_line_vals)
                    iotdb_line_transaction += 1
                # print(f"Total IoTDB Line Transactions ({ship_type}_{time_step}): {iotdb_line_transaction}")

            with UnifiedResourceMonitor(f"IoTDB_Plane_{ship_type}_{time_step}", ["iotdb", "java"]):
                iotdb_plane_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    try:
                        attribute_name = random.choice(VARIABLE_LIST)
                        while True:
                            plane_origin, plane_direction = random_plane(vtk_mesh)
                            cell_indexes = vtk_api.vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)
                            if cell_indexes.size > 0:
                                break
                        iotdb_plane_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)
                        if has_empty_values(iotdb_plane_vals):
                            print(f"SKIP_STAGE_EMPTY: Ship={ship_type} Step={time_step} Stage=Plane")
                            break
                        aggregation(iotdb_plane_vals)
                    except StatementExecutionException:
                        continue
                    else:
                        iotdb_plane_transaction += 1
                # print(f"Total IoTDB Plane Transactions ({ship_type}_{time_step}): {iotdb_plane_transaction}")
            continue

            # 1.1.3 TileDB
            with UnifiedResourceMonitor(f"TileDB_Point_{ship_type}_{time_step}"):
                tiledb_point_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        coordinates = random_points(vtk_mesh)
                        cell_indexes = VTK_API.VTK_Interface().vtk_point_intersection(vtk_mesh, coordinates)
                        if cell_indexes.size > 0: break
                    tiledb_point_vals = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, attribute_name, cell_indexes)
                    aggregation(tiledb_point_vals)
                    tiledb_point_transaction += 1
                # print(f"Total TileDB Point Transactions ({ship_type}_{time_step}): {tiledb_point_transaction}")

            # 1.1.4 VTK
            with UnifiedResourceMonitor(f"VTK_Point_{ship_type}_{time_step}"):
                vtk_point_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        coordinates = random_points(vtk_mesh)
                        cell_indexes = VTK_API.VTK_Interface().vtk_point_intersection(vtk_mesh, coordinates)
                        if cell_indexes.size > 0: break
                    vtk_point_vals = vtk_api.point_query(vtk_entity, cell_indexes, attribute_name)
                    aggregation(vtk_point_vals)
                    vtk_point_transaction += 1
                # print(f"Total VTK Point Transactions ({ship_type}_{time_step}): {vtk_point_transaction}")

            # --- W 1.2 Line Intersection ---
            # 1.2.1 PG
            with UnifiedResourceMonitor(f"PG_Line_{ship_type}_{time_step}", ["postgres"]):
                pg_line_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        line_start, line_end = random_line(vtk_mesh)
                        cell_indexes = VTK_API.VTK_Interface().vtk_line_intersection(vtk_mesh, line_start, line_end)
                        if cell_indexes.size > 0: break
                    pg_line_vals = pg_api.point_query(pg_entity, cell_indexes, attribute_name)
                    aggregation(pg_line_vals)
                    pg_line_transaction += 1
                # print(f"Total PG Line Transactions ({ship_type}_{time_step}): {pg_line_transaction}")

            # # 1.2.2 IoTDB
            # with UnifiedResourceMonitor(f"IoTDB_Line_{ship_type}_{time_step}", ["iotdb", "java"]):
            #     iotdb_line_transaction = 0
            #     start_time = time.time()
            #     while time.time() - start_time < 60:
            #         attribute_name = random.choice(VARIABLE_LIST)
            #         while True:
            #             line_start, line_end = random_line(vtk_mesh)
            #             cell_indexes = vtk_api.vtk_line_intersection(vtk_mesh, line_start, line_end)
            #             if cell_indexes.size > 0: break
            #         iotdb_line_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)
            #         aggregation(iotdb_line_vals)
            #         iotdb_line_transaction += 1
            #     print(f"Total IoTDB Line Transactions ({ship_type}_{time_step}): {iotdb_line_transaction}")

            # 1.2.3 TileDB
            with UnifiedResourceMonitor(f"TileDB_Line_{ship_type}_{time_step}"):
                tiledb_line_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        line_start, line_end = random_line(vtk_mesh)
                        cell_indexes = VTK_API.VTK_Interface().vtk_line_intersection(vtk_mesh, line_start, line_end)
                        if cell_indexes.size > 0: break
                    tiledb_line_vals = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, attribute_name, cell_indexes)
                    aggregation(tiledb_line_vals)
                    tiledb_line_transaction += 1
                # print(f"Total TileDB Line Transactions ({ship_type}_{time_step}): {tiledb_line_transaction}")

            # 1.2.4 VTK
            with UnifiedResourceMonitor(f"VTK_Line_{ship_type}_{time_step}"):
                vtk_line_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        line_start, line_end = random_line(vtk_mesh)
                        cell_indexes = VTK_API.VTK_Interface().vtk_line_intersection(vtk_mesh, line_start, line_end)
                        if cell_indexes.size > 0: break
                    vtk_line_vals = vtk_api.point_query(vtk_entity, cell_indexes, attribute_name)
                    aggregation(vtk_line_vals)
                    vtk_line_transaction += 1
                # print(f"Total VTK Line Transactions ({ship_type}_{time_step}): {vtk_line_transaction}")

            # --- W 1.3 Plane Intersection ---
            # 1.3.1 PG
            with UnifiedResourceMonitor(f"PG_Plane_{ship_type}_{time_step}", ["postgres"]):
                pg_plane_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        plane_origin, plane_direction = random_plane(vtk_mesh)
                        cell_indexes = VTK_API.VTK_Interface().vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)
                        if cell_indexes.size > 0: break
                    pg_plane_vals = pg_api.point_query(pg_entity, cell_indexes, attribute_name)
                    aggregation(pg_plane_vals)
                    pg_plane_transaction += 1
                # print(f"Total PG Plane Transactions ({ship_type}_{time_step}): {pg_plane_transaction}")

            # # 1.3.2 IoTDB
            # with UnifiedResourceMonitor(f"IoTDB_Plane_{ship_type}_{time_step}", ["iotdb", "java"]):
            #     iotdb_plane_transaction = 0
            #     start_time = time.time()
            #     while time.time() - start_time < 60:
            #         try:
            #             attribute_name = random.choice(VARIABLE_LIST)
            #             while True:
            #                 plane_origin, plane_direction = random_plane(vtk_mesh)
            #                 cell_indexes = vtk_api.vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)
            #                 if cell_indexes.size > 0: break
            #             iotdb_plane_vals = iotdb_api.point_query(iotdb_entity, cell_indexes, attribute_name)
            #             aggregation(iotdb_plane_vals)
            #         except StatementExecutionException:
            #             continue
            #         else:
            #             iotdb_plane_transaction += 1
            #     # print(f"Total IoTDB Plane Transactions ({ship_type}_{time_step}): {iotdb_plane_transaction}")

            # 1.3.3 TileDB
            with UnifiedResourceMonitor(f"TileDB_Plane_{ship_type}_{time_step}"):
                tiledb_plane_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        plane_origin, plane_direction = random_plane(vtk_mesh)
                        cell_indexes = VTK_API.VTK_Interface().vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)
                        if cell_indexes.size > 0: break
                    tiledb_plane_vals = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, attribute_name, cell_indexes)
                    aggregation(tiledb_plane_vals)
                    tiledb_plane_transaction += 1
                # print(f"Total TileDB Plane Transactions ({ship_type}_{time_step}): {tiledb_plane_transaction}")

            # 1.3.4 VTK
            with UnifiedResourceMonitor(f"VTK_Plane_{ship_type}_{time_step}"):
                vtk_plane_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    while True:
                        plane_origin, plane_direction = random_plane(vtk_mesh)
                        cell_indexes = VTK_API.VTK_Interface().vtk_plane_intersection(vtk_mesh, plane_origin, plane_direction)
                        if cell_indexes.size > 0: break
                    vtk_point_vals = vtk_api.point_query(vtk_entity, cell_indexes, attribute_name)
                    aggregation(vtk_point_vals)
                    vtk_plane_transaction += 1
                # print(f"Total VTK Plane Transactions ({ship_type}_{time_step}): {vtk_plane_transaction}")

if __name__ == "__main__":
    main()