## import DB_Interface_PG as DB_API  # IoTDB-only mode
import DB_Interface_IoTDB as IoTDB_API
## import TileDB_Interface as TDB_API  # IoTDB-only mode
import numpy as np
from numpy.typing import NDArray
import csv
import VTK_Interface as VTK_API
import vtk
import os
import random
import time
import re
import psutil # 用于监测资源
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
        
        # 累加指定数据库进程的消耗
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

def read_max_diffs_to_dict(delta_file_path):
    max_diffs = {}
    with open(delta_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) 
        for row in reader:
            if len(row) == 2:
                max_diffs[row[0]] = float(row[1])
    return max_diffs

def main():
    vtk.vtkObject.GlobalWarningDisplayOff()
    ''' Pre-processing '''
    iotdb_api = IoTDB_API.Iotdb_Interface(os.getenv("SHIP_TO_RUN", "JBC_615k"), os.getenv("STEP_TO_RUN", "200"))
    iotdb_entity = iotdb_api.iotdb_connect()
    vtk_api = VTK_API.VTK_Interface()

    # 配合自动化脚本，优先从环境变量读取
    env_ship = os.getenv("SHIP_TO_RUN")
    env_step = os.getenv("STEP_TO_RUN")

    SHIP_TYPE_LIST = [env_ship] if env_ship else ["JBC_615k"]
    TIME_STEP_LIST = [env_step] if env_step else ["200"]

    VARIABLE_LIST = ["U", "V", "W", "P", "K", "E"]
    VTK_MESH_DIR = "../../vtk_dir/"
    MAX_RANGE_DIR = "../../Max_Range/"
    TileDB_DIR = "../../TileDB_Instances/"
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    delta_files = [f for f in os.listdir(MAX_RANGE_DIR) if f.endswith(".csv")]

    for ship_type in SHIP_TYPE_LIST:
        iotdb_api.set_ship_type(ship_type)

        for time_step in TIME_STEP_LIST:
            # 跳过不存在的时间步
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                continue

            iotdb_api.set_time_step(time_step)
            
            vtk_query_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
            vtk_mesh = VTK_API.VTK_Interface().vtk_connect(os.path.join(VTK_MESH_DIR, vtk_query_file))

            delta_file, = [f for f in delta_files if ship_type in f and f.endswith(f"_{time_step}_max_diffs.csv")]
            max_diffs_dict = read_max_diffs_to_dict(os.path.join(MAX_RANGE_DIR, delta_file))

            # IoTDB-only mode: run IoTDB section and skip PostgreSQL/TileDB/VTK.
            skip_current_dataset = False
            with UnifiedResourceMonitor(f"IoTDB_{ship_type}_{time_step}", target_process_names=["iotdb", "java"]):
                iotdb_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    try:
                        attribute_name = random.choice(VARIABLE_LIST)
                        delta = max_diffs_dict[attribute_name]
                        point_vals = iotdb_api.point_query(iotdb_entity, [random.randint(0, vtk_mesh.GetNumberOfCells() - 1)], attribute_name)
                        if point_vals is None or len(point_vals) == 0:
                            print(f"SKIP_DATASET_EMPTY: Ship={ship_type} Step={time_step} Stage=IsoPoint")
                            skip_current_dataset = True
                            break
                        iotdb_iso_value = point_vals[0]
                        cell_indexes = iotdb_api.range_query_var(iotdb_entity, iotdb_iso_value - delta, iotdb_iso_value + delta, attribute_name)
                        if cell_indexes is None or len(cell_indexes) == 0:
                            print(f"SKIP_DATASET_EMPTY: Ship={ship_type} Step={time_step} Stage=RangeQuery")
                            skip_current_dataset = True
                            break
                        iotdb_sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)
                        iotdb_isosurface = VTK_API.VTK_Interface.vtk_isosurface_extraction(iotdb_sub_mesh, attribute_name, iotdb_iso_value)
                    except StatementExecutionException:
                        print(f"SKIP_DATASET_EXCEPTION: Ship={ship_type} Step={time_step} Stage=IoTDB")
                        skip_current_dataset = True
                        break
                    else:
                        iotdb_transaction += 1
                # print(f"Total IoTDB Transactions ({ship_type}_{time_step}): {iotdb_transaction}")
            if skip_current_dataset:
                continue
            continue

            ''' W 3.3 TileDB '''
            with UnifiedResourceMonitor(f"TileDB_{ship_type}_{time_step}"):
                tiledb_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    delta = max_diffs_dict[attribute_name]
                    tiledb_iso_value, = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, attribute_name, [random.randint(0, vtk_mesh.GetNumberOfCells()-1)])
                    cell_indexes = tdb_api.Attribute_Range_Query_TileDB(tdb_entity, attribute_name, tiledb_iso_value - delta, tiledb_iso_value + delta)
                    tiledb_sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh) 
                    tiledb_isosurface = VTK_API.VTK_Interface.vtk_isosurface_extraction(tiledb_sub_mesh, attribute_name, tiledb_iso_value)  
                    tiledb_transaction += 1
                # print(f"Total TileDB Transactions ({ship_type}_{time_step}): {tiledb_transaction}")

            ''' W 3.4 VTK '''
            with UnifiedResourceMonitor(f"VTK_{ship_type}_{time_step}"):
                vtk_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    attribute_name = random.choice(VARIABLE_LIST)
                    delta = max_diffs_dict[attribute_name]
                    vtk_entity = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_query_file))
                    vtk_iso_value, = vtk_api.point_query(vtk_entity,[random.randint(0, vtk_mesh.GetNumberOfCells()-1)], attribute_name)
                    cell_indexes = vtk_api.range_query_var(vtk_entity, vtk_iso_value - delta, vtk_iso_value + delta, attribute_name)
                    vtk_sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)
                    vtk_isosurface = VTK_API.VTK_Interface.vtk_isosurface_extraction(vtk_sub_mesh, attribute_name, vtk_iso_value)
                    vtk_transaction += 1
                # print(f"Total VTK Transactions ({ship_type}_{time_step}): {vtk_transaction}")

if __name__ == "__main__":
    main()