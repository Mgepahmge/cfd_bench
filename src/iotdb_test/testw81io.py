## import DB_Interface_PG as DB_API  # IoTDB-only mode
from cfd_bench.infra.iotdb import legacy_variables_client as IoTDB_API
## import TileDB_Interface as TDB_API  # IoTDB-only mode
import numpy as np
from numpy.typing import NDArray
import vtk
from vtk import vtkUnstructuredGrid ,vtkGradientFilter, vtkDataObject
import VTK_Interface as VTK_API
import os
import random
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
        # 1. 统计当前 Python 进程
        total_mem = self.main_process.memory_info().rss
        total_read = self.main_process.io_counters().read_bytes
        
        # 2. 累加指定数据库进程的消耗
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

def read_data_to_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.strip().split('\n')
    data_list = []
    for line in lines:
        nums = [num.strip() for num in line.split(',') if num.strip()]
        data_list.append([np.float64(num) for num in nums])
    return data_list

def main():
    vtk.vtkObject.GlobalWarningDisplayOff()
    ''' Pre-processing '''
    iotdb_api = IoTDB_API.Iotdb_Interface(os.getenv("SHIP_TO_RUN", "JBC_615k"), os.getenv("STEP_TO_RUN", "200"))
    iotdb_entity = iotdb_api.iotdb_connect()
    vtk_api = VTK_API.VTK_Interface()

    # 配合自动化脚本，优先从环境变量读取船型
    env_ship = os.getenv("SHIP_TO_RUN")
    SHIP_TYPE_LIST = [env_ship] if env_ship else ["JBC_615k", "JBC_3843k", "Kvlcc2_351k", "Kvlcc2_3709k", "Suboff_3258k"]
    
    TIME_STEP_LIST = ["200", "400", "600", "800", "1000", "1200", "1400", "1600", "1800", "2000"]
    VTK_MESH_DIR = "../../vtk_dir/"   
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    TileDB_DIR = "../../TileDB_Instances/"
    RANGE_DIR = "../../output/"

    for ship_type in SHIP_TYPE_LIST:
        target_list = read_data_to_list(os.path.join(RANGE_DIR, f"{ship_type}.txt"))

        if ship_type in ["JBC_3843k", "Kvlcc2_3709k"]:
            valid_time_steps = [ts for ts in TIME_STEP_LIST if ts != "2000"]
        else:
            valid_time_steps = TIME_STEP_LIST.copy()

        vtk_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith("_200.vtk")]
        vtk_mesh = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

        # IoTDB-only mode: run IoTDB section and skip PostgreSQL/TileDB/VTK.
        with UnifiedResourceMonitor(f"IoTDB_{ship_type}", target_process_names=["iotdb", "java"]):
            iotdb_api.set_ship_type(ship_type)

            iotdb_transaction = 0
            skip_current_dataset = False
            start_time = time.time()
            while time.time() - start_time < 60:
                attribute_name = "V"
                lower_bound, upper_bound = random.choice(target_list)

                for time_step in valid_time_steps:
                    iotdb_api.set_time_step(time_step)
                    try:
                        cell_indexes = iotdb_api.range_query_var(iotdb_entity, lower_bound, upper_bound, attribute_name)
                    except StatementExecutionException:
                        print(f"SKIP_DATASET_EXCEPTION: Ship={ship_type} Stage=RangeQuery Step={time_step}")
                        skip_current_dataset = True
                        break
                    if cell_indexes is None or len(cell_indexes) == 0:
                        print(f"SKIP_DATASET_EMPTY: Ship={ship_type} Stage=RangeQuery Step={time_step}")
                        skip_current_dataset = True
                        break

                if skip_current_dataset:
                    break
                iotdb_transaction += 1

            if skip_current_dataset:
                continue
            print(f"Total IoTDB Transactions ({ship_type}): {iotdb_transaction}")
        continue

if __name__ == "__main__":
    main()