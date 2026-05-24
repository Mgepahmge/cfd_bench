## import DB_Interface_PG as DB_API  # IoTDB-only mode
from cfd_bench.infra.iotdb import legacy_variables_client as IoTDB_API
## import TileDB_Interface as TDB_API  # IoTDB-only mode
import numpy as np
from numpy.typing import NDArray
import VTK_Interface as VTK_API
import vtk
import os
import random
import time
import re
import psutil  # 用于监测资源

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

# --- 辅助函数 ---
def calculate_force(normals: NDArray[np.float64], pressures: NDArray[np.float64]) -> NDArray[np.float64]:
    total_force = np.array([0.0, 0.0, 0.0])
    for i in range(len(normals)):
        normal = normals[i]
        pressure = pressures[i]
        force = pressure * normal 
        total_force += force
    return total_force

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

    VTK_HULL_DIR = "../../vtk_hull_dir/"
    TileDB_DIR = "../../TileDB_Instances/"
    vtk_hull_files = [f for f in os.listdir(VTK_HULL_DIR) if f.endswith(".vtk")]

    for ship_type in SHIP_TYPE_LIST:
        for time_step in TIME_STEP_LIST:
            # 跳过不存在的时间步
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                continue

            vtk_hull_file, = [f for f in vtk_hull_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
            vtk_hull_mesh = vtk_api.vtk_connect(os.path.join(VTK_HULL_DIR, vtk_hull_file))
            cell_indices = np.array(range(vtk_hull_mesh.GetNumberOfCells()))

            # IoTDB-only mode: run IoTDB section and skip PostgreSQL/TileDB/VTK.
            with UnifiedResourceMonitor(f"IoTDB_{ship_type}_{time_step}", target_process_names=["iotdb", "java"]):
                iotdb_api.set_ship_type(ship_type + "hull")
                iotdb_api.set_time_step(time_step)

                iotdb_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    iotdb_norm_vectors = VTK_API.VTK_Interface.vtk_surface_norm(vtk_hull_mesh)
                    pressures = iotdb_api.point_query(iotdb_entity, cell_indices, 'P')
                    calculate_force(iotdb_norm_vectors, pressures)
                    iotdb_transaction += 1
                # print(f"Total IoTDB Transactions ({ship_type}_{time_step}): {iotdb_transaction}")
            continue

            ''' W 6.3 TileDB '''
            with UnifiedResourceMonitor(f"TileDB_{ship_type}_{time_step}"):
                ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)
                tiledb_file, = [f for f in os.listdir(ship_tiledb_dir) if re.match(rf"^{time_step}(?!\d).*hull\.tdb$", f)]
                tdb_entity = tdb_api.Load_TileDB_File(os.path.join(ship_tiledb_dir, tiledb_file))
                
                tiledb_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    tiledb_norm_vectors = VTK_API.VTK_Interface.vtk_surface_norm(vtk_hull_mesh)
                    pressures = tdb_api.Point_Query_Attribute_TileDB(tdb_entity, 'P', cell_indices)
                    calculate_force(tiledb_norm_vectors, pressures)
                    tiledb_transaction += 1
                # print(f"Total TileDB Transactions ({ship_type}_{time_step}): {tiledb_transaction}")

            ''' W 6.4 VTK '''
            with UnifiedResourceMonitor(f"VTK_{ship_type}_{time_step}"):
                vtk_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    vtk_norm_vectors = VTK_API.VTK_Interface.vtk_surface_norm(vtk_hull_mesh)
                    pressures = vtk_api.point_query(vtk_hull_mesh, cell_indices, 'P')
                    calculate_force(vtk_norm_vectors, pressures)
                    vtk_transaction += 1
                # print(f"Total VTK Transactions ({ship_type}_{time_step}): {vtk_transaction}")

if __name__ == "__main__":
    main()