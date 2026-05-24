## import DB_Interface_PG as DB_API  # IoTDB-only mode
from cfd_bench.infra.iotdb import legacy_variables_client as IoTDB_API
## import TileDB_Interface as TDB_API  # IoTDB-only mode
import numpy as np
from numpy.typing import NDArray
from vtk import vtkUnstructuredGrid ,vtkGradientFilter, vtkDataObject
import vtk
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
        print(f"  内存变化: {end_mem - self.start_mem:+.2f} MB (当前总占用: {end_mem:.2f} MB)")
        print(f"  磁盘读取: {max(0, end_read - self.start_read):.2f} MB")
        print("-" * 55)

# --- 辅助函数 ---
def random_range(vtk_mesh:vtkUnstructuredGrid) -> NDArray[np.float64]:
    bounds = vtk_mesh.GetBounds()
    x1, x2 = sorted([random.uniform(bounds[0], bounds[1]), random.uniform(bounds[0], bounds[1])])
    y1, y2 = sorted([random.uniform(bounds[2], bounds[3]), random.uniform(bounds[2], bounds[3])])
    z1, z2 = sorted([random.uniform(bounds[4], bounds[5]), random.uniform(bounds[4], bounds[5])])
    return np.array([[x1, y1, z1], [x2, y2, z2]])

def ComputeQCriterion(vtk_mesh:vtkUnstructuredGrid) -> NDArray[np.float64]:
    velocity_array = vtk_mesh.GetPointData().GetArray("Velocity")
    if velocity_array:
        gradient_filter = vtkGradientFilter()
        gradient_filter.SetInputData(vtk_mesh)
        gradient_filter.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, "Velocity")
        gradient_filter.SetComputeQCriterion(True)
        gradient_filter.Update()
        mesh_with_q = gradient_filter.GetOutput()
        QCriterion_array = mesh_with_q.GetPointData().GetArray("QCriterion")
        return QCriterion_array
        
def main():
    vtk.vtkObject.GlobalWarningDisplayOff()
    ''' Pre-processing '''
    iotdb_api = IoTDB_API.Iotdb_Interface(os.getenv("SHIP_TO_RUN", "JBC_615k"), os.getenv("STEP_TO_RUN", "200"))
    iotdb_entity = iotdb_api.iotdb_connect()
    vtk_api = VTK_API.VTK_Interface()

    # 优先从环境变量读取，配合自动化 Bash 脚本
    env_ship = os.getenv("SHIP_TO_RUN")
    env_step = os.getenv("STEP_TO_RUN")

    SHIP_TYPE_LIST = [env_ship] if env_ship else ["JBC_615k"]
    TIME_STEP_LIST = [env_step] if env_step else ["200"]

    VTK_MESH_DIR = "../../vtk_dir/"   
    vtk_geo_files = [f for f in os.listdir(VTK_MESH_DIR) if f.endswith(".vtk")]
    TileDB_DIR = "../../TileDB_Instances/"

    for ship_type in SHIP_TYPE_LIST:
        for time_step in TIME_STEP_LIST:
            # 跳过不存在的时间步
            if time_step == "2000" and (ship_type == "JBC_3843k" or ship_type == "Kvlcc2_3709k"):
                continue

            vtk_file, = [f for f in vtk_geo_files if (ship_type in f) and f.endswith(f"_{time_step}.vtk")]
            vtk_mesh = vtk_api.vtk_connect(os.path.join(VTK_MESH_DIR, vtk_file))

            # IoTDB-only mode: run IoTDB section and skip PostgreSQL/TileDB/VTK.
            skip_current_dataset = False
            with UnifiedResourceMonitor(f"IoTDB_{ship_type}_{time_step}", target_process_names=["iotdb", "java"]):
                iotdb_api.set_ship_type(ship_type)
                iotdb_api.set_time_step(time_step)

                iotdb_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    try:
                        while True:
                            lower_bound, upper_bound = random_range(vtk_mesh)
                            cell_indexes = iotdb_api.range_query_coord(iotdb_entity, lower_bound, upper_bound)
                            if cell_indexes.size > 0:
                                break
                        sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)
                        ComputeQCriterion(sub_mesh)
                    except StatementExecutionException:
                        print(f"SKIP_DATASET_EXCEPTION: Ship={ship_type} Step={time_step} Stage=RangeQuery")
                        skip_current_dataset = True
                        break
                    else:
                        iotdb_transaction += 1
                # print(f"Total IoTDB Transactions ({ship_type}_{time_step}): {iotdb_transaction}")
            if skip_current_dataset:
                continue
            continue

            ''' W 7.3  Testing TileDB '''
            with UnifiedResourceMonitor(f"TileDB_{ship_type}_{time_step}"):
                ship_tiledb_dir = os.path.join(TileDB_DIR, ship_type)
                tiledb_file, = [f for f in os.listdir(ship_tiledb_dir) if re.match(rf"^{time_step}(?!\d).*fluid\.tdb$", f)]
                tdb_entity = tdb_api.Load_TileDB_File(os.path.join(ship_tiledb_dir, tiledb_file))

                tiledb_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    while True:
                        lower_bound, upper_bound = random_range(vtk_mesh)
                        cell_indexes = tdb_api.Spatial_Range_Query_TileDB(tdb_entity, lower_bound, upper_bound)
                        if cell_indexes.size > 0:
                            break
                    sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)
                    ComputeQCriterion(sub_mesh)
                    tiledb_transaction += 1
                # print(f"Total TileDB Transactions ({ship_type}_{time_step}): {tiledb_transaction}")

            ''' W 7.4  Testing VTK '''
            with UnifiedResourceMonitor(f"VTK_{ship_type}_{time_step}"):
                vtk_transaction = 0
                start_time = time.time()
                while time.time() - start_time < 60:
                    while True:
                        lower_bound, upper_bound = random_range(vtk_mesh)
                        cell_indexes = vtk_api.range_query_coord(vtk_mesh, lower_bound, upper_bound)
                        if cell_indexes.size > 0:
                            break
                    sub_mesh = VTK_API.VTK_Interface.vtk_extract_submesh(cell_indexes, vtk_mesh)
                    ComputeQCriterion(sub_mesh)
                    vtk_transaction += 1
                # print(f"Total VTK Transactions ({ship_type}_{time_step}): {vtk_transaction}")

if __name__ == "__main__":
    main()