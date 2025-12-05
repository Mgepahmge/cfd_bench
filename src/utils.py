import pandas as pd
import numpy as np
from iotdb.Session import SessionDataSet

def dataset_to_ndarray(dataset: SessionDataSet) -> np.ndarray:
    """
    将 IoTDB 的 SessionDataSet 转换为 numpy array，只返回 'value' 列（转换为 float）。

    参数:
        dataset (SessionDataSet): IoTDB 查询结果

    返回:
        np.ndarray: 'value' 列的 numpy array
    """
    values = []
    cell_ids = []

    while dataset.has_next():
        row = dataset.next()
        cell_ids.append(row.get_timestamp())
        # 假设只取第一个值列（如果有多个值列，可以自行扩展）
        
        values.append(row.get_fields()[0].get_string_value())

    # 转换为 numpy array 并转为 float
    cell_ids_array = np.array(cell_ids, dtype=int)
    values_array = np.array(values, dtype=float)

    return cell_ids_array, values_array