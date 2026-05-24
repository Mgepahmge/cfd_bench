"""IoTDB ingest write helpers."""

from __future__ import annotations

import pandas as pd
from iotdb.Session import Session
from iotdb.utils.Field import TSDataType
from iotdb.table_session import Tablet


def load_dataframe_to_iotdb(device_directory: str, session: Session, df: pd.DataFrame):
    device = device_directory
    times = df.index.values
    measurements_list = list(df.columns)
    types_list = [TSDataType.DOUBLE] * len(measurements_list)
    values_list = [df[col].values for col in df.columns]
    vertical_values_array = list(map(list, zip(*values_list)))
    tablet = Tablet(device, measurements_list, types_list, vertical_values_array, times)
    session.insert_tablet(tablet)
