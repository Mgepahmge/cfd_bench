from textwrap import indent
import types
import iotdb
from iotdb.utils.Field import TSDataType
import pandas as pd
import numpy as np
import os
import time
from iotdb.Session import Session
from iotdb.table_session import Tablet
import sys
'''
# The function used for loading dataframe to IoTDB
'''
def load_dataframe_to_IoTDB(device_directory: str, session: Session, df: pd.DataFrame):
        
    device = device_directory
    times = df.index.values
    measurements_list = []
    types_list = []
    values_list = []
    for col in df.columns:
        measurements_list.append(col)
        tmp_col = df[col]
        types_list.append(TSDataType.DOUBLE)
        values_list.append(tmp_col.values)

    vertical_values_array = list(map(list, zip(*values_list)))
    tablet = Tablet(device, measurements_list, types_list, vertical_values_array, times)

    # Batch insert the data
    print("Inserting data...")
    try:
        session.insert_tablet(tablet)
        print("Insertion successful.")
    except Exception as e:
        print(f"Failed to insert data: {e}")
    return


#===TOOLS===#
# Below are the tools for implementing the workloads

'''
    Func get_variables_by_ids(variables:list, ids:list, path:str)
    Command like
    SELECT variables FROM path WHERE Time = ids
    return: The SQL command for IoTDB
'''
def get_variables_by_ids(variables:list, ids:list, path:str):
    variable_as_string = ""
    for v in variables:
        variable_as_string += v + ","
    variable_as_string = variable_as_string[:-1]
    
    ids_as_string = ""
    for i in ids:
        ids_as_string += str(i) + ","
    ids_as_string = ids_as_string[:-1]
        
    SQL_COMMAND = "SELECT " + variable_as_string + " FROM " + path + " WHERE Time in (" + ids_as_string +")" 

    return SQL_COMMAND


'''
    Func get_ids_by_threshold(variables:list, ranges:list)
    Command Like
    SELECT variables FROM path WHERE variables[i] in ranges[i]
    ranges[i] is an array of length 2, like [lower, upper], specifying the range of query. Setting to -/+inf if the lower/upper is not specified
    return: The SQL command for IoTDB
'''
def get_ids_by_threshold(variables:list, ranges:list, path:str):
    if len(variables) != len(ranges):
        print("ERROR, parameters do not match...")
        return ""
    
    variable_as_string = ""
    for v in variables:
        variable_as_string += v + ","
    variable_as_string = variable_as_string[:-1]
    
    sub_predicate = ""
    for i in range(0, len(variables)):
        v = variables[i]
        r = ranges[i]
        
        # Processing the lower boundary
        if ranges[i][0] != sys.float_info.min:
            sub_predicate += v + ">" + str(ranges[i][0]) + " AND "
        
        if ranges[i][1] != sys.float_info.max:
            sub_predicate += v + "<" + str(ranges[i][1]) + " AND "
        
    tail_to_remove = " AND "
    sub_predicate = sub_predicate[0 : -len(tail_to_remove)]
    
    SQL_COMMAND = "SELECT " + variable_as_string + " FROM " + path +" WHERE " + sub_predicate
        
    return SQL_COMMAND



def print_query_result(session_dataset, length_limit):
    # Iterate over the TimeSeries in the query result
    # Check if the query returned any data
    if session_dataset is not None:
        # 1. Print Column Names (Schema)
        print("Columns:")
        for column in session_dataset.get_columns():
            print(f"  - {column}")

        print("\nData:")
        # 2. Iterate through each row in the dataset
        
        for _ in range(0,length_limit):
            for row in session_dataset:
                # 3. Access data points in the row
                #    row is an object with methods like get_timestamp() and get_field(index)
                timestamp = row.get_timestamp()
                print(f"Timestamp: {timestamp}")

                # Print each value in the row
                for i in range(len(session_dataset.get_columns())):
                    column_name = session_dataset.get_columns()[i]
                    value = row.get_field(i)
                    print(f"  {column_name}: {value}")

                print() # Add a blank line between rows for readability
    else:
        print("Query returned no data or an error occurred.")