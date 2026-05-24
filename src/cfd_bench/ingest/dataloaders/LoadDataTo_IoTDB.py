from operator import index
import os
from tkinter import Variable
from xml.dom.minidom import Element
from iotdb.Session import Session
from iotdb.utils.IoTDBConstants import TSDataType, TSEncoding, Compressor
from numpy import var
from .Dat_Data_Decoder import CAE_Decoder
from .IoTDB_Interface import load_dataframe_to_IoTDB
from . import Zone
import pandas as pd
import numpy as np
from pathlib import Path




def main(input_path = None):
    # Establishing IoTDB connections
    # Constants for connecting to IoTDB
    HOST = "127.0.0.1"
    PORT = "6667"
    USER = "root"
    PASSWORD = "root"

    # Create a session to connect to IoTDB
    session = Session(HOST, PORT, USER, PASSWORD)
    session.open()

    
    DIMENSIONS = ['X','Y','Z']
    COMMON_IOTDB_DIRECTORY = "root.simulation_data.post_processing_management"
    SHIP_TYPE = "Kvlcc2_351k"

    if input_path is not None:
        path = input_path
    else:
        project_root = Path(__file__).resolve().parents[4]
        path = project_root / "data" / "Kvlcc2_351k" / "Postprocessing"
    print("Decoding Post-processing data")
    if not os.path.exists(path):
        print(f"Error, path does not exist: {path}")
        return
    elif not os.path.isdir(path):
        print(f"Error, please input the directory to which the .dat files are located: {path}")
        
    Geometry_Loaded = False
    for file in os.listdir(path):
        
        IoTDir = COMMON_IOTDB_DIRECTORY + "." + SHIP_TYPE + "."
        
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath):  # Process Files only
            # Check if the filename ends with .dat
            if filepath.lower().endswith('.dat') == False:
                continue
                
        print(f"Processing file: {filepath}")
        data = CAE_Decoder(3)
        data.Decode_dat_file(filepath)
        # Thus far, only the fluid zone and hull zone are our interest
        
        fluid_zone:Zone.Zone_3D = data.Zones[0]

        hull_zone:Zone.Zone_3D = data.Zones[1]

        

        # Loading Node coordinates
        if Geometry_Loaded == False:
            # node_indexes = [i for i in range(0, fluid_zone.Node_count)]
            # df_node_coordinates:pd.DataFrame = pd.DataFrame(index = node_indexes, columns = [])
            # for i in range(0, len(DIMENSIONS)): 
            #     tmp_series = pd.Series(fluid_zone.Node_Coordinates[i], index = node_indexes, dtype = np.float64, name = DIMENSIONS[i])
            #     df_node_coordinates = pd.concat([df_node_coordinates, tmp_series], axis = 1)
            
            # load_dataframe_to_IoTDB(IoTDir + "Nodes", session, df_node_coordinates)


            # Loading Element Coordinates
            element_indexes = [i for i in range(0, fluid_zone.Element_count)]
            df_element_coordinates:pd.DataFrame = pd.DataFrame(index = element_indexes, columns = [])
            hull_element_indexes = [i for i in range(0, hull_zone.Element_count)]
            hull_df_element_coordinates:pd.DataFrame = pd.DataFrame(index = hull_element_indexes, columns = [])

            for i in range(0, len(DIMENSIONS)): 
                tmp_series = pd.Series(fluid_zone.Element_Coordinates[i], index = element_indexes, dtype = np.float64, name = DIMENSIONS[i])
                df_element_coordinates = pd.concat([df_element_coordinates, tmp_series], axis = 1)
                hull_tmp_series = pd.Series(hull_zone.Element_Coordinates[i], index = hull_element_indexes, dtype = np.float64, name = DIMENSIONS[i])
                hull_df_element_coordinates = pd.concat([hull_df_element_coordinates, hull_tmp_series], axis = 1)
            
            load_dataframe_to_IoTDB(IoTDir + "Elements", session, df_element_coordinates)
            load_dataframe_to_IoTDB(IoTDir + "Elements", session, hull_df_element_coordinates)
            
            Geometry_Loaded = True
                
        # Loading Variables
        df_element_variables:pd.DataFrame = pd.DataFrame(index = element_indexes, columns = [])
        hull_df_element_variables:pd.DataFrame = pd.DataFrame(index = hull_element_indexes, columns = [])

        variables = data.Variables[3:]
        for i in range(0, len(variables)):
            tmp_series = pd.Series(fluid_zone.Element_Variables[i], index = element_indexes, dtype = np.float64, name = variables[i])
            df_element_variables = pd.concat([df_element_variables, tmp_series], axis = 1)

        hull_tmp_series = pd.Series(hull_zone.Element_Variables[3], index = hull_element_indexes, dtype = np.float64, name = "P")
        hull_df_element_variables = pd.concat([hull_df_element_variables, hull_tmp_series], axis = 1)
        
        tmp_IoTDir = IoTDir + "step_" + file.removesuffix(".dat")
        load_dataframe_to_IoTDB(tmp_IoTDir + ".Variables", session, df_element_variables)

        hull_tmp_IoTDir = IoTDir + "step_" + file.removesuffix(".dat")
        load_dataframe_to_IoTDB(hull_tmp_IoTDir + ".Variables", session, hull_df_element_variables)



if __name__ == "__main__":
    main()