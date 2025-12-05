import numpy as np
from numpy.typing import NDArray
from iotdb.Session import Session
from utils import dataset_to_ndarray

class Iotdb_Interface:
    def __init__(self, ship_type: str, time_step: str):
        self.ship_type = ship_type
        self.time_step = time_step
        self.db_entity = None

    def iotdb_connect(self):
        HOST = "127.0.0.1"
        PORT = "6667"
        USER = "root"
        PASSWORD= "root"
        db_entity = Session(HOST, PORT, USER, PASSWORD)
        db_entity.open()
        return db_entity
    
    def set_ship_type(self, ship_type: str):
        self.ship_type = ship_type

    def set_time_step(self, time_step: str):
        self.time_step = time_step

    def point_query(self, db_entity, cell_indexes:np.array, attribute_name:str) -> NDArray[np.float64]:
        path = "root.simulation_data.post_processing_management" + f".{self.ship_type}.step_{self.time_step}.Variables"
        indices_str = ','.join(map(str, cell_indexes))
        sql = f"SELECT {attribute_name} FROM {path} WHERE Time IN ({indices_str});"
        query_result = db_entity.execute_query_statement(sql)
        _,values = dataset_to_ndarray(query_result)
        return np.array(values, dtype=np.float64)

    def range_query_var(self, db_entity, lower_bound:float, upper_bound:float, attribute_name:str):
        path = "root.simulation_data.post_processing_management" + f".{self.ship_type}.step_{self.time_step}.Variables"
        sql = f"SELECT * FROM {path} WHERE {attribute_name} BETWEEN {lower_bound} AND {upper_bound};"
        query_result = db_entity.execute_query_statement(sql)
        cell_indexes,_ = dataset_to_ndarray(query_result) 
        return np.array(cell_indexes, dtype=np.int32)
    
    def range_query_coord(self, db_entity, lower_bound:NDArray[np.float64], upper_bound:NDArray[np.float64]):
        path = "root.simulation_data.post_processing_management" + f".{self.ship_type}.Elements"
        X_lower, Y_lower, Z_lower = lower_bound
        X_upper, Y_upper, Z_upper = upper_bound
        sql = f"SELECT * FROM {path} WHERE (X BETWEEN {X_lower} AND {X_upper}) AND (Y BETWEEN {Y_lower} AND {Y_upper}) AND (Z BETWEEN {Z_lower} AND {Z_upper});"
        query_result = db_entity.execute_query_statement(sql)
        cell_indexes,_ = dataset_to_ndarray(query_result) 
        return np.array(cell_indexes, dtype=np.int32)
