class Tiledb_Interface:

    def tiledb_connect():
        # TODO
        return db_entity

    def point_query(db_entity, cell_indexes:np.array, attribute_names:list[str]) -> NDArray[np.float64]:
        # TODO
        return attribute_values

    def range_query(db_entity, ranges:NDArray[np.float64], attribute_names:list[str]): # ranges is a 2-D array, with len(ranges = len(attribute_names))
        # TODO
        return cell_indexes