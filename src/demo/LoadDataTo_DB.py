from ..DataLoader.LoadDataTo_PG import main as load_data_to_PG
from ..DataLoader.LoadDataTo_IoTDB import main as load_data_to_IoTDB
from ..DataLoader.LoadDataTo_TileDB import main as load_data_to_TileDB
from ..DataLoader.LoadDataTo_VTK import main as load_data_to_VTK

def main():

    # load data to PostgreSQL
    load_data_to_PG()

    # Load data to IoTDB
    load_data_to_IoTDB()
    
    # Load data to TileDB
    load_data_to_TileDB()
    
    # Load data to VTK
    load_data_to_VTK()
    return

if __name__ == "__main__":
    main()
