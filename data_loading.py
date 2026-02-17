import lasio
import pandas as pd
import os

def load_well_data(file_path):
    """
    Loads a .las file and converts it to a cleaned Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    # 1. Read the LAS file
    las = lasio.read(file_path)

    # 2. Convert to Dataframe
    # las.df() automatically sets 'DEPT' (Depth) as the index
    df = las.df().reset_index()

    print(f"Successfully loaded {file_path}")
    print(f"Curves present: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    # Update the filename below to match what you downloaded
    well_path = "data/15_9-13.las" 
    well_df = load_well_data(well_path)
    
    if well_df is not None:
        print("\nFirst 5 rows of data:")
        print(well_df.head())