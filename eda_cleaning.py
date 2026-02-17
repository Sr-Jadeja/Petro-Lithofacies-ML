import pandas as pd
import numpy as np
from data_loading import load_well_data

def clean_and_map(df):
    """
    Cleans the dataframe and maps numeric facies to names.
    """
    # 1. Rename columns for readability (Industry Standard)
    # FORCE 2020 uses specific codes for lithology
    lithology_numbers = {
        30000: 'Sandstone',
        65030: 'Sandstone/Shale',
        65000: 'Shale',
        80000: 'Marl',
        74000: 'Dolomite',
        70000: 'Limestone',
        70032: 'Chalk',
        88000: 'Halite',
        86000: 'Anhydrite',
        99000: 'Tuff',
        90000: 'Coal',
        93000: 'Basement'
    }

    # Rename the long target column to 'LITHOLOGY'
    df.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY': 'FACIES_ID'}, inplace=True)
    
    # Map IDs to Labels
    df['LITHOLOGY'] = df['FACIES_ID'].map(lithology_numbers)

    # 2. Check for missing values (crucial for Classical ML)
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # 3. Simple Cleaning: Drop rows where our Target (Facies) is missing
    # We can't train a supervised model if we don't know the answer!
    df = df.dropna(subset=['FACIES_ID'])

    return df

if __name__ == "__main__":
    well_path = "data/15_9-13.las"
    raw_df = load_well_data(well_path)
    
    if raw_df is not None:
        clean_df = clean_and_map(raw_df)
        print("\nCleaned Data Preview:")
        print(clean_df[['DEPT', 'FACIES_ID', 'LITHOLOGY']].head())
        
        # Save this as a CSV for the next step (Model Training)
        clean_df.to_csv("data/cleaned_well_data.csv", index=False)
        print("\nCleaned data saved to data/cleaned_well_data.csv")