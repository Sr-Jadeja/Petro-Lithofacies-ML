import pandas as pd
import joblib
import lasio
from data_loading import load_well_data

def predict_new_well(las_file_path):
    # 1. Load raw data using our Step 2 logic
    print(f"Reading new well data from: {las_file_path}")
    df = load_well_data(las_file_path)
    
    # 2. Load the artifacts from Step 5 & 6
    scaler = joblib.load('data/scaler.pkl')
    model = joblib.load('data/lithology_model.pkl')
    
    # 3. Feature Selection (Must match training exactly)
    features = ['DEPT', 'CALI', 'RDEP', 'RMED', 'RSHA']
    
    # Handle missing values in the new data
    df_clean = df.dropna(subset=features).copy()
    
    # 4. Scaling
    X_scaled = scaler.transform(df_clean[features])
    
    # 5. Prediction
    print("Generating lithology predictions...")
    predictions = model.predict(X_scaled)
    
    # Add predictions back to the dataframe
    df_clean['PREDICTED_FACIES'] = predictions
    
    # 6. Map IDs back to Rock Names (for the human user)
    lithology_numbers = {
        30000: 'Sandstone', 65030: 'Sandstone/Shale', 65000: 'Shale',
        80000: 'Marl', 74000: 'Dolomite', 70000: 'Limestone',
        70032: 'Chalk', 88000: 'Halite', 86000: 'Anhydrite',
        99000: 'Tuff', 90000: 'Coal', 93000: 'Basement'
    }
    df_clean['LITHOLOGY_NAME'] = df_clean['PREDICTED_FACIES'].map(lithology_numbers)
    
    # Save the final result
    output_path = "data/well_predictions.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return df_clean

if __name__ == "__main__":
    # We test it on our current well, but this could be any .las file!
    results = predict_new_well("data/15_9-13.las")
    print("\nSample of Predicted Results:")
    print(results[['DEPT', 'LITHOLOGY_NAME']].tail(10))