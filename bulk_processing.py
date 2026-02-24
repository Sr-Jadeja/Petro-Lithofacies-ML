import lasio
import pandas as pd
import os

def process_all_wells(folder_path):
    all_data = []
    # Get list of all .las files
    las_files = [f for f in os.listdir(folder_path) if f.endswith('.las')]
    total_files = len(las_files)
    
    print(f"Found {total_files} wells. Starting conversion...")
    
    for index, file in enumerate(las_files):
        try:
            path = os.path.join(folder_path, file)
            # lasio reads the file
            las = lasio.read(path)
            # Convert to dataframe and reset index to make 'DEPT' a column
            df = las.df().reset_index()
            
            # Identify the well name for the resume-ready 'Leave-One-Well-Out' validation
            df['WELL_NAME'] = file
            
            all_data.append(df)
            
            # Simple progress tracker
            if (index + 1) % 5 == 0 or (index + 1) == total_files:
                print(f"Processed {index + 1}/{total_files} wells...")
                
        except Exception as e:
            print(f"Could not read {file}: {e}")

    # Combine all into one massive dataframe
    if all_data:
        final_df = pd.concat(all_data, axis=0, ignore_index=True)
        return final_df
    else:
        print("No data was loaded. Check your folder path.")
        return None

if __name__ == "__main__":
    # 1. Update this path to where your unzipped .las files are
    raw_data_path = "data/FORCE_2020_RAW" 
    
    # 2. Run the process
    master_df = process_all_wells(raw_data_path)
    
    if master_df is not None:
        # 3. Save as a single CSV for the model
        # Using a CSV for now so you can easily inspect it in Excel
        master_df.to_csv("data/master_training_data.csv", index=False)
        print(f"Success! Master dataset created with {master_df.shape[0]} rows.")
        print("File saved to: data/master_training_data.csv")