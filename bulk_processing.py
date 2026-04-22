import os
import lasio
import pandas as pd


def load_las_folder(folder_path):
    """Read all .las files in a folder and combine into one DataFrame"""
    las_files = [f for f in os.listdir(folder_path) if f.endswith(".las")]
    print(f"Found {len(las_files)} LAS files. Processing...")

    all_data = []
    for i, filename in enumerate(las_files):
        try:
            las = lasio.read(os.path.join(folder_path, filename))
            df = las.df().reset_index()
            df["WELL_NAME"] = filename
            all_data.append(df)
        except Exception as e:
            print(f"Skipped {filename}: {e}")

        if (i + 1) % 5 == 0 or (i + 1) == len(las_files):
            print(f"  {i + 1}/{len(las_files)} done")

    if not all_data:
        print("No data loaded. Check folder path.")
        return None

    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    folder   = "data/FORCE_2020_RAW"
    out_path = "data/master_training_data.csv"

    df = load_las_folder(folder)
    if df is not None:
        df.to_csv(out_path, index=False)
        print(f"Saved {df.shape[0]} rows to {out_path}")
