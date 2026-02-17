import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # 1. Feature Selection
    # We select common logs available in this well that usually affect lithology
    # 'DEPT' is included because lithology often changes with depth (compaction/age)
    features = ['DEPT', 'CALI', 'RDEP', 'RMED', 'RSHA'] 
    target = 'FACIES_ID'

    # Drop any remaining rows with NaNs in our specific feature set
    df_ml = df.dropna(subset=features + [target])

    X = df_ml[features]
    y = df_ml[target]

    # 2. Train-Test Split (Resume Note: 80/20 split is standard)
    # random_state ensures reproducibility - crucial for professional projects
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 'stratify=y' ensures the % of Shale/Sandstone is the same in both sets

    # 3. Scaling (The "Classical ML" essential)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Save the scaler for later use (Inference)
    joblib.dump(scaler, 'data/scaler.pkl')
    print("Scaler saved to data/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feat_names = prepare_data("data/cleaned_well_data.csv")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Features used: {feat_names}")