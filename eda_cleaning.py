import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def train_global_model(file_path):
    # 1. Load Data
    print("Loading cleaned master data...")
    df = pd.read_csv(file_path)
    
    # 2. Define Features and Target
    features = ['DEPT', 'CALI', 'RDEP', 'RMED', 'RSHA']
    X = df[features]
    y = df['FACIES_ID']
    groups = df['WELL_NAME'] # This is the "secret sauce" for professional projects

    # 3. Group-Based Split (80% wells for training, 20% wells for testing)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Training on {groups.iloc[train_idx].nunique()} wells.")
    print(f"Testing on {groups.iloc[test_idx].nunique()} wells.")

    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'data/master_scaler.pkl')

    # 5. Training - Random Forest
    # We use n_jobs=-1 to use all your CPU cores for faster training
    print("Training Global Random Forest (this may take a minute)...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # 6. Evaluation
    y_pred = rf.predict(X_test_scaled)
    print("\n--- Global Model Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 7. Confusion Matrix Visualization
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Confusion Matrix: Global Lithology Model")
    plt.xlabel("Predicted Facies ID")
    plt.ylabel("True Facies ID")
    plt.savefig("data/master_confusion_matrix.png")
    
    # 8. Save the Global Model
    joblib.dump(rf, 'data/master_lithology_model.pkl')
    print("Global model saved to data/master_lithology_model.pkl")

if __name__ == "__main__":
    train_global_model("data/cleaned_master_data.csv")