import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def train_lithology_model(csv_path):
    print(f"Loading data from {csv_path}...")
    # Using low_memory=False for better handling of large datasets
    df = pd.read_csv(csv_path, low_memory=False)

    # 1. Feature Selection
    features = ['DEPT', 'CALI', 'RDEP', 'RMED', 'RSHA']
    X = df[features]
    y = df['FACIES_ID']
    groups = df['WELL_NAME'] 

    # 2. Group-Based Splitting (20% of wells for validation)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Training on {groups.iloc[train_idx].nunique()} wells.")
    print(f"Validating on {groups.iloc[test_idx].nunique()} unseen wells.")

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler immediately
    joblib.dump(scaler, 'data/lithology_scaler.pkl')

    # 4. Optimized Model Training
    # max_samples and n_estimators are tuned to prevent RAM issues
    print("Training Optimized Random Forest (Estimated time: 3-7 minutes)...")
    rf = RandomForestClassifier(
        n_estimators=50, 
        max_samples=0.1,    # Uses 10% of rows per tree to save RAM
        max_depth=20,       # Prevents over-complex trees
        class_weight='balanced', 
        n_jobs=-1, 
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)

    # 5. Evaluation
    y_pred = rf.predict(X_test_scaled)
    print("\n--- Model Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 6. Visualization
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    # Get unique labels present in the test set for the heatmap
    labels = sorted(y_test.unique())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix: Multi-Well Global Model")
    plt.xlabel("Predicted Facies ID")
    plt.ylabel("True Facies ID")
    plt.savefig("data/final_confusion_matrix.png")
    
    # Save the final model
    joblib.dump(rf, 'data/lithology_model.pkl')
    print("Success! Model and Scaler saved to /data folder.")

if __name__ == "__main__":
    train_lithology_model("data/cleaned_master_data.csv")