import pandas as pd
from model_preparation import prepare_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def train_and_evaluate():
    # 1. Load the prepared data
    X_train, X_test, y_train, y_test, features = prepare_data("data/cleaned_well_data.csv")

    # 2. Initialize Models
    # Random Forest is robust to outliers which are common in well logs
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # SVM is excellent for complex boundaries (requires scaled data)
    svm_model = SVC(kernel='rbf', random_state=42, class_weight='balanced')

    # 3. Training
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    print("Training Support Vector Machine...")
    svm_model.fit(X_train, y_train)

    # 4. Evaluation - Random Forest
    rf_preds = rf_model.predict(X_test)
    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_test, rf_preds))

    # 5. Visualizing results with a Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, rf_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix: Random Forest Lithology Prediction")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("data/rf_confusion_matrix.png")
    
    # 6. Save the best model
    joblib.dump(rf_model, 'data/lithology_model.pkl')
    print("\nBest model saved to data/lithology_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()