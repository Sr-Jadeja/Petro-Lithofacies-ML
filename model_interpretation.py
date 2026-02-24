import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

def explain_model():
    # 1. Load the saved model
    # Ensure the path matches where your model was saved in Step 6
    try:
        model = joblib.load('data/lithology_model.pkl')
    except FileNotFoundError:
        print("Error: lithology_model.pkl not found. Please run model_training.py first.")
        return

    # 2. Define the features (must match the order used during training in Step 5)
    features = ['DEPT', 'CALI', 'RDEP', 'RMED', 'RSHA']

    # 3. Get Feature Importances from the Random Forest
    # This measures how much each feature decreases the weighted impurity (Gini)
    importances = model.feature_importances_
    
    # Sort indices in descending order
    indices = np.argsort(importances)

    # 4. Plotting the Feature Importance
    plt.figure(figsize=(10, 6))
    plt.title('Random Forest: Feature Importances for Lithology Prediction')
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot for your Resume/Portfolio
    plt.savefig("data/feature_importance.png")
    print("Feature importance plot saved to data/feature_importance.png")

    # 5. Print a summary for the user
    print("\nFeature Ranking:")
    for f in range(len(features)):
        print(f"{f + 1}. {features[indices[-f-1]]}: {importances[indices[-f-1]]:.4f}")

if __name__ == "__main__":
    explain_model()