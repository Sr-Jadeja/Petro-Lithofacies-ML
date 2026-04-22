import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ─────────────────────────────────────────────────────

DATA_PATH     = "data/cleaned_master_data.csv"
FEATURES_PATH = "data/selected_features.json"
MODEL_PATH    = "data/lithology_model.pkl"
SCALER_PATH   = "data/lithology_scaler.pkl"
TARGET        = "FACIES_ID"

# ── Load features selected by clean.py ────────────────────────

with open(FEATURES_PATH) as f:
    FEATURES = json.load(f)

print(f"Features loaded from {FEATURES_PATH}: {FEATURES}")

# ── Load Data ──────────────────────────────────────────────────

print(f"\nLoading {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Dataset: {df.shape[0]:,} rows, {df['WELL_NAME'].nunique()} wells")

X      = df[FEATURES]
y      = df[TARGET]
groups = df["WELL_NAME"]

# ── Group-Based Train/Test Split ───────────────────────────────
# Split by well — test wells are completely unseen during training

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Training wells: {groups.iloc[train_idx].nunique()}")
print(f"Test wells:     {groups.iloc[test_idx].nunique()}")

# ── Scaling ────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# ── Train Random Forest ────────────────────────────────────────

print("Training Random Forest (may take a few minutes)...")
rf = RandomForestClassifier(
    n_estimators=50,
    max_samples=0.1,      # use 10% of rows per tree to save RAM
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# ── Evaluation ─────────────────────────────────────────────────

y_pred = rf.predict(X_test_scaled)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion matrix
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Facies ID")
plt.ylabel("True Facies ID")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png")
print("Saved data/confusion_matrix.png")

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(8, 5))
plt.barh([FEATURES[i] for i in indices], importances[indices], color="teal")
plt.xlabel("Relative Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("data/feature_importance.png")
print("Saved data/feature_importance.png")

# ── Save Model ─────────────────────────────────────────────────

joblib.dump(rf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
