import pandas as pd

INPUT_PATH  = "data/master_training_data.csv"
OUTPUT_PATH = "data/cleaned_master_data.csv"

TARGET = "FORCE_2020_LITHOFACIES_LITHOLOGY"

# All actual wireline log curves in the dataset
ALL_LOG_CURVES = [
    "GR", "CALI", "RDEP", "RMED", "RSHA", "RXO", "RMIC",
    "SP", "DTC", "DTS", "NPHI", "RHOB", "DRHO", "PEF",
    "DCAL", "BS", "SGR"
]

# Numeric facies ID → human-readable lithology name
LITHO_MAP = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}

THRESHOLD = 0.90   # keep only logs where dropping nulls retains >= 90% of rows

# ── Load ───────────────────────────────────────────────────────

print(f"Loading {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Raw shape: {df.shape}")

# Rename target column
df = df.rename(columns={TARGET: "FACIES_ID"})

# Drop rows with no label
before = len(df)
df = df.dropna(subset=["FACIES_ID"])
print(f"Dropped {before - len(df):,} rows with no FACIES_ID")

total_after_label_drop = len(df)

# ── Auto feature selection based on data retention ────────────

print(f"\n--- Log curve retention analysis (threshold: >{THRESHOLD*100:.0f}%) ---")

selected_features = ["DEPT"]   # DEPT is always included (depth index, 0% null)

for col in ALL_LOG_CURVES:
    if col not in df.columns:
        print(f"  {col:8s}  NOT IN DATA — skipped")
        continue

    null_count   = df[col].isnull().sum()
    rows_kept    = total_after_label_drop - null_count
    pct_kept     = rows_kept / total_after_label_drop

    if pct_kept >= THRESHOLD:
        selected_features.append(col)
        print(f"  {col:8s}  {pct_kept*100:5.1f}% kept  ✓ selected")
    else:
        print(f"  {col:8s}  {pct_kept*100:5.1f}% kept  ✗ excluded (too many nulls)")

print(f"\nSelected features: {selected_features}")

# ── Drop rows missing any selected feature ─────────────────────

before = len(df)
df = df.dropna(subset=selected_features)
print(f"Dropped {before - len(df):,} rows with missing feature values")

# ── Add LITHOLOGY name column ──────────────────────────────────

df["LITHOLOGY"] = df["FACIES_ID"].map(LITHO_MAP)

# ── Save ───────────────────────────────────────────────────────

print(f"\nFinal shape:    {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Wells:          {df['WELL_NAME'].nunique()}")
print(f"Facies classes: {sorted(df['FACIES_ID'].unique())}")

# Save selected features list for train.py to read
features_no_dept = [f for f in selected_features if f != "DEPT"]
print(f"\nFeatures for training (excl. DEPT): {features_no_dept}")

df.to_csv(OUTPUT_PATH, index=False)

# Also save selected features list so train.py picks it up automatically
import json
with open("data/selected_features.json", "w") as f:
    json.dump(selected_features, f)

print(f"Saved cleaned data to {OUTPUT_PATH}")
print(f"Saved feature list  to data/selected_features.json")
