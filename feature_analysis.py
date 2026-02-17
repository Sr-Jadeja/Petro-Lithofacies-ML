import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_features(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Identify "Classical" features
    # For Lithology, we usually want: Gamma Ray, Resistivity, Density, Neutron, Photoelectric.
    # In your specific well, we have: CALI, RDEP, RMED, etc.
    
    # Let's look at the correlation between numerical logs
    # This shows which logs provide similar information (redundancy)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Log-to-Log Correlation Matrix")
    plt.savefig("data/correlation_matrix.png") # Saving visualization for resume/readme
    print("Correlation matrix saved to data/correlation_matrix.png")

    # 2. Visualize the Target Distribution
    plt.figure(figsize=(10, 6))
    df['LITHOLOGY'].value_counts().plot(kind='bar')
    plt.title("Distribution of Rock Types in Well 15/9-13")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/lithology_distribution.png")
    print("Lithology distribution plot saved.")

    return df

if __name__ == "__main__":
    well_data = analyze_features("data/cleaned_well_data.csv")