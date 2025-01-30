import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Define shift levels and models
shift_files = {
    "0.01": "0.01/subgroup-log.csv",
    "0.05": "0.05/subgroup-log.csv",
    "0.1": "0.1/subgroup-log.csv",
    "0.2": "0.2/subgroup-log.csv"
}
dataset = "wn18"
models = ["distmult", "complex"]

# Define column names
columns = [
    "Relation ID", "Relation Strings", "Test Triple Count", "Train Triple Count",
    "Relation Type", "MR", "MRR", "Hits@1", "Hits@3", "Hits@10"
]

# Load data from all models and shifts
data = []
for shift, file_path in shift_files.items():
    for model in models:
        full_path = f'./results/{dataset}/{model}/{file_path}'
        try:
            df = pd.read_csv(full_path, header=0, engine="python", names=columns)
            df["Shift"] = float(shift)  # Ensure Shift is numeric
            df["Model"] = model
            data.append(df)
        except Exception as e:
            print(f"Error reading {full_path}: {e}")

# Combine data into one DataFrame
data = pd.concat(data, ignore_index=True)
pd.set_option('display.max_rows', None)

# Ensure correct data types
data["Shift"] = data["Shift"].astype(float)

# === Plot 1: Line Plot - MR Across Noise Levels for Different Models ===
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x="Shift", y="MR", hue="Model", marker="o", ci=None)
plt.title("MR Across Noise Levels for Different Models")
plt.xlabel("Noise Level (Shift)")
plt.ylabel("Mean Rank (MR)")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=6))
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# === Plot 2: Heatmap - MR Across Noise Levels and Models ===
pivot = data.pivot_table(index="Model", columns="Shift", values="MR", aggfunc="mean")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap of MR Across Noise Levels and Models")
plt.xlabel("Noise Level (Shift)")
plt.ylabel("Model")
plt.show()

# === Plot 3: Bar Plot - MR Comparison at a Specific Noise Level ===
noise_level = 0.1  # Change this for different comparisons
filtered_data = data[data["Shift"] == noise_level]
plt.figure(figsize=(12, 6))
sns.barplot(data=filtered_data, x="Relation Strings", y="MR", hue="Model")
plt.title(f"MR Comparison Across Models at Noise Level {noise_level}")
plt.xlabel("Relation Strings")
plt.ylabel("MR")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Model")
plt.tight_layout()
plt.show()