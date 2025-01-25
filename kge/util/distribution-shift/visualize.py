import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

shift_files = {
    "0.01": "0.01/subgroup-log.csv",
    "0.05": "0.05/subgroup-log.csv",
    "0.1":  "0.1/subgroup-log.csv",
    "0.2":  "0.2/subgroup-log.csv"
}

# Combine all files into one DataFrame
data = []
columns = [
    "Relation ID", "Relation Strings", "Test Triple Count", "Train Triple Count",
    "Relation Type", "MR", "MRR", "Hits@1", "Hits@3", "Hits@10"
]
for shift, file_path in shift_files.items():
    model = "distmult"
    file_path = f'./results/{model}/{file_path}'
    df = pd.read_csv(file_path, header=0, engine="python", names=columns)
    df["Shift"] = shift
    df["Model"] = model
    data.append(df)

# Combine data into one DataFrame
data = pd.concat(data, ignore_index=True)
pd.set_option('display.max_rows', None)

data["Shift"] = data["Shift"].astype(float)

plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x="Shift", y="MR", hue="Relation Type", marker="o", ci=None)
plt.title("MR Across Noise Levels for Different Relation Types")
plt.xlabel("Noise Level (Shift)")
plt.ylabel("Mean Rank (MR)")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=6))
plt.legend(title="Relation Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

pivot = data.pivot_table(index="Relation Type", columns="Shift", values="MR", aggfunc="mean")
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap of MR Across Noise Levels and Relation Types")
plt.xlabel("Noise Level (Shift)")
plt.ylabel("Relation Type")
plt.show()

noise_level = 0.1
filtered_data = data[data["Shift"] == noise_level]
plt.figure(figsize=(12, 6))
sns.barplot(data=filtered_data, x="Relation Strings", y="MR", hue="Relation Type")
plt.title(f"MR for Relation Types at Noise Level {noise_level}")
plt.xlabel("Relation Strings")
plt.ylabel("MR")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Relation Type")
plt.tight_layout()
plt.show()