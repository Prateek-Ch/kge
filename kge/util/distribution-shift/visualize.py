import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    df = pd.read_csv(file_path, header=None, names=columns, engine="python")
    df["Shift"] = shift
    df["Model"] = model
    data.append(df)

# Combine data into one DataFrame
data = pd.concat(data, ignore_index=True)

data["Shift"] = data["Shift"].astype(float)

plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x="Shift", y="MR", hue="Relation Type", marker="o")
plt.title("MR Across Noise Levels for Different Relation Types")
plt.xlabel("Noise Level (Shift)")
plt.ylabel("Mean Rank (MR)")
plt.legend(title="Relation Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
