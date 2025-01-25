import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

shift_files = {
    "0.01": "0.01/subgroup-log.txt",
    "0.05": "0.05/subgroup-log.txt",
    "0.1":  "0.1/subgroup-log.txt",
    "0.2":  "0.2/subgroup-log.txt"
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
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=columns, engine="python")
    df["Shift"] = shift
    df["Model"] = model
    data.append(df)

# Combine data into one DataFrame
data = pd.concat(data, ignore_index=True)
print(data)

# # Step 2: Visualization
# # Convert columns to appropriate types
# data["Shift"] = data["Shift"].astype(float)

# # Plot 1: Trend Line Plot for MR across shifts
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=data, x="Shift", y="MR", hue="Model", marker="o")
# plt.title("MR Across Shifts for Different Models")
# plt.xlabel("Shift")
# plt.ylabel("Mean Rank (MR)")
# plt.legend(title="Model")
# plt.grid(True)
# plt.show()

# # Plot 2: Box Plot for MR Distribution Across Shifts
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=data, x="Shift", y="MR", hue="Model")
# plt.title("MR Distribution Across Shifts")
# plt.xlabel("Shift")
# plt.ylabel("Mean Rank (MR)")
# plt.legend(title="Model")
# plt.grid(True)
# plt.show()

# # Plot 3: Heatmap of Aggregated MR Values
# heatmap_data = data.groupby(["Model", "Shift"]).mean()["MR"].unstack()
# plt.figure(figsize=(8, 6))
# sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Mean Rank (MR)"})
# plt.title("Heatmap of Mean Rank (MR) by Shift and Model")
# plt.xlabel("Shift")
# plt.ylabel("Model")
# plt.show()
