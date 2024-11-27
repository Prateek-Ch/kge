import matplotlib.pyplot as plt
import distribution_shift_utils as dsutils

class DistributionAnalysis:
    def __init__(self, checkpoint_path):
        _, self.dataset = dsutils.load_model_and_dataset(checkpoint_path)
        self.current_relation_count_df = dsutils.create_dataframe(dsutils.RELATION_COLUMNS_COUNT)
        self.current_relation_distribution_df = dsutils.create_dataframe(dsutils.RELATION_COLUMNS_DISTRIBUTION)

    def _create_dataframes(self, relation_counts: dict):
        rows_count, rows_distribution = dsutils.counts_and_distribution_rows(relation_counts, self.dataset)
        self.current_relation_count_df = dsutils.append_to_dataframe(self.current_relation_count_df, rows_count)
        self.current_relation_distribution_df = dsutils.append_to_dataframe(self.current_relation_distribution_df, rows_distribution)

    def current_distribution(self):
        relation_counts = {}
        for split in ["train", "test", "valid"]:
            triples = self.dataset.split(split)
            counts_per_relation = dsutils.get_triples_counts_per_group(triples, group_type="relation")
            counts_per_relation["total_count"] = len(triples)
            relation_counts[split] = counts_per_relation

        self._create_dataframes(relation_counts)

        print("\nCurrent Triple Count:")
        print(self.current_relation_count_df)
        print("\nCurrent Triple Distribution:")
        print(self.current_relation_distribution_df)

    def plot_relation_distribution(self):
        # Plotting the relation distribution for Train, Valid, and Test
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Relation Distribution by Split")

        for i, (split, color, title) in enumerate(zip(
            ["Train", "Valid", "Test"], 
            ["skyblue", "lightgreen", "salmon"], 
            ["Training Set", "Validation Set", "Test Set"]
        )):
            split_counts = self.current_relation_count_df[["Relation ID", f"{split} Triple Count"]].sort_values(by="Relation ID")
            max_count = split_counts[f"{split} Triple Count"].max()

            dsutils.plot_bar(
                ax[i], 
                split_counts["Relation ID"], split_counts[f"{split} Triple Count"], 
                title=title, xlabel="Relation ID", ylabel="Triple Count" if i == 0 else "", 
                color=color, ylim=(0, max_count * 1.1)
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

if __name__ == "__main__":
    analysis = DistributionAnalysis('local/experiments/20241125-062127-custom-rescal/checkpoint_best.pt')
    analysis.current_distribution()
    analysis.plot_relation_distribution()
