import pandas as pd
import matplotlib.pyplot as plt
import distribution_shift_utils as dsutils

from collections import defaultdict

class DistributionAnalysis:
    COLUMNS_COUNT = ["Relation ID", "Relation Strings", "Train Triple Count", "Valid Triple Count", "Test Triple Count"]
    COLUMNS_DISTRIBUTION = ["Relation ID", "Relation Strings", "Train Triple Distribution", "Valid Triple Distribution", "Test Triple Distribution"]

    def __init__(self, checkpoint_path):
        _, self.dataset = dsutils.load_model_and_dataset(checkpoint_path)
        self.current_relation_count_df = pd.DataFrame(columns=self.COLUMNS_COUNT)
        self.current_relation_distribution_df = pd.DataFrame(columns=self.COLUMNS_DISTRIBUTION)

    def _create_dataframes(self, relation_counts: dict):
        """Populates DataFrames for counts and distributions based on relation counts."""
        rows_count = []
        rows_distribution = []
        
        for relation_id in relation_counts["train"]:
            if relation_id == "total_count":
                continue

            relation_name = self.dataset.relation_strings(relation_id)
            train_count = relation_counts["train"][relation_id]
            valid_count = relation_counts["valid"][relation_id]
            test_count = relation_counts["test"][relation_id]
            train_total = relation_counts["train"]["total_count"] or 1  # Avoid division by zero
            valid_total = relation_counts["valid"]["total_count"] or 1
            test_total = relation_counts["test"]["total_count"] or 1
            
            rows_count.append({
                "Relation ID": relation_id,
                "Relation Strings": relation_name if relation_name else "",
                "Train Triple Count": train_count,
                "Valid Triple Count": valid_count,
                "Test Triple Count": test_count
            })
            rows_distribution.append({
                "Relation ID": relation_id,
                "Relation Strings": relation_name if relation_name else "",
                "Train Triple Distribution": train_count / train_total,
                "Valid Triple Distribution": valid_count / valid_total,
                "Test Triple Distribution": test_count / test_total
            })

        self.current_relation_count_df = pd.concat([self.current_relation_count_df, pd.DataFrame(rows_count)], ignore_index=True)
        self.current_relation_distribution_df = pd.concat([self.current_relation_distribution_df, pd.DataFrame(rows_distribution)], ignore_index=True)

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

        # Extract data for plotting
        train_counts = self.current_relation_count_df[['Relation ID', 'Train Triple Count']]
        valid_counts = self.current_relation_count_df[['Relation ID', 'Valid Triple Count']]
        test_counts = self.current_relation_count_df[['Relation ID', 'Test Triple Count']]
        
        # Sort by Relation ID for consistency
        train_counts = train_counts.sort_values(by="Relation ID")
        valid_counts = valid_counts.sort_values(by="Relation ID")
        test_counts = test_counts.sort_values(by="Relation ID")
        
        # Get max counts for custom scaling
        max_train = train_counts["Train Triple Count"].max()
        max_valid = valid_counts["Valid Triple Count"].max()
        max_test = test_counts["Test Triple Count"].max()

        # Use the utility function to plot each bar plot
        dsutils.plot_bar(ax[0], 
                train_counts["Relation ID"], train_counts["Train Triple Count"], 
                title="Training Set", xlabel="Relation ID", ylabel="Triple Count", 
                color="skyblue", ylim=(0, max_train * 1.1))
        
        dsutils.plot_bar(ax[1], 
                valid_counts["Relation ID"], valid_counts["Valid Triple Count"], 
                title="Validation Set", xlabel="Relation ID", ylabel="", 
                color="lightgreen", ylim=(0, max_valid * 1.1))
        
        dsutils.plot_bar(ax[2], 
                test_counts["Relation ID"], test_counts["Test Triple Count"], 
                title="Test Set", xlabel="Relation ID", ylabel="", 
                color="salmon", ylim=(0, max_test * 1.1))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

if __name__ == "__main__":
    custom_distribution = DistributionAnalysis('local/experiments/20241118-205700-custom-rescal/checkpoint_best.pt')
    custom_distribution.current_distribution()
    custom_distribution.plot_relation_distribution()