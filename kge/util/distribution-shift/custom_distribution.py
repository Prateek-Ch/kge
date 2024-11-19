import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import random
import os

class CustomDistribution:
    COLUMNS_COUNT = ["Relation ID", "Relation Strings", "Train Triple Count", "Valid Triple Count", "Test Triple Count"]
    COLUMNS_DISTRIBUTION = ["Relation ID", "Relation Strings", "Train Triple Distribution", "Valid Triple Distribution", "Test Triple Distribution"]

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = load_checkpoint(self.checkpoint_path)
        self.model = KgeModel.create_from(self.checkpoint)
        self.current_relation_count_df = pd.DataFrame(columns=self.COLUMNS_COUNT)
        self.current_relation_distribution_df = pd.DataFrame(columns=self.COLUMNS_DISTRIBUTION)

    def _count_triples_per_relation(self) -> dict:
        """Counts triples per relation in train, test, and validation splits."""
        relation_counts = {"train": defaultdict(int), "test": defaultdict(int), "valid": defaultdict(int)}
        
        for split in relation_counts:
            triples = self.model.dataset.split(split)
            relation_counts[split]["total_count"] = len(triples)
            for triple in triples:
                relation_id = triple[1].item()
                relation_counts[split][relation_id] += 1

        return relation_counts

    def _create_dataframes(self, relation_counts: dict):
        """Populates DataFrames for counts and distributions based on relation counts."""
        rows_count = []
        rows_distribution = []
        
        for relation_id in relation_counts["train"]:
            if relation_id == "total_count":
                continue

            relation_name = self.model.dataset.relation_strings(relation_id)
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
        relation_counts = self._count_triples_per_relation()
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

        # Bar plot for training data
        ax[0].bar(train_counts["Relation ID"], train_counts["Train Triple Count"], color="skyblue")
        ax[0].set_ylim(0, max_train * 1.1)
        ax[0].set_title("Training Set")
        ax[0].set_xlabel("Relation ID")
        ax[0].set_ylabel("Triple Count")
        ax[0].tick_params(axis='x')
        # Bar plot for validation data
        ax[1].bar(valid_counts["Relation ID"], valid_counts["Valid Triple Count"], color="lightgreen")
        ax[1].set_ylim(0, max_valid * 1.1)
        ax[1].set_title("Validation Set")
        ax[1].set_xlabel("Relation ID")
        
        # Bar plot for test data
        ax[2].bar(test_counts["Relation ID"], test_counts["Test Triple Count"], color="salmon")
        ax[2].set_ylim(0, max_test * 1.1)
        ax[2].set_title("Test Set")
        ax[2].set_xlabel("Relation ID")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


    def sample_training_validation_set(self, target_distribution):
        """Samples the training and validation set based on the target distribution."""

        all_triples = torch.cat((self.model.dataset.split("train"), self.model.dataset.split("valid")),dim=0)
        relation_triples = defaultdict(list)

        # Group triples by relation
        for triple in all_triples:
            relation_id = triple[1].item()
            relation_triples[relation_id].append(triple)

        # Calculate target number of triples per relation for training/validation
        target_counts = {relation: int(len(all_triples) * prob) for relation, prob in zip(relation_triples.keys(), target_distribution)}

        # Sample triples for each relation according to target_counts
        train_valid_triples = []
        for relation, count in target_counts.items():
            available_triples = relation_triples[relation]
            # Ensure we don't sample more than available
            sample_size = min(count, len(available_triples))
            sampled_triples = random.sample(available_triples, sample_size)
            train_valid_triples.extend(sampled_triples)

        # Shuffle and split into train and validation (e.g., 80-20 split)
        random.shuffle(train_valid_triples)
        split_index = int(len(train_valid_triples) * 0.8)
        train_triples = train_valid_triples[:split_index]
        valid_triples = train_valid_triples[split_index:]

        return train_triples, valid_triples

    def save_triples(self, triples, filepath):
        with open(filepath, "w") as f:
            for triple in triples:
                f.write("\t".join(map(str, triple.tolist())) + "\n")


#TODO: 
# Improve the code quality.
# Make all these functions reusable. Especially the plot one
# Some code has overlap with the subgroups.py code as well. See how to modify stuff in both so we can use the reusable components.

if __name__ == "__main__":
    custom_distribution = CustomDistribution('local/experiments/20241119-072058-wnrr-rescal/checkpoint_best.pt')
    custom_distribution.current_distribution()
    custom_distribution.plot_relation_distribution()
    # train_triples, valid_triples = custom_distribution.sample_training_validation_set([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05])
    # dataset_folder = "data/custom_dataset"
    # os.makedirs(dataset_folder, exist_ok=True)

    # custom_distribution.save_triples(train_triples, os.path.join(dataset_folder, "train.txt"))
    # custom_distribution.save_triples(valid_triples, os.path.join(dataset_folder, "valid.txt"))
    # custom_distribution.save_triples(custom_distribution.model.dataset.split("test"), os.path.join(dataset_folder, "test.txt"))

    # print("Custom dataset saved.")
