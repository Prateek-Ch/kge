import torch
import distribution_shift_utils as dsutils

import random
import os

class CustomDistribution:
    def __init__(self, checkpoint_path, target_distribution):
        _, self.dataset = dsutils.load_model_and_dataset(checkpoint_path)
        self.target_distribution = target_distribution

    def sample_training_validation_set(self):
        """Samples the training and validation set based on the target distribution."""

        all_triples = torch.cat((self.dataset.split("train"), self.dataset.split("valid")),dim=0)
        relation_triples = dsutils.group_triples(all_triples, group_type="relation")

        # Calculate target number of triples per relation for training/validation
        target_counts = {relation: int(len(all_triples) * prob) for relation, prob in zip(relation_triples.keys(), self.target_distribution)}

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


#TODO: 
# Improve the code quality.
# Make all these functions reusable. Especially the plot one
# Some code has overlap with the subgroups.py code as well. See how to modify stuff in both so we can use the reusable components.

if __name__ == "__main__":
    custom_distribution = CustomDistribution('local/experiments/20241119-072058-wnrr-rescal/checkpoint_best.pt', 
                                             [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.05])
    train_triples, valid_triples = custom_distribution.sample_training_validation_set()
    dataset_folder = "data/custom_dataset"
    os.makedirs(dataset_folder, exist_ok=True)

    dsutils.save_triples(train_triples, os.path.join(dataset_folder, "train.txt"))
    dsutils.save_triples(valid_triples, os.path.join(dataset_folder, "valid.txt"))
    dsutils.save_triples(custom_distribution.dataset.split("test"), os.path.join(dataset_folder, "test.txt"))

    print("Custom dataset saved.")
