import torch
import random
import os
import distribution_shift_utils as dsutils


class CustomDistribution:
    def __init__(self, checkpoint_path, noise_magnitude):
        _, self.dataset = dsutils.load_model_and_dataset(checkpoint_path)
        self.noise_magnitude = noise_magnitude

    def calculate_target_distribution(self, triples):
        """Calculate a slightly shifted target distribution by adding random noise."""

        relation_triples = dsutils.group_triples(triples, group_type="relation")
        original_distribution = {relation: 1.0 for relation, _ in relation_triples.items()}
        
        # Add small random noise to the original distribution
        target_distribution = {
            relation: max(0, prob + random.uniform(-self.noise_magnitude, self.noise_magnitude))
            for relation, prob in original_distribution.items()
        }
        
        # Normalize the distribution to ensure it sums to 1
        total = sum(target_distribution.values())
        return {relation: prob / total for relation, prob in target_distribution.items()}

    def sample_training_validation_set(self):
        """Samples the training and validation set based on the target distribution."""

        all_triples = torch.cat((self.dataset.split("train"), self.dataset.split("valid")),dim=0)
        relation_triples = dsutils.group_triples(all_triples, group_type="relation")
        target_distribution = self.calculate_target_distribution(all_triples)

        # Calculate target number of triples per relation
        target_counts = {relation: int(len(all_triples) * prob) for relation, prob in target_distribution.items()}

        # Sample triples for each relation
        train_valid_triples = []
        for relation, count in target_counts.items():
            available_triples = relation_triples[relation]
            sample_size = min(count, len(available_triples) - 1)  # Leave at least one triple
            if sample_size > 0:
                sampled_triples = random.sample(available_triples, sample_size)
                train_valid_triples.extend(sampled_triples)

        # Shuffle and split into train and validation (e.g., 80-20 split)
        random.shuffle(train_valid_triples)
        split_index = int(len(train_valid_triples) * 0.89)
        train_triples = train_valid_triples[:split_index]
        valid_triples = train_valid_triples[split_index:]

        return train_triples, valid_triples

    def sample_test_set(self, train_valid_triples, test_ratio):
        """Samples the test set while maintaining the test-train ratio."""

        train_valid_triples = train_valid_triples.tolist()
        test_set_size = int(len(train_valid_triples) * test_ratio)
        test_triples = random.sample(train_valid_triples, test_set_size)
        remaining_triples = [triple for triple in train_valid_triples if triple not in test_triples]
        return test_triples, remaining_triples


if __name__ == "__main__":
    checkpoint_path = "local/experiments/20241119-072058-wnrr-rescal/checkpoint_best.pt"
    output_folder = "data/custom-dataset-updated"
    noise_magnitude = 0.05

    # Initialize distribution class
    custom_distribution = CustomDistribution(checkpoint_path, noise_magnitude)

    # Create training and validation splits
    train_triples, valid_triples = custom_distribution.sample_training_validation_set()

    # Sample test set to maintain original ratio
    original_test_ratio = len(custom_distribution.dataset.split("test")) / len(
        torch.cat((custom_distribution.dataset.split("train") , custom_distribution.dataset.split("valid"), custom_distribution.dataset.split("test")))
    )
    test_triples, train_valid_triples = custom_distribution.sample_test_set(torch.cat((custom_distribution.dataset.split("train") , custom_distribution.dataset.split("valid"))), original_test_ratio)

    # Save the datasets
    os.makedirs(output_folder, exist_ok=True)
    dsutils.save_triples(train_valid_triples, os.path.join(output_folder, "train.txt"))
    dsutils.save_triples(valid_triples, os.path.join(output_folder, "valid.txt"))
    dsutils.save_triples(test_triples, os.path.join(output_folder, "test.txt"))

    print("Custom dataset saved.")
