import pandas as pd
from collections import defaultdict
from kge.model import KgeModel
from kge.util.io import load_checkpoint

class CustomDistribution:
    COLUMNS_COUNT = ["Relation ID", "Relation Strings", "Train Triple Count", "Valid Triple Count", "Test Triple Count"]
    COLUMNS_DISTRIBUTION = ["Relation ID", "Relation Strings", "Train Triple Distribution", "Valid Triple Distribution", "Test Triple Distribution"]

    def __init__(self, checkpoint_path: str = 'local/experiments/20241021-193745-wnrr-rescal/checkpoint_best.pt'):
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

if __name__ == "__main__":
    custom_distribution = CustomDistribution()
    custom_distribution.current_distribution()