import pandas as pd

from collections import defaultdict
from kge.model import KgeModel
from kge.util.io import load_checkpoint

class CustomDistribution:
    def __init__(self):
        self.checkpoint_path = 'local/experiments/20241021-193745-wnrr-rescal/checkpoint_best.pt'
        self.checkpoint = load_checkpoint(self.checkpoint_path)
        self.model = KgeModel.create_from(self.checkpoint)
        self.current_dist_df = pd.DataFrame(
            columns=["Relation ID", "Relation Strings", "Train Triple Count", "Valid Triple Count"
                     , "Test Triple Count"])
    
    def current_distribution(self):
        # Dictionary to store relation counts for each split
        relation_counts = {
            "train": defaultdict(int),
            "test": defaultdict(int),
            "valid": defaultdict(int)
        }
        
        for key in relation_counts.keys():
            key_split = self.model.dataset.split(key)
            for triple in key_split:
                relation = triple[1].item()
                relation_counts[key][relation] += 1
        
        all_relation_ids = relation_counts["train"].keys()
        for relation_id in all_relation_ids:
            relation_name = self.model.dataset.relation_strings(relation_id)
            self.current_dist_df = self.current_dist_df._append({
                "Relation ID": relation_id,
                "Relation Strings": relation_name if relation_name else "",
                "Train Triple Count": relation_counts["train"][relation_id],
                "Valid Triple Count": relation_counts["valid"][relation_id],
                "Test Triple Count": relation_counts["test"][relation_id]
            }, ignore_index = True)

        print("\n Current Distribution: ")
        print(self.current_dist_df)



if __name__ == "__main__":
    custom_distribution = CustomDistribution()
    custom_distribution.current_distribution()