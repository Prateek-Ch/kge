import torch
import pandas as pd
import distribution_shift_utils as dsutils

from kge.job.eval import EvaluationJob
from collections import defaultdict

class SubgroupEvaluator:
    def __init__(self, checkpoint_path, group_type):
        # Load model and set the grouping type
        self.model, self.dataset = dsutils.load_model_and_dataset(checkpoint_path)
        self.group_type = group_type
        self.test_triples = self.dataset.split("test")
        self.train_triples = self.dataset.split("train")
        self.relation_per_type = self.dataset.index("relations_per_type")

        self.results_df = pd.DataFrame(
            columns=["Relation Strings", "Test Triple Count", "Train Triple Count", "Relation ID", "Relation Type", "MR", "MRR", "Hits@1", "Hits@3", "Hits@10"]
        )

        self.filtered_results_df = pd.DataFrame(
            columns=["Relation Strings", "Test Triple Count", "Train Triple Count", "Relation ID", "Relation Type", "Filtered MR", "Filtered MRR", "Filtered Hits@1", "Filtered Hits@3", "Filtered Hits@10"]
        )

    def group_triples(self, triples):
        """Groups triples by the specified type (subject, relation, or object)."""
        groups = defaultdict(list)
        for triple in triples:
            if self.group_type == "subject":
                key = triple[0].item()
            elif self.group_type == "relation":
                key = triple[1].item()
            elif self.group_type == "object":
                key = triple[2].item()
            else:
                raise ValueError("Invalid group_type. Choose from 'subject', 'relation', or 'object'.")
            groups[key].append(triple)
        return groups

    def get_train_counts(self):
        """Counts training triples for each subgroup and returns a dictionary."""
        train_groups = self.group_triples(self.train_triples)
        train_counts = {key: len(triples) for key, triples in train_groups.items()}
        return train_counts

    def evaluate(self, triples):
        """Evaluates a batch of triples and returns results such as MRR and Hits@k."""
        
        triples_tensor = torch.stack(triples).to(self.model.config.get("job.device"))

        eval_job = EvaluationJob.create(self.model.config, dataset=self.dataset, model=self.model)
        eval_job._prepare()
        
        custom_loader = torch.utils.data.DataLoader(
            triples_tensor, batch_size=self.model.config.get("eval.batch_size"), shuffle=False,
            collate_fn=eval_job._collate,
        )
        eval_job.loader = custom_loader
        eval_job.result = eval_job._run()

        return eval_job.result

    def eval_subgroups(self):
        """Evaluates and prints results for each subgroup based on the grouping type."""  

        test_groups = self.group_triples(self.test_triples)
        train_counts = self.get_train_counts()

        for key, triples in test_groups.items():
            # Retrieve the relation name for the relation ID if group_type is relation
            if self.group_type == "relation":
                name = self.dataset.relation_strings(key)
            else:
                name = self.dataset.entity_strings(key)

            # Retrieve relation type
            for relation_type, value in self.relation_per_type.items():
                if key in value:
                    key_relation_type = relation_type
                    break

            # Evaluate the subgroup
            results = self.evaluate(triples)

            # Extract metrics
            mr = results["mean_rank"]  # Mean Rank
            mrr = results["mean_reciprocal_rank"]  # Mean Reciprocal Rank
            hits_at_1 = results["hits_at_1"]
            hits_at_3 = results["hits_at_3"]
            hits_at_10 = results["hits_at_10"]
            triple_count = len(triples)
            train_triple_count = train_counts.get(key, 0)
            filtered_mr = results["mean_rank_filtered"]
            filtered_mrr = results["mean_reciprocal_rank_filtered"]
            filtered_hits_at_1 = results["hits_at_1_filtered"]
            filtered_hits_at_3 = results["hits_at_3_filtered"]
            filtered_hits_at_10 = results["hits_at_10_filtered"]

            # Append results to DataFrame
            self.results_df = self.results_df._append({
                "Relation Strings": name if name else "",
                "Test Triple Count": triple_count,
                "Train Triple Count": train_triple_count,
                "Relation ID": key,
                "Relation Type": key_relation_type if key_relation_type else "",
                "MR": mr,
                "MRR": mrr,
                "Hits@1": hits_at_1,
                "Hits@3": hits_at_3,
                "Hits@10": hits_at_10,
            }, ignore_index=True)

            self.filtered_results_df = self.filtered_results_df._append({
                "Relation Strings": name if name else "",
                "Test Triple Count": triple_count,
                "Train Triple Count": train_triple_count,
                "Relation ID": key,
                "Relation Type": key_relation_type if key_relation_type else "",
                "Filtered MR": filtered_mr,
                "Filtered MRR": filtered_mrr,
                "Filtered Hits@1": filtered_hits_at_1,
                "Filtered Hits@3": filtered_hits_at_3,
                "Filtered Hits@10": filtered_hits_at_10,
            }, ignore_index=True)

        print("Evaluation results:")
        print(self.results_df)
        print("\n Filtered Evaluation results:")
        print(self.filtered_results_df)

if __name__ == "__main__":
    evaluator = SubgroupEvaluator(
        checkpoint_path='local/experiments/20241118-205700-custom-rescal/checkpoint_best.pt',
        group_type="relation"  # Change to "subject" or "object" as needed
    )
    evaluator.eval_subgroups()
