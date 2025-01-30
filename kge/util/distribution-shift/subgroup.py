import distribution_shift_utils as dsutils
import os
import argparse

class SubgroupEvaluator:
    def __init__(self, checkpoint_path, group_type):
        # Load model and set the grouping type
        self.model, self.dataset = dsutils.load_model_and_dataset(checkpoint_path)
        self.group_type = group_type
        self.test_triples = self.dataset.split("test")
        self.train_triples = self.dataset.split("train")
        self.relation_per_type = self.dataset.index("relations_per_type")

        self.results_df = dsutils.create_dataframe(dsutils.EVALUATION_COLUMNS)
        self.filtered_results_df = dsutils.create_dataframe(dsutils.FILTERED_EVALUATION_COLUMNS)

    def eval_subgroups(self, output_dir):
        """Evaluates and saves results for each subgroup based on the grouping type."""
        
        os.makedirs(output_dir, exist_ok=True)

        test_groups = dsutils.group_triples(self.test_triples, self.group_type)
        train_counts = dsutils.get_triples_counts_per_group(self.train_triples, self.group_type)

        for key, triples in sorted(test_groups.items()):
            # Retrieve the relation name for the relation ID if group_type is relation
            if self.group_type == "relation":
                name = self.dataset.relation_strings(key)
            else:
                name = self.dataset.entity_strings(key)

            # Retrieve relation type
            key_relation_type = None
            for relation_type, value in self.relation_per_type.items():
                if key in value:
                    key_relation_type = relation_type
                    break

            # Evaluate the subgroup
            results = dsutils.evaluate(self.model, self.dataset, triples)
            row_data = {
                "Relation ID": key,
                "Relation Strings": name,
                "Test Triple Count": len(triples),
                "Train Triple Count": train_counts.get(key, 0),
                "Relation Type": key_relation_type,
            }
            results_data = {**row_data, **{
                "MR": results["mean_rank"], 
                "MRR": results["mean_reciprocal_rank"],
                "Hits@1": results["hits_at_1"],
                "Hits@3": results["hits_at_3"],
                "Hits@10": results["hits_at_10"]
            }}
            filtered_results_data = {**row_data, **{
                "Filtered MR": results["mean_rank_filtered"],
                "Filtered MRR": results["mean_reciprocal_rank_filtered"],
                "Filtered Hits@1": results["hits_at_1_filtered"],
                "Filtered Hits@3": results["hits_at_3_filtered"],
                "Filtered Hits@10": results["hits_at_10_filtered"]
            }}

            self.results_df = dsutils.append_to_dataframe(self.results_df, results_data)
            self.filtered_results_df = dsutils.append_to_dataframe(self.filtered_results_df, filtered_results_data)

        # Save DataFrames to CSV
        results_csv_path = os.path.join(output_dir, "evaluation_results.csv")
        filtered_results_csv_path = os.path.join(output_dir, "filtered_evaluation_results.csv")

        self.results_df.to_csv(results_csv_path, index=False)
        self.filtered_results_df.to_csv(filtered_results_csv_path, index=False)

        print(f"Evaluation results saved to {results_csv_path}")
        print(f"Filtered evaluation results saved to {filtered_results_csv_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--group_type", type=str, required=True, choices=["relation", "subject", "object"], help="Grouping type")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    args = parser.parse_args()

    evaluator = SubgroupEvaluator(args.checkpoint_path, args.group_type)
    evaluator.eval_subgroups(args.output_dir)
