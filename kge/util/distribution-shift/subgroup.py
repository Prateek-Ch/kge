import distribution_shift_utils as dsutils

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

    def eval_subgroups(self):
        """Evaluates and prints results for each subgroup based on the grouping type."""  

        test_groups = dsutils.group_triples(self.test_triples, self.group_type)
        train_counts = dsutils.get_triples_counts_per_group(self.train_triples, self.group_type)

        for key, triples in sorted(test_groups.items()):
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
            results = dsutils.evaluate(self.model, self.dataset, triples)
            row_data = {
                "Relation ID": key, 
                "Relation Strings": name, "Test Triple Count": len(triples), 
                "Train Triple Count": train_counts.get(key, 0),
                "Relation Type": key_relation_type,
            }
            results_data = {
                **row_data, **{"MR": results["mean_rank"], 
                "MRR": results["mean_reciprocal_rank"], "Hits@1": results["hits_at_1"], 
                "Hits@3": results["hits_at_3"], "Hits@10": results["hits_at_10"]
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

        print("Evaluation results:")
        print(self.results_df)
        print("\nFiltered Evaluation results:")
        print(self.filtered_results_df)

if __name__ == "__main__":
    evaluator = SubgroupEvaluator(
        checkpoint_path='local/experiments/20241125-062127-custom-rescal/checkpoint_best.pt',
        group_type="relation"  # Change to "subject" or "object" as needed
    )
    evaluator.eval_subgroups()
