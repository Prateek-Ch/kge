import torch
from kge.job.eval import EvaluationJob
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from collections import defaultdict

class SubgroupEvaluator:
    def __init__(self, checkpoint_path, group_type):
        # Load model and set the grouping type
        self.checkpoint = load_checkpoint(checkpoint_path)
        self.model = KgeModel.create_from(self.checkpoint)
        self.group_type = group_type
        self.test_triples = self.model.dataset.split("test")
    
    def group_triples(self):
        """Groups triples in the test set by the specified type (subject, relation, or object)."""

        groups = defaultdict(list)
        for triple in self.test_triples:
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

    def evaluate(self, triples):
        """Evaluates a batch of triples and returns results such as MRR and Hits@k."""

        triples_tensor = torch.stack(triples).to(self.model.config.get("job.device"))

        eval_job = EvaluationJob.create(self.model.config, dataset=self.model.dataset, model=self.model)
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

        groups = self.group_triples()
        for key, triples in groups.items():
            print(f"Evaluating {self.group_type} subgroup with key {key}")
            self.evaluate(triples)

evaluator = SubgroupEvaluator(
    checkpoint_path='local/experiments/20241021-193745-wnrr-rescal/checkpoint_best.pt',
    group_type="relation"  # Change to "subject" or "object" as needed
)
evaluator.eval_subgroups()
