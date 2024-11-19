import torch

from kge.model import KgeModel
from kge.util.io import load_checkpoint
from collections import defaultdict
from kge.job.eval import EvaluationJob

def load_model_and_dataset(checkpoint_path):
    """Load and return the model and dataset"""

    checkpoint = load_checkpoint(checkpoint_path)
    model = KgeModel.create_from(checkpoint)
    return model, model.dataset

def group_triples(triples, group_type):
    """Groups triples by the specified type (subject, relation, or object)."""

    groups = defaultdict(list)
    for triple in triples:
        key = triple[{"subject": 0, "relation": 1, "object": 2}[group_type]].item()
        groups[key].append(triple)
    return groups

def get_triples_counts_per_group(triples, group_type):
        """Counts triples for each subgroup and returns a dictionary."""

        groups = group_triples(triples, group_type)
        return {key: len(triples) for key, triples in groups.items()}

def save_triples(triples, filepath):
        """Save the triples to a specific filepath. Format of the file can be sent along the filepath"""

        with open(filepath, "w") as f:
            for triple in triples:
                f.write("\t".join(map(str, triple.tolist())) + "\n")

def evaluate(model: KgeModel, dataset, triples):
        """Evaluates a batch of triples and returns results such as MRR and Hits@k."""
        
        triples_tensor = torch.stack(triples).to(model.config.get("job.device"))
        eval_job = EvaluationJob.create(model.config, dataset=dataset, model=model)
        eval_job._prepare()
        eval_job.loader = torch.utils.data.DataLoader(
            triples_tensor, batch_size=model.config.get("eval.batch_size"), shuffle=False,
            collate_fn=eval_job._collate,
        )
        return eval_job._run()
        
