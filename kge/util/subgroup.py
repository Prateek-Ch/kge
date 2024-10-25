import torch
from kge.job.eval import EvaluationJob
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from collections import defaultdict

# Load model and data
checkpoint = load_checkpoint('local/experiments/20241021-193745-wnrr-rescal/checkpoint_best.pt')
model = KgeModel.create_from(checkpoint)
test_triples = model.dataset.split("test")

def group_triples(triples, group_type="relation"):
    groups = defaultdict(list)
    for triple in triples:
        if group_type == "subject":
            key = triple[0].item()
        elif group_type == "relation":
            key = triple[1].item()
        elif group_type == "object":
            key = triple[2].item()
        else:
            raise ValueError("Invalid group_type. Choose from 'subject', 'relation', or 'object'.")
        groups[key].append(triple)
    return groups

# Evaluate function to assess performance on a batch of triples
def evaluate(model, triples):
    # Convert list of triples to 2D tensor
    triples_tensor = torch.stack(triples).to(model.config.get("job.device"))

    eval_job = EvaluationJob.create(model.config, dataset=model.dataset, model=model)
    eval_job._prepare()
    
    # Override DataLoader with custom triples tensor
    custom_loader = torch.utils.data.DataLoader(
        triples_tensor, batch_size=model.config.get("eval.batch_size"), shuffle=False,
        collate_fn=eval_job._collate,
    )
    eval_job.loader = custom_loader
    eval_job.result = eval_job._run()
    return eval_job.result

# Main function to evaluate subgroups based on group type
def eval_subgroups(model, triples, group_type):
    groups = group_triples(triples, group_type)
    for key, triples in groups.items():
        print(f"Evaluating {group_type} subgroup with key {key}")
        results = evaluate(model, triples)
        print(f"Results for {group_type} {key}: {results}")

# Execute subgroup evaluation by desired grouping type
eval_subgroups(model, test_triples, group_type="relation")
