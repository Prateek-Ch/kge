import torch
from kge.job.eval import EvaluationJob
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from collections import defaultdict

checkpoint = load_checkpoint('local/experiments/20241021-193745-wnrr-rescal/checkpoint_best.pt')
model = KgeModel.create_from(checkpoint)
test_triples  = model.dataset.split("test")

relation_groups = defaultdict(list)

for triple in test_triples:
    relation = triple[1].item()
    relation_groups[relation].append(triple)

def evaluate_group(model, triples):
    # Convert the list of triples to a tensor
    triples_tensor = torch.stack(triples).to(model.config.get("job.device"))

    eval_job = EvaluationJob.create(model.config, dataset=model.dataset, model=model)
    # Prepare and run the evaluation
    eval_job._prepare()
    # Override the loader with the custom DataLoader
    custom_loader = torch.utils.data.DataLoader(
        triples_tensor, batch_size=model.config.get("eval.batch_size"), shuffle=False,
        collate_fn=eval_job._collate,
    )
    eval_job.loader = custom_loader
    eval_job.result = eval_job._run()
    
    # Return the evaluation results (including MRR, Hits@k)
    return eval_job.result

for relation, triples in relation_groups.items():
    print(f"Evaluating relation: {relation}")
    results = evaluate_group(model, relation_groups[relation])
