from kge.model import KgeModel
from kge.util.io import load_checkpoint
from collections import defaultdict

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
        
