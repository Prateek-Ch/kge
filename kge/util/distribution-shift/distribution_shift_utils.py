import torch
import pandas as pd

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

def plot_bar(ax, x, y, title, xlabel, ylabel, color, ylim=None):
    """
    Utility function to create a bar plot on a given axis.

    Parameters:
    - ax: matplotlib axis
    - x: Data for x-axis
    - y: Data for y-axis
    - title: Title of the plot
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    - color: Color for the bars
    - ylim: Tuple specifying y-axis limits (optional)
    """
    ax.bar(x, y, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis='x')


# Dataframes related stuff
# Constants for distribution analysis column names
RELATION_COLUMNS_COUNT = ["Relation ID", "Relation Strings", "Train Triple Count", "Valid Triple Count", "Test Triple Count"]
RELATION_COLUMNS_DISTRIBUTION = ["Relation ID", "Relation Strings", "Train Triple Distribution", "Valid Triple Distribution", "Test Triple Distribution"]

# Constants for subgroups evaluation column names
EVALUATION_COLUMNS = ["Relation ID","Relation Strings", "Test Triple Count", "Train Triple Count", "Relation Type", "MR", "MRR", "Hits@1", "Hits@3", "Hits@10"]
FILTERED_EVALUATION_COLUMNS = ["Relation ID", "Relation Strings", "Test Triple Count", "Train Triple Count", "Relation Type", "Filtered MR", "Filtered MRR", "Filtered Hits@1", "Filtered Hits@3", "Filtered Hits@10"]

# Utility functions
def create_dataframe(columns):
    """Create an empty DataFrame with specified columns."""
    return pd.DataFrame(columns=columns)

def append_to_dataframe(df, rows_data):
    """Append rows to the DataFrame."""
    if isinstance(rows_data, list):
        return pd.concat([df, pd.DataFrame(rows_data)], ignore_index=True)
    elif isinstance(rows_data, dict):
        return pd.concat([df, pd.DataFrame([rows_data])], ignore_index=True)
    else:
        raise ValueError("Invalid row data format. Must be a list or dictionary.")


def calculate_distribution(count, total_count):
    """Safely calculate the distribution."""
    return count / (total_count or 1)  # Avoid division by zero

def counts_and_distribution_rows(relation_counts, dataset):
    """Generate rows for counts and distributions."""
    rows_count = []
    rows_distribution = []

    for relation_id, train_count in relation_counts["train"].items():
        if relation_id == "total_count":
            continue

        relation_name = dataset.relation_strings(relation_id)
        valid_count = relation_counts["valid"].get(relation_id, 0)
        test_count = relation_counts["test"].get(relation_id, 0)
        train_total = relation_counts["train"].get("total_count", 1)
        valid_total = relation_counts["valid"].get("total_count", 1)
        test_total = relation_counts["test"].get("total_count", 1)

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
            "Train Triple Distribution": calculate_distribution(train_count, train_total),
            "Valid Triple Distribution": calculate_distribution(valid_count, valid_total),
            "Test Triple Distribution": calculate_distribution(test_count, test_total)
        })

    return rows_count, rows_distribution
        
